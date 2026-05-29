// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5+6+7+8 — last-chunk finalization:
//!   • final RMS-norm on the last token's hidden state
//!   • LM head → logits buffer
//!   • diagnostic dumps (long-context / Gemma4 paths)
//!   • prefix-cache insert + Marconi snapshot save (with reclaim retry)
//!   • DFlash ctx-len bookkeeping

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        logits_slot_idx: usize,
        buffers: &spark_runtime::buffers::BufferArena,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_b_finalize_last_at(
            tokens,
            seq,
            kv_cache,
            chunk_start,
            chunk_len,
            proc_count,
            0,
            logits_slot_idx,
            buffers,
            stream,
        )
    }

    /// Q12 Path B: stream-offset-aware finalize for the kernel-batched
    /// orchestrator. `hidden_stream_offset_tokens` is `b * chunk_len`
    /// where `b` is the stream's index in the batched dispatch.
    /// `logits_slot_idx` is the destination slot in `buffers.logits()` —
    /// non-zero only when multiple concurrent is_last_chunk prefill streams
    /// are batched into the same dispatch (see `prefill_batch_chunk_dispatch`).
    pub(in crate::model) fn prefill_b_finalize_last_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        hidden_stream_offset_tokens: usize,
        logits_slot_idx: usize,
        buffers: &spark_runtime::buffers::BufferArena,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = buffers.hidden_states();
        let bs = kv_cache.block_size();

        // ── 6. Final norm on LAST token only ──
        let last_token_offset = hidden_stream_offset_tokens + proc_count - 1;
        let last_hidden = hidden.offset(last_token_offset * h * fp32);
        let normed = buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            last_hidden,
            &self.final_norm,
            normed,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // Diagnostic: post-norm hidden state
        if (chunk_start + chunk_len) > 16384
            || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
        {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_bf16(normed, h.min(16))?;
            tracing::warn!(
                "DIAG post-norm: norm={norm:.4} first2={:.4?}",
                &vals[..2.min(vals.len())]
            );
        }

        // ── 7. LM head on last token → logits ──
        let logits_ptr = self.lm_head(normed, buffers, logits_slot_idx, stream)?;

        // Diagnostic: logits stats — read from the actual slot lm_head wrote to.
        if (chunk_start + chunk_len) > 16384
            || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
        {
            self.gpu.synchronize(stream)?;
            let n_logits = self.config.vocab_size;
            let mut buf = vec![0u8; n_logits * 2];
            self.gpu.copy_d2h(logits_ptr, &mut buf)?;
            let logit_vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let nan_count = logit_vals.iter().filter(|v| v.is_nan()).count();
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| {
                logit_vals[b]
                    .partial_cmp(&logit_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
            tracing::warn!(
                "DIAG logits[0..{}]: max={max:.4} min={min:.4} nan={nan_count} top5={top5:?}",
                n_logits,
            );
        }

        // ── 8. Insert into prefix cache + Marconi snapshot ──
        //
        // EarlyReturn warm-cache case: `proc_count == 1` together with a
        // non-zero `marconi_skip_to` means we restored a snapshot for the
        // full prompt and then re-ran the last token through the decode
        // kernel just to populate `hidden[0]` for `lm_head`. The decode
        // kernel ADVANCED the SSM state by one step (it cannot read h
        // without also writing the next h), so the live state is now
        // h_(L) when it should be h_(L-1). Saving a snapshot here would
        // persist that drift; the next warm-cache hit would restore the
        // drifted state and advance to h_(L+1), and so on. Empirically
        // (2026-05-29 T=0 single-prompt sanity test): each warm iteration
        // produced slightly different outputs from the previous and after
        // ~4 iterations responses degenerated into 2-token EOS-suppressed
        // stops. The cold-prefill snapshot is correct and can be restored
        // indefinitely; skip the save here so it survives.
        let is_warm_cache_last_token =
            proc_count == 1 && seq.marconi_skip_to > 0 && self.ssm_snapshots.is_enabled();
        if self.ssm_snapshots.is_enabled() && !is_warm_cache_last_token {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            if let Some(snap_id) = snap_result {
                if self.tokens_have_vision_pad(tokens) {
                    self.ssm_snapshots.free(snap_id);
                } else {
                    // Diagnostic: same first-32-tokens FNV hash as the restore
                    // side logs, so save/restore can be correlated in one grep.
                    let prompt_hash = {
                        let mut h: u64 = 0xcbf29ce484222325;
                        for &t in tokens.iter().take(32) {
                            h ^= t as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                        h
                    };
                    tracing::info!(
                        target: "atlas::lockprof",
                        "ssm_snapshot SAVE snap_id={} token_count={} session_hash=0x{:x} prompt_hash32=0x{:x} ssm_slot={}",
                        snap_id, tokens.len(), seq.session_hash, prompt_hash, seq.slot_idx,
                    );
                    tracing::info!(
                        "Saved SSM snapshot {} for {} tokens ({} blocks) [chunk]",
                        snap_id,
                        tokens.len(),
                        seq.block_table.len(),
                    );
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                    );
                    super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                    if let Some(old) = displaced {
                        tracing::info!(
                            target: "atlas::lockprof",
                            "ssm_snapshot FREE_DISPLACED snap_id={} (replaced by snap_id={})",
                            old, snap_id,
                        );
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                );
                super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
            }
        } else if !self.tokens_have_vision_pad(tokens) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
            );
            super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }

        // DFlash: advance ctx_len after the LAST chunk of chunked prefill.
        self.update_dflash_ctx_len_after_prefill(seq, chunk_start, chunk_len)?;

        // Return what lm_head actually wrote to. For mixed-batch prefill on
        // a non-default stream, `buffers` is `secondary_buffers`, so the
        // sampler must read from `secondary_buffers.logits()` — NOT
        // `self.decode_logits_ptr()` which hardcodes the primary arena and
        // would alias decode's logits buffer.
        Ok(logits_ptr)
    }
}
