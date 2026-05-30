// SPDX-License-Identifier: AGPL-3.0-only

//! Batched K=3 verify (Phase IIb).
//!
//! Dispatches N concurrent sequences' K=3 verify in a single forward pass.
//! Models the structure of `verify_c.rs` (single-seq K=3) but extends the
//! batch axis: `N*3` positions go through one layer loop, the SSM kernels
//! use the per-batch slot pointer arrays staged by Phase IIb's
//! `stage_slot_ptrs_dispatch`, and the attention layers use
//! `decode_multi_seq` with `num_seqs = N*K` and per-position metadata.
//!
//! Per-seq accept-count computation and SSM-state commit happen in the
//! scheduler (`mtp_step.rs`) after this returns — this dispatcher is
//! purely the forward pass + sampling. The kernels write per-token SSM
//! intermediates into each seq's own pool slots so the commit can pick
//! the right intermediate per accept-count without further bookkeeping.
//!
//! When the model PTX module lacks the `_batched` SSM kernel entry
//! points, this path bails and the caller falls back to the per-seq
//! sequential verify loop. The conv1d_l2norm path uses the model
//! generic `causal_conv1d_update_l2norm_batched` from Phase IIa; the
//! SSM K=3 verify path uses `gated_delta_rule_wy3_batched`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use super::slot_ptrs::SsmPtrKind;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Dispatch one batched K=3 verify across N concurrent sequences.
    /// `per_seq_tokens[i]` is the `[last_accepted, draft_1, draft_2]`
    /// token triple for seq `i`. Returns `Vec<[u32; 3]>` parallel to
    /// `seqs` — caller is responsible for per-seq accept-count
    /// computation and SSM-state commit.
    pub(in crate::model) fn decode_verify_batched_k3_dispatch(
        &self,
        per_seq_tokens: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 3]>> {
        let n = seqs.len();
        assert_eq!(per_seq_tokens.len(), n);
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = 3usize;
        let total = n * k;

        // The batched-pool wy3 kernel and the batched conv1d_l2norm
        // kernel are required for this path. If either is missing (e.g.
        // the model's PTX module hasn't picked up Phase IIa yet), bail
        // so the caller falls back to N sequential single-seq verifies.
        // We don't try to be clever about partial-batched fallback —
        // either the whole batch goes through the batched path or none
        // does.
        if !self.has_batched_verify_k3_kernels() {
            bail!(
                "decode_verify_batched_k3: required _batched SSM kernels \
                 not loaded for this model. Caller should fall back to \
                 per-seq decode_verify_graphed_k3."
            );
        }

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() { 4usize } else { 2usize };

        // Pre-verify async checkpoint copy for every seq in the batch.
        // The single-seq path does this for one slot; for the batch we
        // do it per-seq before the layer loop. The commits all queue on
        // `self.secondary_stream`, so the next `sync_secondary` call in
        // the scheduler ordering catches them all.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async_dispatch(seq)?;
        }

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: embed N*K tokens contiguously ──
        // Layout: [seq_0_tok_0, seq_0_tok_1, seq_0_tok_2, seq_1_tok_0, ...]
        // i.e. flat batched-K order — same layout the K=3 kernels
        // expect at qk_stride/v_stride = conv_dim/dim.
        for (i, toks) in per_seq_tokens.iter().enumerate() {
            for t in 0..k {
                self.embed(toks[t], hidden.offset((i * k + t) * h * fp32), stream)?;
            }
        }

        // ── Phase 1b: allocate KV blocks for each seq's K new positions ──
        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            for t in 0..k {
                let pos = seq.seq_len + t;
                let blocks_needed = (pos / bs) + 1;
                ensure_blocks_through_decode(
                    seq,
                    blocks_needed - 1,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }

        // ── Phase 1c: upload attention metadata for N*K positions ──
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;

        // positions: u32[N*K]
        let mut positions: Vec<u32> = Vec::with_capacity(total);
        for seq in seqs.iter() {
            for t in 0..k {
                positions.push((seq.seq_len + t) as u32);
            }
        }
        self.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(positions.as_ptr() as *const u8, total * 4)
            },
            meta_base,
            stream,
        )?;

        // slots: i64[N*K] — physical block_idx * block_size + offset
        let mut slots: Vec<i64> = Vec::with_capacity(total);
        for seq in seqs.iter() {
            for t in 0..k {
                let pos = seq.seq_len + t;
                let block_idx = pos / bs;
                let block_offset = pos % bs;
                let phys = seq.physical_block_for(block_idx).unwrap_or(0);
                slots.push((phys as i64) * (bs as i64) + (block_offset as i64));
            }
        }
        self.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(slots.as_ptr() as *const u8, total * 8)
            },
            meta_base.offset(256),
            stream,
        )?;

        // seq_lens: i32[N*K]
        let mut seq_lens: Vec<i32> = Vec::with_capacity(total);
        for seq in seqs.iter() {
            for t in 0..k {
                seq_lens.push((seq.seq_len + t + 1) as i32);
            }
        }
        self.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, total * 4)
            },
            meta_base.offset(512),
            stream,
        )?;

        // block_table: i32[N*K, max_blocks] — for batch pos b, this is
        // the parent seq's block_table padded to max_blocks.
        let bt_total = total * mb;
        let mut bt_buf: Vec<i32> = vec![0; bt_total];
        for (i, seq) in seqs.iter().enumerate() {
            for t in 0..k {
                let row = i * k + t;
                for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + j] = block as i32;
                }
            }
        }
        self.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, bt_total * 4)
            },
            meta_base.offset(768),
            stream,
        )?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: total as u32,
        };

        // Graphs are intentionally NOT used here yet. The slot tuple
        // (n, [slot_0, slot_1, ..., slot_{n-1}]) makes the capture key
        // combinatorial across batch compositions, and Phase IIb's
        // first cut prefers correctness over capture caching.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,

            slot_ptrs_host_pinned: Some(self.slot_ptrs_host_pinned),

            slot_ptrs_buf: Some(self.slot_ptrs_buf),
        };

        // ── Phase 2: layer loop over N*K positions ──
        //
        // SSM layers: stage the per-batch slot pointer arrays (main
        // h_state + 2 K=3 intermediates + conv_state) and dispatch the
        // batched kernels with batch=N.
        // Attention layers: route through decode_multi_seq with
        // num_seqs=N*K and per-position metadata.
        let seq_lens_for_attn: Vec<usize> = seqs
            .iter()
            .flat_map(|s| (0..k).map(move |t| s.seq_len + t))
            .collect();
        let block_tables_for_attn: Vec<Vec<u32>> = seqs
            .iter()
            .flat_map(|s| std::iter::repeat(s.block_table.clone()).take(k))
            .collect();

        let mut ssm_layer_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            if layer_type == LayerType::FullAttention {
                // Attention layers use decode_multi_seq with N*K virtual
                // single-token decodes — matches the single-seq verify
                // path's treatment of K tokens as K dummy seqs, just
                // scaled up across N batch sequences.
                let mut dummy_states: Vec<Box<dyn LayerState>> = (0..total)
                    .map(|_| layer.alloc_state(self.gpu.as_ref()))
                    .collect::<Result<_>>()?;
                let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                    dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    total,
                    &mut refs,
                    &mut kv_cache,
                    &seq_lens_for_attn,
                    &block_tables_for_attn,
                    &ctx,
                    stream,
                )?;
            } else {
                // SSM layer (LinearAttention). Stage the slot pointer
                // arrays for this layer, then call the batched kernels
                // directly — bypassing the per-layer single-seq
                // `decode_batched` so we get true batching at the
                // kernel level.
                let slots_vec: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
                let h_state_ptrs = self.stage_slot_ptrs_dispatch(
                    SsmPtrKind::HState,
                    ssm_layer_idx,
                    &slots_vec,
                    stream,
                )?;
                let conv_state_ptrs = self.stage_slot_ptrs_dispatch(
                    SsmPtrKind::ConvState,
                    ssm_layer_idx,
                    &slots_vec,
                    stream,
                )?;
                let h_inter0_ptrs = self.stage_slot_ptrs_dispatch(
                    SsmPtrKind::HInter(0),
                    ssm_layer_idx,
                    &slots_vec,
                    stream,
                )?;
                let h_inter1_ptrs = self.stage_slot_ptrs_dispatch(
                    SsmPtrKind::HInter(1),
                    ssm_layer_idx,
                    &slots_vec,
                    stream,
                )?;

                // Delegate to a layer-side helper that knows how to run
                // its own conv1d_l2norm_batched + gdn_decode_wy3_batched
                // pair for this layer's weights. Layers without the
                // batched-verify-K3 implementation return an error,
                // which the caller treats as "bail and fall back to
                // per-seq sequential verify."
                layer.verify_batched_k3(
                    hidden,
                    residual,
                    n as u32,
                    h_state_ptrs,
                    conv_state_ptrs,
                    h_inter0_ptrs,
                    h_inter1_ptrs,
                    &mut kv_cache,
                    &ctx,
                    stream,
                )?;

                ssm_layer_idx += 1;
            }
        }

        // ── Phase 3: final norm, lm_head, argmax over N*K positions ──
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            total as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        self.lm_head_batched(normed, total as u32, stream)?;

        let vocab = self.config.vocab_size;
        let argmax_out = self.buffers.scratch();
        for t in 0..total {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            let out_t = argmax_out.offset(t * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                out_t,
                vocab as u32,
                stream,
            )?;
        }

        // ── Phase 4: D2H copy of N*K sampled tokens ──
        let mut buf = vec![0u8; total * 4];
        self.gpu.copy_d2h(argmax_out, &mut buf)?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * k * 4;
            let tok0 = u32::from_le_bytes([
                buf[base],
                buf[base + 1],
                buf[base + 2],
                buf[base + 3],
            ]);
            let tok1 = u32::from_le_bytes([
                buf[base + 4],
                buf[base + 5],
                buf[base + 6],
                buf[base + 7],
            ]);
            let tok2 = u32::from_le_bytes([
                buf[base + 8],
                buf[base + 9],
                buf[base + 10],
                buf[base + 11],
            ]);
            out.push([tok0, tok1, tok2]);
        }

        // Bookkeeping for each seq — matches the single-seq path's
        // `seq.tokens.push(t); seq.seq_len += k;` at the end.
        for (toks, seq) in per_seq_tokens.iter().zip(seqs.iter_mut()) {
            for &t in toks {
                seq.tokens.push(t);
            }
            seq.seq_len += k;
        }

        Ok(out)
    }

    /// Returns true when the model's PTX module set has the
    /// `_batched` SSM kernel entries required by the K=3 batched verify
    /// path. The required kernels are landed by Phase IIa
    /// (`gated_delta_rule_wy3_batched` for the SSM step;
    /// `causal_conv1d_update_l2norm_batched` for the conv step).
    fn has_batched_verify_k3_kernels(&self) -> bool {
        // We can't directly query the qwen3.6 layer's batched kernel
        // handles from here (they live on the SSM layer struct, not on
        // the model). Instead, peek at the first SSM layer's
        // `verify_batched_k3_supported` flag via the trait. Layers that
        // don't override the default return false → we bail.
        for layer in self.layers.iter() {
            if self
                .config
                .layer_type(0) // probe first layer (model has uniform SSM layer kind)
                == LayerType::LinearAttention
                && layer.verify_batched_k3_supported()
            {
                return true;
            }
            // Continue scanning until first LinearAttention.
        }
        false
    }
}
