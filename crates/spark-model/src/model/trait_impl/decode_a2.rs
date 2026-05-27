// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! `TransformerModel::decode_batch_dispatch` — hoisted from `decode_a.rs`
//! to keep that file under the 500 LoC cap.
//!
//! Single entry point preserves the original control flow 1:1: special-case
//! n=1 and EP, otherwise pad to the nearest captured graph size, build a
//! `ForwardContext`, dispatch through `decode_multi_seq` for each layer,
//! and run final norm + per-seq LM-head GEMVs.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    pub(super) fn decode_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        assert_eq!(n, seqs.len(), "tokens.len() must equal seqs.len()");

        // Single-sequence: delegate to decode() which uses CUDA graphs.
        // decode_batch disables graphs for n≥2 (SSM state pointer staleness),
        // but n=1 is safe and benefits from graph replay (2x throughput).
        if n == 1 {
            self.decode(tokens[0], seqs[0], stream)?;
            return Ok(self.decode_logits_ptr());
        }

        // EP mode: use per-sequence decode() to match the worker's batch size.
        if self.comm.is_some() {
            for i in 0..n {
                self.decode(tokens[i], seqs[i], stream)?;
            }
            return Ok(self.decode_logits_ptr());
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STREAM HAZARD: the original `stream` parameter is shadowed
        // below to `default_stream`, because the n≥2 graph capture/
        // replay must happen on a fixed stream. But the caller (see e.g.
        // the default `mixed_forward_batch` impl in `traits/model.rs`)
        // continues issuing GPU ops on its own `stream` parameter
        // immediately after this function returns — writing to the same
        // `self.buffers.*` singletons (hidden_states, residual, logits)
        // that our queued work on default_stream is still using.
        //
        // `prefill_chunk_dispatch` (prefill_b.rs:71-77) respects the
        // caller's stream; `decode_batch_dispatch` silently switches.
        // Two streams + shared GPU buffers + no inter-stream sync
        // = race + CUDA_ERROR_ILLEGAL_ADDRESS the moment the next
        // caller op starts dispatching on the original stream while
        // our decode kernels are still running on default_stream.
        // This is the trigger for the
        // `phase_continue_prefills::run_batched_mixed: Mixed-batch
        // forward error (n_decode=N, n_prefill=M): cuMemcpyHtoDAsync_v2
        // failed: status 700` crash on hybrid Mamba-MoE models.
        //
        // Preserve the caller's stream so we can post a cross-stream
        // event handshake at exit (see the matching block before the
        // `Ok(...)` return).
        let caller_stream = stream;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad to nearest captured graph size [2, 4, 8]
        let padded_n = [2, 4, 8].iter().copied().find(|&s| s >= n).unwrap_or(n);

        // ── Phase 1: Pre-graph (runs every step, NOT captured) ──

        // 1a. Embed active tokens into hidden[0..n)
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(i * h * fp32), stream)?;
        }

        // 1b. Zero padding hidden[n..padded_n)
        for i in n..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }

        // 1c. Allocate KV blocks for active sequences
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1d. Upload metadata with fixed stride (active + padding)
        let metadata = self.upload_batch_metadata_fixed(seqs, padded_n, &mut kv_cache, stream)?;

        // CUDA graphs DISABLED for multi-sequence decode: SSM state pointers
        // (h_state, conv_state) are baked into per-seq kernel args of
        // gdn_decode/conv1d_update at capture time. When batch composition
        // changes (sequences finish, swap_remove reorders), the graph
        // replays with stale pointers and corrupts SSM state across seqs.
        // Attention metadata uses fixed device addresses and is safe.
        // n==1 still uses self.decode()'s correct graph cache.
        let use_graphs = false;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
        };

        // ── Phase 2: CUDA graph lookup / capture ──
        let mut graphs = if use_graphs {
            Some(self.batch_decode_graphs.lock())
        } else {
            None
        };

        if let Some(ref graphs) = graphs
            && let Some(&graph) = graphs.get(&padded_n)
        {
            // Graph exists — replay (kernels use updated metadata + SSM pool addresses)
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }

            // ── Phase 3: Post-graph (update sequence state) ──
            for (i, seq) in seqs.iter_mut().enumerate() {
                seq.tokens.push(tokens[i]);
                seq.seq_len += 1;
            }
            return Ok(self.decode_logits_ptr());
        }
        {
            // First time for this padded_n — capture a new graph (or run eagerly for EP).
            // Build layer states for all padded_n sequences (real + dummy padding).
            let seq_lens: Vec<usize> = (0..padded_n)
                .map(|i| if i < n { seqs[i].seq_len } else { 0 })
                .collect();
            let block_tables: Vec<Vec<u32>> = (0..padded_n)
                .map(|i| {
                    if i < n {
                        seqs[i].block_table.clone()
                    } else {
                        vec![self.dummy_kv_block]
                    }
                })
                .collect();

            // Extract real layer_states from sequences
            let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = seqs
                .iter_mut()
                .map(|s| std::mem::take(&mut s.layer_states))
                .collect();

            // Build dummy layer_states for padding positions. Use the
            // dedicated `dummy_slot()` so pad SSM kernel writes can never
            // collide with another claimed sequence's pool memory if the
            // scheduler invariant ("active occupies contiguous slots
            // [0..n)") ever drifts.
            let dummy_ssm_slot = self.ssm_pool.dummy_slot();
            for _pad_pos in n..padded_n {
                let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
                let mut ssm_idx = 0usize;
                for (li, layer) in self.layers.iter().enumerate() {
                    if self.config.layer_type(li) == LayerType::LinearAttention {
                        dummy.push(Box::new(SsmLayerState {
                            h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                            conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                            h_state_checkpoint: None,
                            conv_state_checkpoint: None,
                            h_state_intermediates: Vec::new(),
                            conv_state_intermediates: Vec::new(),
                        }));
                        ssm_idx += 1;
                    } else {
                        dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                    }
                }
                all_layer_states.push(dummy);
            }

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // CONC_HSD: per-seq hidden-state dump diagnostic. Logs first 4 FP32
            // hidden values for each seq after each layer to localize where
            // pos>=1 diverges from pos 0 in concurrent batched decode.
            let conc_hsd = std::env::var("ATLAS_CONC_HSD").is_ok_and(|v| v == "1" || v == "true")
                && padded_n >= 2
                && self.comm.is_none();
            let dump_hidden = |label: &str, stream: u64| -> Result<()> {
                if !conc_hsd {
                    return Ok(());
                }
                self.gpu.synchronize(stream)?;
                let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(padded_n);
                for i in 0..padded_n {
                    let mut buf = vec![0u8; 4 * 4]; // 4 FP32 values
                    let _ = self.gpu.copy_d2h(hidden.offset(i * h * fp32), &mut buf);
                    let vals: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    bufs.push(vals);
                }
                let pretty: Vec<String> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("s{i}=[{:.4},{:.4},{:.4},{:.4}]", v[0], v[1], v[2], v[3]))
                    .collect();
                tracing::info!("CONC_HSD {label}: {}", pretty.join(" "));
                Ok(())
            };

            dump_hidden("post_embed", stream)?;

            // Layer loop for padded_n sequences
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &ctx,
                    stream,
                )?;
                if conc_hsd {
                    let _ = dump_hidden(&format!("after_L{:02}", layer_idx), stream);
                }
            }

            // Final norm [padded_n, H]
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                padded_n as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head: padded_n sequential GEMVs (weights in L2 after first)
            let logits = self.buffers.logits();
            let v = self.config.vocab_size;
            for i in 0..padded_n {
                let normed_i = normed.offset(i * h * bf16);
                let logits_i = logits.offset(i * v * bf16);
                if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                    ops::w4a16_gemv(
                        self.gpu.as_ref(),
                        self.w4a16_gemv_kernel,
                        normed_i,
                        nvfp4,
                        logits_i,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        self.gpu.as_ref(),
                        self.dense_gemv_kernel,
                        normed_i,
                        &self.lm_head_weight,
                        logits_i,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!("Captured CUDA graph for batch size {padded_n}");
                    if let Some(ref mut g) = graphs {
                        g.insert(padded_n, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }

            // Restore real layer_states to sequences (dummy states dropped)
            for (seq, ls) in seqs.iter_mut().zip(all_layer_states.drain(..n)) {
                seq.layer_states = ls;
            }
        }

        // ── Phase 3: Post-graph (update sequence state) ──
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[i]);
            seq.seq_len += 1;
        }

        // Close the stream hazard described at the top of this function.
        // The caller will start issuing ops on `caller_stream` as soon
        // as we return; without this handshake, those ops race against
        // our pending default_stream work on the shared GPU scratch
        // buffers.
        //
        // GPU-side cross-stream event handshake: record our completion
        // marker on `stream` (= default_stream) after all decode work is
        // queued there, then have the caller's stream wait for it. The
        // CPU never blocks, the GPU enforces the ordering, and decode
        // and prefill can overlap on the GPU as much as their resource
        // use allows.
        //
        // Uses a dedicated `decode_batch_done_event` (not
        // `secondary_event`, which is owned by `async_chkpt`'s flow —
        // that flow can interleave with this one inside the same
        // forward tick, so sharing the handle would race two unrelated
        // record/wait pairs).
        //
        // When the caller already passed `default_stream` (the
        // decode-only fast path), the work is on the same stream they
        // continue on and no extra sync is needed.
        if caller_stream != stream {
            self.gpu
                .record_event(self.decode_batch_done_event, stream)?;
            self.gpu
                .stream_wait_event(caller_stream, self.decode_batch_done_event)?;
        }

        Ok(self.decode_logits_ptr())
    }
}
