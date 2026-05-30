// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode at N seqs × 1 token each.
    ///
    /// Tries the batched-kernel path first (Phase IIb/c — uses the Phase IIa
    /// `_batched` GDN/conv1d kernels with per-batch slot pointer arrays
    /// staged into ctx scratch); falls back to N sequential single-seq
    /// decode calls if the batched path errors or required kernels aren't
    /// loaded.
    #[allow(unused_variables)]
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // N=1: defer to single-seq decode directly (the n>=2 batched path
        // has overhead that doesn't pay off at N=1).
        if num_seqs == 1 {
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            return self.decode(
                hidden,
                residual,
                states[0],
                _kv_cache,
                _seq_lens[0],
                &mut _block_tables[0].clone(),
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            );
        }

        // N>=2 batched-kernel path. Opt-in via `ATLAS_BATCHED_SSM_DECODE=1`
        // because, with phases 2/3/6/7 still per-seq, the saved launches on
        // conv1d_l2norm + gdn_decode don't currently outweigh per-seq
        // `decode()`'s CUDA-graph replay savings (measured ~60% throughput
        // vs per-seq at N=4 decode-heavy 6.7K-prompt 500-tok). Left as a
        // toggle so the follow-up (CUDA graph capture across batched path,
        // or batched QKVZ/out_proj/MoE) can re-validate end-to-end.
        let batched_enabled = std::env::var("ATLAS_BATCHED_SSM_DECODE")
            .is_ok_and(|v| v == "1" || v == "true");
        if !batched_enabled {
            // Default: per-seq sequential decode (proven path).
            let h = ctx.config.hidden_size;
            let residual_elem = if ctx.config.use_fp32_residual() { 4 } else { 2 };
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            for i in 0..num_seqs {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let residual_i = residual.offset(i * h * residual_elem);
                self.decode(
                    hidden_i,
                    residual_i,
                    states[i],
                    _kv_cache,
                    _seq_lens[i],
                    &mut _block_tables[i].clone(),
                    &mut stub_disk,
                    &mut stub_last_offloaded,
                    ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        // Try the batched-kernel path. On error, fall back to per-seq.
        if let Err(e) = self.decode_multi_seq_batched_inner(
            hidden, residual, num_seqs, states, ctx, stream,
        ) {
            tracing::warn!(
                target: "atlas::lockprof",
                "decode_multi_seq batched path failed ({e}); falling back to per-seq sequential"
            );
            let h = ctx.config.hidden_size;
            let residual_elem = if ctx.config.use_fp32_residual() { 4 } else { 2 };
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            for i in 0..num_seqs {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let residual_i = residual.offset(i * h * residual_elem);
                self.decode(
                    hidden_i,
                    residual_i,
                    states[i],
                    _kv_cache,
                    _seq_lens[i],
                    &mut _block_tables[i].clone(),
                    &mut stub_disk,
                    &mut stub_last_offloaded,
                    ctx,
                    stream,
                )?;
            }
        }
        return Ok(());
    }

    /// N>=2 batched decode using Phase IIa batched kernels. Returns
    /// `Err` if any required kernel handle isn't loaded (caller falls
    /// back to per-seq sequential).
    ///
    /// Structure:
    /// - Phase 1: rms_norm_residual batched over N seqs
    /// - Phase 2: per-seq QKVZ projection (+ optional deinterleave)
    /// - Phase 3: per-seq BA projection + GDN gate compute
    /// - Phase 4: ONE batched conv1d_update_l2norm at N seqs
    /// - Phase 5: ONE batched gdn_decode at N seqs
    /// - Phase 6: per-seq gated_rms_norm + out_proj
    /// - Phase 7: per-seq residual + post-norm + MoE + residual_add
    ///
    /// Buffer layout (all sized for max_batch_tokens ≥ N):
    /// - normed: `[N, hidden_size]` BF16
    /// - deinterleaved: `[N, qkvz_size]` BF16 (Q|K|V|Z packed per seq)
    /// - ssm_qkvz: `[N, qkvz_size]` BF16 (intermediate when not sequential)
    /// - ssm_ba: `[N, ba_size]` BF16
    /// - ssm_gates: `[N, 2*nv]` FP32 (gate(nv) | beta(nv) per seq)
    /// - attn_output (conv_out): `[N, conv_dim]` BF16 (conv1d output, reused for gated_rms_norm output post-gdn)
    /// - qkv_output (gdn_out): `[N, value_dim]` BF16
    /// - moe_output: `[N, hidden_size]` BF16 (SSM out_proj results, copied to ssm_out_safe before MoE)
    ///
    /// h_state_ptrs and conv_state_ptrs are staged in scratch at fixed
    /// offsets (after the 32K BatchedAttnMetadata region used by
    /// attention layers in the same forward).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_multi_seq_batched_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.conv1d_l2norm_batched_k.0 == 0 || self.gdn_batched_k.0 == 0 {
            anyhow::bail!(
                "decode_multi_seq_batched_inner: batched SSM kernels not loaded"
            );
        }
        // One-shot diagnostic so we can confirm in the log whether the
        // batched path is actually firing at N>=2 (helps distinguish
        // "ran batched and got X t/s" from "fell back to per-seq").
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                target: "atlas::lockprof",
                "decode_multi_seq_batched_inner: ENTERED at N={num_seqs}"
            );
        });

        let n = num_seqs;
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let residual_elem = if ctx.config.use_fp32_residual() { 4 } else { 2 };

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qk_ch = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();

        // ── Stage h_state_ptrs / conv_state_ptrs ──
        // Source: this layer's `slot_ptrs_host` (host-pinned, stable
        // address for the layer's lifetime). Destination: this layer's
        // `slot_ptrs_dev` (device-side mirror, also stable). Both
        // layouts are `[2 × MAX_BATCHED_DECODE_SLOTS]` u64 — first half
        // is h_state pointers, second half is conv_state pointers.
        //
        // The stability matters under CUDA graph capture: the graph's
        // memcpy nodes reference these addresses and replay at later
        // ticks. With the prior stack `Vec<u64>` source the pointer was
        // dead by replay, causing hangs (see decode_a2.rs notes on the
        // F-redux experiment).
        anyhow::ensure!(
            n <= super::MAX_BATCHED_DECODE_SLOTS,
            "decode_multi_seq_batched: n={n} exceeds MAX_BATCHED_DECODE_SLOTS={}",
            super::MAX_BATCHED_DECODE_SLOTS,
        );
        // SAFETY: slot_ptrs_host is allocated for layer lifetime with
        // size MAX_BATCHED_DECODE_SLOTS * 8 * 2 bytes (init.rs).
        let h_ptrs_host = self.slot_ptrs_host as *mut u64;
        let conv_ptrs_host = unsafe { h_ptrs_host.add(super::MAX_BATCHED_DECODE_SLOTS) };
        for (i, state) in states.iter_mut().take(n).enumerate() {
            let ssm = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode_multi_seq_batched: expected SsmLayerState"
                    )
                })?;
            // SAFETY: i < n <= MAX_BATCHED_DECODE_SLOTS.
            unsafe {
                h_ptrs_host.add(i).write(ssm.h_state.0);
                conv_ptrs_host.add(i).write(ssm.conv_state.0);
            }
        }
        let h_ptrs_dev = self.slot_ptrs_dev;
        let conv_ptrs_dev =
            self.slot_ptrs_dev.offset(super::MAX_BATCHED_DECODE_SLOTS * 8);
        ctx.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(h_ptrs_host as *const u8, n * 8)
            },
            h_ptrs_dev,
            stream,
        )?;
        ctx.gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(conv_ptrs_host as *const u8, n * 8)
            },
            conv_ptrs_dev,
            stream,
        )?;

        // ── Phase 1: RMS norm + residual (BATCHED) ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // ── Phase 2: per-seq QKVZ projection (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let qkvz_out = ctx.buffers.ssm_qkvz();
        for i in 0..n {
            let normed_i = normed.offset(i * h * bf16);
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            if self.sequential_qkvz {
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        &self.ssm.in_proj_qkvz,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            } else {
                let qkvz_i = qkvz_out.offset(i * qkvz_size * bf16);
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        qkvz_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        &self.ssm.in_proj_qkvz,
                        qkvz_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_i,
                    deint_i,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        // ── Phase 3: per-seq fused BA projection + GDN gates ──
        // Layout per seq: gates_buf[i * (2*nv) .. + 2*nv] = [gate(nv) | beta(nv)] FP32
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_beta_stride_bytes = nv * 2 * fp32;
        for i in 0..n {
            let normed_i = normed.offset(i * h * bf16);
            let ba_out = ctx.buffers.ssm_ba().offset(i * ba_size * bf16);
            let gate_i = gates_buf.offset(i * gate_beta_stride_bytes);
            let beta_i = gates_buf.offset(i * gate_beta_stride_bytes + nv * fp32);
            // ba_gates_k fuses BA projection + gate/beta inline.
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed_i,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gate_i,
                beta_i,
                ba_size as u32,
                h as u32,
                vpg as u32,
                stream,
            )?;
            // ba_out is the intermediate BA projection output; the fused
            // kernel uses it as scratch internally but we keep the
            // per-seq offset for the buffer for diagnostic purposes.
            let _ = ba_out;
        }

        // ── Phase 4: BATCHED conv1d_update_l2norm ──
        // Input: deinterleaved at offset 0 (seq 0 token), with
        // input_stride = qkvz_size (skips Z gate between seqs since
        // conv reads only the first conv_dim of each row).
        // Output: conv_out_buf at offset 0, with output_stride = conv_dim.
        let conv_out_buf = ctx.buffers.attn_output();
        ops::conv1d_update_l2norm_batched(
            ctx.gpu,
            self.conv1d_l2norm_batched_k,
            conv_ptrs_dev,
            deinterleaved,
            &self.ssm.conv1d,
            conv_out_buf,
            conv_dim as u32,
            d_conv as u32,
            n as u32,
            qk_ch,
            kd as u32,
            qkvz_size as u32,
            conv_dim as u32,
            1e-6,
            stream,
        )?;

        // ── Phase 5: BATCHED gdn_decode ──
        // q/k/v read from conv_out_buf with stride conv_dim (= per-batch row width).
        // gate/beta read from gates_buf with stride 2*nv (FP32 elements).
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        let gdn_out_buf = ctx.buffers.qkv_output();
        // Strides between consecutive batch rows in each buffer:
        // - conv_out_buf is [N, conv_dim] BF16 (Q/K/V packed per row).
        // - gates_buf is [N, 2*nv] FP32 (gate then beta interleaved per row).
        // - gdn_out_buf is [N, value_dim] BF16 (per-seq SSM output).
        ops::gdn_decode_batched(
            ctx.gpu,
            self.gdn_batched_k,
            h_ptrs_dev,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            (2 * nv) as u32,
            value_dim as u32,
            stream,
        )?;

        // ── Phase 6: per-seq gated_rms_norm + out_proj ──
        let moe_output = ctx.buffers.moe_output();
        for i in 0..n {
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            let gdn_out_i = gdn_out_buf.offset(i * value_dim * bf16);
            // Reuse conv_out_buf region as normed_ssm output buffer
            // — per-seq slice of value_dim (no longer need conv data).
            let normed_ssm_i = conv_out_buf.offset(i * value_dim * bf16);
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_k,
                gdn_out_i,
                z_i,
                &self.ssm.norm,
                normed_ssm_i,
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;
            let ssm_out_i = moe_output.offset(i * h * bf16);
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_ssm_i,
                    dense_out,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed_ssm_i,
                    &self.ssm.out_proj,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }

        // ── Phase 7: per-seq residual + post-norm + MoE + residual_add ──
        // Bug #6 mitigation: MoE writes its output to `moe_output[0..h]`
        // (single-token MoE doesn't know about batch). Copy each seq's
        // SSM out to `ssm_out_safe` BEFORE running MoE so the per-seq
        // residual_add_rms_norm reads the right SSM result.
        let ssm_out_safe = ctx.buffers.ssm_deinterleaved(); // reuse — no longer needed
        for i in 0..n {
            let src = moe_output.offset(i * h * bf16);
            let dst = ssm_out_safe.offset(i * h * bf16);
            ctx.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        }
        for i in 0..n {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let ssm_out_i = ssm_out_safe.offset(i * h * bf16);
            let residual_i = residual.offset(i * h * residual_elem);
            let normed2 = ctx.buffers.norm_output().offset(i * h * bf16);
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden_i,
                ssm_out_i,
                &self.post_attn_norm,
                normed2,
                residual_i,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden_i,
                moe_out,
                h as u32,
                stream,
            )?;
        }

        Ok(())
    }
}
