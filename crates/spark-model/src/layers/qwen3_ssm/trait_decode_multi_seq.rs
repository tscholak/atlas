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

        // N>=2: try the batched-kernel path. On error, fall back to the
        // proven per-seq sequential path.
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
    /// back).
    #[allow(clippy::too_many_arguments, unused_variables)]
    pub(super) fn decode_multi_seq_batched_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
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
        // Disable batched-path entry until the layer body is fully
        // implemented (in flight: Phase IIb). Returns Err so the caller
        // falls back to per-seq decode — currently the only path
        // exercised by production.
        anyhow::bail!(
            "decode_multi_seq_batched_inner: layer body not yet implemented; \
             caller should fall back to per-seq decode"
        );
    }

}
