// SPDX-License-Identifier: AGPL-3.0-only

//! Per-batch slot-pointer staging for the batched SSM kernels (Phase IIb+).
//!
//! Background: the `_batched` SSM kernels added in Phase IIa
//! (`gated_delta_rule_decode_batched`, `gated_delta_rule_wy3_batched`,
//! `causal_conv1d_update_l2norm_batched`) take per-batch pointer arrays
//! instead of a single contiguous state buffer. Each batch position's
//! state can live at any slot in `SsmStatePool`, so the caller stages
//! the `batch_size` slot-base pointers into a small GPU buffer before
//! each kernel launch.
//!
//! The staging buffer is `TransformerModel::slot_ptrs_buf` — a single
//! allocation sliced by `(kind, ssm_layer_idx)` so multiple in-flight
//! launches at different layers don't clobber each other within a single
//! forward pass. CUDA graphs capture the destination DevicePtrs (one per
//! `(kind, layer)`); replay reads whatever values are currently in those
//! regions, which the caller updates per launch.
//!
//! The kind enum gives the staging method a uniform interface across the
//! four pool buffers a single batched-SSM call may need to address.

#![allow(unused_imports, dead_code)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;

/// Which `SsmStatePool` buffer the staged pointers should reference.
///
/// The variants map to dedicated regions within `slot_ptrs_buf` so a
/// single batched kernel launch can take multiple pointer arrays (e.g.
/// `wy3` needs main state + 2 intermediates = 3 staged arrays per layer)
/// without the arrays overlapping or fighting for the same staging slot.
#[derive(Clone, Copy, Debug)]
pub(in crate::model) enum SsmPtrKind {
    /// Main h_state pool: `SsmStatePool::h_state(layer, slot)`.
    HState,
    /// Main conv_state pool: `SsmStatePool::conv_state(layer, slot)`.
    ConvState,
    /// Intermediate h_state pool, indexed by token_idx ∈ 0..num_intermediates.
    /// Used by WY-chunkwise verify kernels (K=2/3/4/17) to write per-token
    /// intermediates that the scheduler later picks ONE of to commit, based
    /// on the per-seq accept count. `SsmStatePool::h_intermediate`.
    HInter(usize),
    /// Intermediate conv_state pool, same scheme. `SsmStatePool::conv_intermediate`.
    ConvInter(usize),
}

/// Max intermediate index supported by the fixed slot-ptrs region layout.
/// SLOT_PTRS_NUM_KINDS = 16 is the buffer's per-layer reservation
/// (`impl_a1.rs::SLOT_PTRS_NUM_KINDS`). With 2 reserved for HState+
/// ConvState that leaves 14 slots split evenly: 7 each for HInter and
/// ConvInter. Qwen3.6's `num_intermediates` peaks at 5 (DFlash K=γ with
/// γ=4, +1 for the final) so 7 is comfortable headroom.
const MAX_INTERMEDIATE_KIND: usize = 7;

impl SsmPtrKind {
    /// Fixed index in `[0..SLOT_PTRS_NUM_KINDS)` that selects this kind's
    /// reservation band within the staging buffer's per-layer block.
    fn kind_idx(self) -> usize {
        match self {
            SsmPtrKind::HState => 0,
            SsmPtrKind::ConvState => 1,
            SsmPtrKind::HInter(i) => {
                debug_assert!(i < MAX_INTERMEDIATE_KIND);
                2 + i
            }
            SsmPtrKind::ConvInter(i) => {
                debug_assert!(i < MAX_INTERMEDIATE_KIND);
                2 + MAX_INTERMEDIATE_KIND + i
            }
        }
    }
}

impl TransformerModel {
    /// Stage per-batch slot-base pointers for one (kind, layer) and return
    /// the DevicePtr to the start of the array. The caller passes this
    /// DevicePtr to the matching `_batched` SSM kernel's pointer-array
    /// argument.
    ///
    /// `slots` has `batch_size` entries, one slot index per batch position.
    /// Each entry must be a valid claimed slot in `SsmStatePool` — the
    /// kernel dereferences `slot_ptrs_buf[ptr_offset .. ptr_offset + B*8]`
    /// as `float* const*` and offsets within each slot internally.
    ///
    /// Subtle: the underlying H2D copy runs on the caller's stream. CUDA
    /// graphs capture the destination address (returned `DevicePtr`) but
    /// **not** the contents; replay reads whatever's at that address at
    /// replay time. The caller is responsible for restaging immediately
    /// before each launch when batch composition changes — see
    /// `verify_b/batched_dispatch.rs` (Phase IIb).
    pub(in crate::model) fn stage_slot_ptrs_dispatch(
        &self,
        kind: SsmPtrKind,
        ssm_layer_idx: usize,
        slots: &[usize],
        stream: u64,
    ) -> Result<DevicePtr> {
        let batch_size = slots.len();
        assert!(
            batch_size <= self.ssm_pool.max_slots,
            "stage_slot_ptrs: batch_size {batch_size} exceeds pool max_slots {}",
            self.ssm_pool.max_slots,
        );

        // Compute destination offset in slot_ptrs_buf. The buffer is
        // partitioned as `[NUM_KINDS][num_ssm_layers][max_slots]` of u64
        // pointers, so kind k's layer L starts at:
        //   ((k * num_ssm_layers + L) * max_slots) * 8 bytes.
        let num_ssm_layers = self.ssm_pool.num_ssm_layers.max(1);
        let max_slots = self.ssm_pool.max_slots.max(1);
        let slot_index =
            (kind.kind_idx() * num_ssm_layers + ssm_layer_idx) * max_slots;
        let byte_offset = slot_index * 8;
        let dst = self.slot_ptrs_buf.offset(byte_offset);

        // Resolve per-slot pool pointers and write them into the
        // matching slice of `slot_ptrs_host_pinned`. The source slice
        // we hand to `copy_h2d_async` is THIS pinned region — a stable
        // host address that's still live at CUDA graph replay time
        // (the previous stack-based version captured a dead pointer
        // when graphs were enabled, hanging replay; see decode_a2.rs).
        //
        // The host-pinned mirror has the same `[NUM_KINDS][num_ssm_layers]
        // [max_slots]` partition as `slot_ptrs_buf`, so the byte offset
        // here is the same one we already computed for `dst`.
        //
        // SAFETY: `slot_ptrs_host_pinned` is allocated for the model's
        // lifetime with size `slot_ptrs_host_pinned_bytes`, which the
        // assert above guards against overruns of (`batch_size <=
        // max_slots` and the partition includes (kind, layer)).
        let host_base = unsafe {
            self.slot_ptrs_host_pinned.add(byte_offset) as *mut u64
        };
        for (i, &slot) in slots.iter().enumerate() {
            let ptr = match kind {
                SsmPtrKind::HState => self.ssm_pool.h_state(ssm_layer_idx, slot),
                SsmPtrKind::ConvState => self.ssm_pool.conv_state(ssm_layer_idx, slot),
                SsmPtrKind::HInter(t) => {
                    self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t)
                }
                SsmPtrKind::ConvInter(t) => {
                    self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t)
                }
            };
            // SAFETY: i < batch_size <= max_slots, host_base is the
            // base of this (kind, layer) band which reserves
            // max_slots * 8 bytes.
            unsafe { host_base.add(i).write(ptr.0); }
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(host_base as *const u8, batch_size * 8)
        };
        self.gpu.copy_h2d_async(bytes, dst, stream)?;
        Ok(dst)
    }
}
