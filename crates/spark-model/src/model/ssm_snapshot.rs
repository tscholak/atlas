// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated GPU memory pool for SSM state snapshots (Marconi caching).
///
/// Each snapshot slot stores a copy of h_state + conv_state for all SSM layers
/// at a specific point in a token sequence. When a prefix cache hit occurs, the
/// snapshot is restored to avoid re-running SSM computation for cached tokens.
pub(crate) struct SsmSnapshotPool {
    pub(super) h_snapshots: Vec<DevicePtr>,
    pub(super) conv_snapshots: Vec<DevicePtr>,
    /// Single allocation of `num_slots * hidden_bytes`. Holds the
    /// post-final-RMS-norm hidden state at the end of the prompt (the
    /// exact buffer that cold prefill feeds into `lm_head`). When a
    /// warm cache hit covers the entire prompt we can skip Phase 4
    /// entirely and pass this pointer directly into `lm_head`, which
    /// makes warm == cold bit-for-bit and stops the 1-step state-advance
    /// drift documented in `prefill_b/finalize_last.rs`'s `is_warm_cache_
    /// last_token` comment. Empty (`DevicePtr::NULL`) when snapshots are
    /// disabled.
    pub(super) hidden_snapshots: DevicePtr,
    pub(super) hidden_bytes: usize,
    pub(super) free_slots: Mutex<Vec<usize>>,
    pub(super) num_slots: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    pub(super) num_ssm_layers: usize,
    /// Maps snapshot_slot_id → session_hash for session-scoped isolation.
    /// When restoring, skip snapshots that belong to a different session.
    pub(super) session_tags: Mutex<std::collections::HashMap<usize, u64>>,
}

impl SsmSnapshotPool {
    pub(super) fn new(
        num_slots: usize,
        h_bytes: usize,
        conv_bytes: usize,
        hidden_bytes: usize,
        num_ssm_layers: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        if num_slots == 0 || num_ssm_layers == 0 {
            return Ok(Self {
                h_snapshots: Vec::new(),
                conv_snapshots: Vec::new(),
                hidden_snapshots: DevicePtr::NULL,
                hidden_bytes,
                free_slots: Mutex::new(Vec::new()),
                num_slots: 0,
                h_bytes,
                conv_bytes,
                num_ssm_layers,
                session_tags: Mutex::new(std::collections::HashMap::new()),
            });
        }

        let mut h_snapshots = Vec::with_capacity(num_ssm_layers);
        let mut conv_snapshots = Vec::with_capacity(num_ssm_layers);
        for _ in 0..num_ssm_layers {
            h_snapshots.push(gpu.alloc(num_slots * h_bytes)?);
            conv_snapshots.push(gpu.alloc(num_slots * conv_bytes)?);
        }
        let hidden_snapshots = gpu.alloc(num_slots * hidden_bytes)?;

        let free_slots: Vec<usize> = (0..num_slots).rev().collect();
        let total_mb = (num_ssm_layers * num_slots * (h_bytes + conv_bytes)
            + num_slots * hidden_bytes)
            / (1024 * 1024);
        tracing::info!(
            "SSM snapshot pool (Marconi): {num_slots} slots × {num_ssm_layers} layers \
             + {hidden_bytes}B hidden = {total_mb} MB",
        );

        Ok(Self {
            h_snapshots,
            conv_snapshots,
            hidden_snapshots,
            hidden_bytes,
            free_slots: Mutex::new(free_slots),
            num_slots,
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            session_tags: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.num_slots > 0
    }

    /// Save SSM state from active pool slot into a snapshot slot.
    /// `normed_src = Some(ptr)` additionally caches the post-final-RMS-norm
    /// hidden state (the `lm_head` input) so warm-cache full-prompt hits
    /// can skip Phase 4 entirely via [`Self::hidden_snapshot_ptr`].
    /// Intermediate checkpoints pass `None` because they live mid-prompt
    /// and have no meaningful "post-LN of final token" — those snapshots
    /// are only consumed via partial-match recompute, which never reads
    /// the hidden slot.
    ///
    /// Returns `None` if no free snapshot slots are available.
    /// Tags the snapshot with `session_hash` for session-scoped isolation.
    pub(super) fn save(
        &self,
        ssm_slot: usize,
        session_hash: u64,
        main_pool: &SsmStatePool,
        normed_src: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let snap_slot = match self.free_slots.lock().pop() {
            Some(s) => s,
            None => return Ok(None),
        };
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        if let Some(src) = normed_src {
            gpu.copy_d2d_async(
                src,
                self.hidden_snapshots
                    .offset(snap_slot * self.hidden_bytes),
                self.hidden_bytes,
                stream,
            )?;
        }
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Ok(Some(snap_slot))
    }

    /// Pointer to the cached post-LN hidden state for `snap_slot`. The
    /// warm-cache `prefill_chunk_dispatch` path passes this directly into
    /// `lm_head` instead of re-running embed→layers→LN on the last token
    /// (which would advance SSM state by one step and shift logits — the
    /// 1-step drift documented in `prefill_b/finalize_last.rs`).
    pub(super) fn hidden_snapshot_ptr(&self, snap_slot: usize) -> DevicePtr {
        self.hidden_snapshots.offset(snap_slot * self.hidden_bytes)
    }

    /// Check if a snapshot belongs to the given session.
    /// Returns true if: session tracking is disabled (hash=0), no tag exists, or tags match.
    pub(super) fn session_matches(&self, snap_slot: usize, session_hash: u64) -> bool {
        if session_hash == 0 {
            return true;
        } // Legacy: no session tracking
        let tags = self.session_tags.lock();
        match tags.get(&snap_slot) {
            None => true, // Untagged snapshot (pre-session-manager) — allow
            Some(&tag) => tag == session_hash,
        }
    }

    /// Restore SSM state from a snapshot slot into an active pool slot.
    pub(super) fn restore(
        &self,
        snap_slot: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Return a snapshot slot to the free list.
    pub(super) fn free(&self, snap_slot: usize) {
        self.free_slots.lock().push(snap_slot);
    }

    /// Try to reclaim a snapshot slot by evicting the LRU snapshot from the
    /// prefix cache's snapshot index. Snapshots are decoupled from tree nodes,
    /// so this directly frees a snapshot without needing to evict KV blocks.
    pub(super) fn reclaim_from_cache(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        _kv_cache: &mut PagedKvCache,
    ) -> bool {
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}
