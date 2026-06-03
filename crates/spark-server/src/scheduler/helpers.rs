// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers: BF16 conversion, hard-stop registry, loop detection, sampling defaults.

/// Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// Diagnostic: D2H-copy a logits buffer, extract top-5 tokens, log under
/// `atlas::lockprof`. Used to verify whether two streams' first-token
/// samples are reading the same memory (alias) vs reading distinct logits.
///
/// `label` is a short identifier embedded in the log line. Silent at
/// default RUST_LOG; enable with `RUST_LOG="info,atlas::lockprof=info"`.
///
/// Cost: one D2H copy of `vocab_size * 2` bytes (≈300 KB for Qwen3.6) plus
/// a single-pass partial sort. Adds ~1 ms per call on the Spark. Off the
/// hot path (only called from prefill first-token sample sites and the
/// first few decode ticks).
pub fn log_logits_top5(
    model: &dyn spark_model::traits::Model,
    logits: spark_runtime::gpu::DevicePtr,
    label: &str,
) {
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; vocab * 2];
    if model.copy_logits_to_host(logits, &mut buf).is_err() {
        tracing::info!(
            target: "atlas::lockprof",
            "logits_top5 {label}: copy_logits_to_host failed (ptr=0x{:x})",
            logits.0,
        );
        return;
    }
    let mut vals: Vec<(usize, f32)> = (0..vocab)
        .map(|i| (i, bf16_to_f32(buf[i * 2], buf[i * 2 + 1])))
        .collect();
    vals.select_nth_unstable_by(5.min(vocab.saturating_sub(1)), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top5: Vec<(usize, f32)> = vals.into_iter().take(5).collect();
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let nan_count = (0..vocab)
        .filter(|&i| bf16_to_f32(buf[i * 2], buf[i * 2 + 1]).is_nan())
        .count();
    tracing::info!(
        target: "atlas::lockprof",
        "logits_top5 {label} ptr=0x{:x} nan_count={nan_count} top5={top5:?}",
        logits.0,
    );
}

// ── Sampling defaults (SSOT) ────────────────────────────────────────────────
// All SamplingParams constructors reference these constants. Change here, not
// at each call site.
pub const DEFAULT_LZ_PENALTY: f32 = 0.0;
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
pub const DEFAULT_DRY_BASE: f32 = 1.75;
// Was 2 (oobabooga's reference value, optimised for free-form text).
// Bumped to 3 (2026-04-25) because at allowed_length=2 the DRY sampler
// penalises legitimate code micro-repetition (consecutive `(`, `,`,
// indentation, two-line `let x =` patterns) and breaks tool-call JSON
// emission. allowed_length=3 still catches the bash-fence
// "Running: …Executing: …" attractor (which spans 6+ tokens) while
// letting normal source-code patterns through. Per Agent 8 SOTA
// research, this matches the consensus for code workloads.
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 3;

/// F2 (2026-04-26): cap on free-text tokens between successive
/// `<tool_call>` opens when `tool_choice="auto"`. The grammar FSM
/// in `auto` mode (grammar.rs:461-462) sets `at_least_one=false`
/// and `stop_after_first=false`, so `is_terminated()` stays false
/// forever after the first tool call — the model can emit
/// prose↔tool↔prose↔tool indefinitely. 384 tokens is enough for
/// three normal "I'll now do X" paragraphs of agentic narrative;
/// anything beyond is the failure mode (re-narrating the plan
/// rather than executing it). Counted across non-thinking,
/// non-tool-body tokens only.
pub const MAX_INTER_TOOL_PROSE: u32 = 384;

/// F26 (2026-04-26): kernel-level entropy-collapse guard.
///
/// Disabled (`STREAK_K = 0`). Field experience: F26's pure-entropy
/// threshold can't distinguish wedged sampling from legitimate
/// high-confidence output (confident prose, code, JSON arrays). The
/// content-loop watchdog (`detect_content_token_loop`) gated on
/// per-model `enable_loop_watchdog` is the correct detector for actual
/// attractor states.
///
/// Constants kept so the call site at `decode_logits_seq.rs:285` still
/// type-checks; with `STREAK_K = 0` the gate is a no-op.
pub const ENTROPY_COLLAPSE_THRESHOLD_NATS: f32 = 0.5;
pub const ENTROPY_COLLAPSE_STREAK_K: u32 = 0;
pub const ENTROPY_COLLAPSE_WARMUP_TOKENS: usize = 32;

/// F27 (2026-04-26): logit-space attractor fingerprint.
///
/// Hash the f32_logits at 64 strided positions (~vocab/64 spacing).
/// Two tokens with near-identical logit distributions produce the
/// same fingerprint. If the same fingerprint repeats across recent
/// samples WHILE tokens varied (different sampled token each step),
/// the model is in an attractor where its decision space is stable
/// but it samples differently each time — exactly the
/// "different-tokens, same-internal-state" pattern hidden-state
/// cosine catches but at logit level (~1 µs/token, no kernel work).
///
/// `F27_RING_CAP`: ring buffer of recent fingerprints.
/// `F27_STREAK_K`: consecutive matches before tripping.
/// Same warmup + guard semantics as F26.
pub const F27_RING_CAP: usize = 16;
pub const F27_STREAK_K: u32 = 6;
pub const F27_FINGERPRINT_SAMPLES: usize = 64;
pub const F27_FINGERPRINT_QUANT: f32 = 2.0; // ~0.5 nat resolution

/// Compute a strided 64-bit fingerprint of the logit distribution.
/// Quantises each sampled value to ~0.5 nat resolution before
/// hashing so tiny FP-noise differences don't change the hash.
pub fn fingerprint_logits_strided(logits: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    let stride = (logits.len() / F27_FINGERPRINT_SAMPLES).max(1);
    let mut taken = 0;
    let mut i = 0;
    while i < logits.len() && taken < F27_FINGERPRINT_SAMPLES {
        let v = logits[i];
        let q: i32 = if v.is_finite() {
            (v * F27_FINGERPRINT_QUANT).round() as i32
        } else {
            i32::MIN
        };
        h.write_i32(q);
        i += stride;
        taken += 1;
    }
    h.finish()
}

