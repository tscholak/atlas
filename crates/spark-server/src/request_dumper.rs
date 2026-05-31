// SPDX-License-Identifier: AGPL-3.0-only

//! Request / response dumper.
//!
//! When dumping is enabled via `--dump`, the server emits one
//! `tracing::info!(target: "atlas::dump", kind="request", ...)` event
//! per incoming request and one `kind="response"` event per outgoing
//! response. The subscriber set up in
//! `main_modules::serve_phases::runtime::init_tracing` decides where
//! those events land — by default the JSON-formatted stderr layer
//! catches them and journald ingests the line; with `--dump <path>`
//! an additional file layer mirrors the same events to a JSONL file
//! for offline replay / fixture capture.
//!
//! Entries are correlated with a monotonic `seq` counter: a request
//! and its response share the same `seq`. Group pairs with a jq
//! pipeline, e.g.:
//! ```
//! journalctl -u atlas-qwen -o json | jq -c \
//!   'select(.TARGET == "atlas::dump") | {seq: .SEQ, kind: .KIND}'
//! ```
//!
//! Emission is gated by the `atlas::dump` target's enabled level in
//! `init_tracing` (off by default; INFO when `--dump` is set), so
//! body serialisation is skipped via `tracing::event_enabled!` when
//! no one is listening.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-wide monotonic counter for request/response correlation.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Reserve a sequence number. Caller later references the same `seq`
/// when emitting the matching response entry.
pub fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Emit a `kind="request"` dump event. `body` is any serde-serialisable
/// value — typically the `ChatCompletionRequest` / `ResponsesRequest` /
/// Anthropic `MessagesRequest` struct already deserialised by the
/// handler.
///
/// No-op (no serialisation, no event) when the `atlas::dump` target is
/// disabled by the active tracing filter.
pub fn dump_request<T: serde::Serialize>(endpoint: &str, seq: u64, body: &T) {
    write_event("request", endpoint, seq, body, None);
}

/// Emit a `kind="response"` dump event. `is_stream` is true when `body`
/// is the aggregated SSE chunk list rather than a final non-streaming
/// response object. No-op when dumping is disabled (same filter as
/// `dump_request`).
pub fn dump_response<T: serde::Serialize>(
    endpoint: &str,
    seq: u64,
    body: &T,
    is_stream: bool,
) {
    write_event("response", endpoint, seq, body, Some(is_stream));
}

fn write_event<T: serde::Serialize>(
    kind: &'static str,
    endpoint: &str,
    seq: u64,
    body: &T,
    is_stream: Option<bool>,
) {
    // Short-circuit when no subscriber wants the event — saves the
    // serde_json::to_string cost on bodies that can run to tens of KB.
    if !tracing::event_enabled!(target: "atlas::dump", tracing::Level::INFO) {
        return;
    }
    let body_json = match serde_json::to_string(body) {
        Ok(s) => s,
        Err(e) => format!("<serialization error: {e}>"),
    };
    let ts = iso8601_now();
    match is_stream {
        Some(s) => tracing::info!(
            target: "atlas::dump",
            kind = kind,
            endpoint = endpoint,
            seq = seq,
            stream = s,
            ts = %ts,
            body = %body_json,
        ),
        None => tracing::info!(
            target: "atlas::dump",
            kind = kind,
            endpoint = endpoint,
            seq = seq,
            ts = %ts,
            body = %body_json,
        ),
    }
}

/// ISO-8601 UTC timestamp with milliseconds. The JSON tracing
/// subscriber adds its own `timestamp` to each event, but we keep an
/// explicit `ts` field so the dump record's logical timestamp survives
/// any subscriber-format changes.
fn iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = secs.div_euclid(86400);
    let time_secs = secs.rem_euclid(86400) as u32;

    // Civil-from-days algorithm (Howard Hinnant). Unix epoch is day 0
    // = 1970-01-01.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };

    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;

    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z")
}

/// Resolve the `--dump` argument to a final file path for the optional
/// file-mirror layer. `"<auto>"` (from clap's `default_missing_value`)
/// maps to a timestamped file under `$TMPDIR`. The sentinels `"-"` and
/// `"/dev/stderr"` are NOT handled here — `init_tracing` checks for
/// them before calling and skips file-layer installation in that case.
pub fn resolve_path(arg: &str) -> std::path::PathBuf {
    if arg == "<auto>" {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        std::env::temp_dir().join(format!("atlas-dump-{ts}.jsonl"))
    } else {
        std::path::PathBuf::from(arg)
    }
}

/// True for `--dump` values that mean "journald only, no file
/// mirror" — `"-"` (POSIX convention) and `"/dev/stderr"` (legacy
/// pre-Phase-A heim module config, retained because it semantically
/// matches "emit to whatever stderr is").
pub fn is_journald_only_sentinel(arg: &str) -> bool {
    matches!(arg, "-" | "/dev/stderr")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_auto_goes_to_tmp() {
        let p = resolve_path("<auto>");
        assert!(p.starts_with(std::env::temp_dir()));
        assert!(
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("atlas-dump-")
        );
    }

    #[test]
    fn resolve_explicit_is_verbatim() {
        let p = resolve_path("/tmp/my-dump.jsonl");
        assert_eq!(p, std::path::PathBuf::from("/tmp/my-dump.jsonl"));
    }

    #[test]
    fn journald_only_sentinels() {
        assert!(is_journald_only_sentinel("-"));
        assert!(is_journald_only_sentinel("/dev/stderr"));
        assert!(!is_journald_only_sentinel("/tmp/foo.jsonl"));
        assert!(!is_journald_only_sentinel("<auto>"));
    }

    #[test]
    fn iso8601_has_expected_shape() {
        let s = iso8601_now();
        // YYYY-MM-DDTHH:MM:SS.sssZ  = 24 chars
        assert_eq!(s.len(), 24, "{s}");
        assert!(s.ends_with('Z'));
        assert_eq!(s.as_bytes()[10], b'T');
    }

    #[test]
    fn next_seq_is_monotonic_per_process() {
        let a = next_seq();
        let b = next_seq();
        let c = next_seq();
        assert!(b > a);
        assert!(c > b);
    }
}
