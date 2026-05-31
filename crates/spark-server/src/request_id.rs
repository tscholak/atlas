// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request identifier extraction.
//!
//! Every inbound API request gets a `RequestId` — either echoed from
//! the client's `x-request-id` header, or generated server-side as a
//! UUIDv7. The id is then:
//!
//! - attached to a `tracing::info_span!` covering the handler so all
//!   nested `tracing::info!()` events carry it as a structured field
//!   (queryable via `journalctl REQUEST_ID=<uuid>` or LogQL
//!   `{...} | json | request_id="<uuid>"`);
//! - threaded into the `InferenceRequest` and then `ActiveSeq` so
//!   the scheduler tick log can attribute per-tick metrics back to
//!   the request that occupied the tick;
//! - included in `atlas::dump` request/response events for offline
//!   replay correlation;
//! - echoed back on the HTTP response as `x-request-id` so the
//!   client (typically the heim agent) can confirm the round-trip
//!   and pin the same id on session-DB entries.
//!
//! UUIDv7 is chosen because it's monotonic-by-time, sorts in
//! chronological order in log aggregators, and carries a millisecond
//! timestamp prefix that's still useful even when stripped from
//! tracing context.

use axum::http::{HeaderMap, HeaderValue};

/// Standard HTTP header name. Lowercase per RFC 7540 §8.1.2.
pub const HEADER_NAME: &str = "x-request-id";

/// A per-request identifier suitable for log filtering and
/// cross-store joins. Cheap to clone (Arc-backed String would be
/// nicer, but the overhead of a small String on the request path is
/// negligible and avoids leaking Arc through public APIs).
#[derive(Clone, Debug)]
pub struct RequestId(String);

impl RequestId {
    /// Wrap an already-resolved id. Use [`RequestId::from_headers`]
    /// at API entry points; this constructor is for tests and for
    /// rare callers that compute the id elsewhere.
    pub fn new(id: String) -> Self {
        Self(id)
    }

    /// Generate a fresh UUIDv7. Each call returns a unique id with a
    /// millisecond-timestamp prefix.
    pub fn generate() -> Self {
        Self(uuid::Uuid::now_v7().to_string())
    }

    /// Read the `x-request-id` header if present and well-formed; fall
    /// back to a generated UUIDv7. "Well-formed" = ASCII, non-empty,
    /// length cap 256 (anti-abuse — a client shouldn't be sending
    /// kilobyte ids).
    pub fn from_headers(headers: &HeaderMap) -> Self {
        if let Some(v) = headers.get(HEADER_NAME) {
            if let Ok(s) = v.to_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() && trimmed.len() <= 256 && trimmed.is_ascii() {
                    return Self(trimmed.to_string());
                }
            }
        }
        Self::generate()
    }

    /// Borrow the inner string for tracing field emission, dump
    /// records, and HTTP response header construction.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Owned clone of the inner string. Use when the id needs to live
    /// past the handler's borrow lifetime (e.g. moved into the
    /// `InferenceRequest` sent across an mpsc).
    pub fn into_string(self) -> String {
        self.0
    }

    /// Build the `x-request-id: <id>` response header. Returns `None`
    /// only when the id contains bytes invalid for an HTTP header
    /// value, which `from_headers` and `generate` both prevent.
    pub fn to_header_value(&self) -> Option<HeaderValue> {
        HeaderValue::from_str(&self.0).ok()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_are_unique() {
        let a = RequestId::generate();
        let b = RequestId::generate();
        assert_ne!(a.as_str(), b.as_str());
        // UUIDv7 string form: 36 chars (8-4-4-4-12 hex with dashes).
        assert_eq!(a.as_str().len(), 36);
    }

    #[test]
    fn header_takes_priority_when_valid() {
        let mut h = HeaderMap::new();
        h.insert(HEADER_NAME, HeaderValue::from_static("client-supplied-123"));
        let id = RequestId::from_headers(&h);
        assert_eq!(id.as_str(), "client-supplied-123");
    }

    #[test]
    fn empty_header_falls_back_to_generated() {
        let mut h = HeaderMap::new();
        h.insert(HEADER_NAME, HeaderValue::from_static(""));
        let id = RequestId::from_headers(&h);
        assert_eq!(id.as_str().len(), 36, "should be a UUIDv7");
    }

    #[test]
    fn overlong_header_falls_back_to_generated() {
        let mut h = HeaderMap::new();
        let long = "a".repeat(257);
        h.insert(HEADER_NAME, HeaderValue::from_str(&long).unwrap());
        let id = RequestId::from_headers(&h);
        assert_eq!(id.as_str().len(), 36);
    }

    #[test]
    fn whitespace_around_header_is_trimmed() {
        let mut h = HeaderMap::new();
        h.insert(HEADER_NAME, HeaderValue::from_static("  inner-id  "));
        let id = RequestId::from_headers(&h);
        assert_eq!(id.as_str(), "inner-id");
    }

    #[test]
    fn round_trip_through_header_value() {
        let id = RequestId::generate();
        let hv = id.to_header_value().unwrap();
        assert_eq!(hv.to_str().unwrap(), id.as_str());
    }

    #[test]
    fn missing_header_generates_uuid_v7() {
        let h = HeaderMap::new();
        let id = RequestId::from_headers(&h);
        assert_eq!(id.as_str().len(), 36);
    }
}
