// SPDX-License-Identifier: AGPL-3.0-only

//! Stop-string predicate (A5). OpenAI's Chat Completions spec mandates
//! that any configured `stop` strings are trimmed from the returned
//! content. The streaming path previously did this via substring match
//! over an accumulated content buffer at the FSM level; the blocking
//! path did it via `strip_suffix` posthoc. After Stage 3 both paths
//! share this predicate.
//!
//! Semantics: `feed(delta)` appends `delta` to an internal accumulated
//! content buffer, then scans for the first occurrence (`str::find`,
//! not `strip_suffix`) of any configured stop string. Returns:
//!   * `Pass` — no stop matched; the full `delta` is safe to emit.
//!   * `Fire { safe_prefix, matched }` — a stop fired; `safe_prefix`
//!     is the slice of `delta` that comes BEFORE the stop boundary
//!     (may be empty when the stop straddled prior deltas). `matched`
//!     is the verbatim stop string that fired.
//!
//! Stops are sorted longest-first at construction so overlapping
//! prefixes (e.g. `["</answer", "</answer>"]`) match the longer
//! variant. Matches the historical sort in
//! `chat_stream_dispatch.rs:67-68` and `inference_impl.rs:310-311`.

#[derive(Debug)]
pub(super) struct StopPredicate {
    accumulated: String,
    stops: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum StopOutcome {
    Pass,
    Fire { safe_prefix: String, matched: String },
}

impl StopPredicate {
    pub(super) fn new(mut stops: Vec<String>) -> Self {
        stops.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Self {
            accumulated: String::new(),
            stops,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    pub(super) fn feed(&mut self, delta: &str) -> StopOutcome {
        let already_emitted = self.accumulated.len();
        self.accumulated.push_str(delta);
        for stop in &self.stops {
            if let Some(pos) = self.accumulated.find(stop.as_str()) {
                let safe_prefix = if pos > already_emitted {
                    self.accumulated[already_emitted..pos].to_string()
                } else {
                    String::new()
                };
                return StopOutcome::Fire {
                    safe_prefix,
                    matched: stop.clone(),
                };
            }
        }
        StopOutcome::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_when_no_stop() {
        let mut p = StopPredicate::new(vec!["</answer>".to_string()]);
        assert_eq!(p.feed("hello "), StopOutcome::Pass);
        assert_eq!(p.feed("world"), StopOutcome::Pass);
    }

    #[test]
    fn fire_mid_delta() {
        let mut p = StopPredicate::new(vec!["STOP".to_string()]);
        assert_eq!(p.feed("hello "), StopOutcome::Pass);
        assert_eq!(
            p.feed("worldSTOPbye"),
            StopOutcome::Fire {
                safe_prefix: "world".to_string(),
                matched: "STOP".to_string()
            }
        );
    }

    #[test]
    fn fire_straddle_across_deltas() {
        let mut p = StopPredicate::new(vec!["STOP".to_string()]);
        assert_eq!(p.feed("hello ST"), StopOutcome::Pass);
        // The "ST" is already in `accumulated`; the next delta
        // completes "STOP". `safe_prefix` must be empty because every
        // byte of the current delta is at or after the stop boundary.
        assert_eq!(
            p.feed("OPgoodbye"),
            StopOutcome::Fire {
                safe_prefix: String::new(),
                matched: "STOP".to_string()
            }
        );
    }

    #[test]
    fn longest_match_wins() {
        // Stops sorted longest-first; both have a common prefix.
        let mut p = StopPredicate::new(vec![
            "</answer".to_string(),
            "</answer>".to_string(),
        ]);
        // The "</answer>" stop is strictly longer; it fires first.
        match p.feed("foo</answer>") {
            StopOutcome::Fire { matched, .. } => assert_eq!(matched, "</answer>"),
            _ => panic!("expected Fire"),
        }
    }

    #[test]
    fn empty_predicate_always_passes() {
        let mut p = StopPredicate::new(vec![]);
        assert!(p.is_empty());
        assert_eq!(p.feed("hello"), StopOutcome::Pass);
    }
}
