// SPDX-License-Identifier: AGPL-3.0-only
//
// Mutable per-stream state captured by the `flat_map` closure in
// `chat_stream.rs`. Lifted out of that closure so each `StreamEvent`
// arm can be extracted to a free function (`handle_token`,
// `handle_done`, `handle_error`) that takes `&mut StreamState` plus
// any additional non-state arguments.
//
// Read-only context (`Arc<AppState>`, model name, tool defs, ...) is
// passed via `StreamCtx` (see `ctx.rs`) so the helpers don't need to
// duplicate two dozen function-parameter slots.

use std::collections::HashMap;

use crate::tool_parser;

/// Major phase of the chat-stream emit FSM. The decoder feeds one
/// stream of tokens into this state machine; the phase determines
/// where the decoded bytes go.
///
///   Thinking ── substring match of `</end_tag>` ──► Content
///   Content  ── substring match of `<start_tag>` (re-open) ──► Thinking
///   Thinking | Content ── stop string / watchdog / tool-cap trip ──► Stopped
///
/// `Stopped` is terminal: no further content is emitted to the SSE
/// stream, though the `handle_done` arm still runs to write the
/// finish_reason and trailing usage block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum StreamPhase {
    Thinking,
    Content,
    Stopped,
}

/// Per-tool-call state held in `StreamState::streaming_tool_args`
/// across the streaming `ToolCallStart` → `ToolCallArgDelta`* →
/// `ToolCallEnd` event sequence. The id comes from the detector
/// (via `next_tool_call_id`) at `ToolCallStart` and is preserved
/// verbatim so the observability dump records the same id the
/// client saw over SSE.
pub(super) struct StreamingToolCall {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) args: String,
}

pub(super) struct StreamState {
    /// Lazy streaming-decoder over the model output. Created on the
    /// first token with `skip_special_tokens=false` so that protocol
    /// markers (`</think>`, ChatML specials if they appear) reach the
    /// emit-layer state machine as literal text. Spans both the
    /// Thinking and Content phases — the same `DecodeStream` instance
    /// is fed every generated token in order, so HuggingFace's
    /// prefix-stability invariant (`tokenizers-0.23.1/src/tokenizer/
    /// mod.rs:1119-1126`) holds end-to-end.
    pub(super) decoder: Option<crate::tokenizer::StreamingDecoder<'static>>,
    /// Thinking-phase streaming scanner. Lazily constructed from
    /// `ReasoningParser::create_thinking_scanner()` on the first
    /// thinking chunk; reset to `None` on `enter_content` so a
    /// hallucinated `<think>` re-open from Content rebuilds a
    /// fresh scanner on the next thinking chunk.
    ///
    /// Owns BOTH end-tag (`</think>`) detection across chunks AND
    /// model-specific leak-pattern cleanup, with the same stateful
    /// safe-emit idiom used by `StreamingToolDetector` and
    /// `sanitize_content_chunk` in the Content phase. Replaces the
    /// previous `pending_pre_tag` hold-drain buffer + stateless
    /// `cleanup_reasoning_leaks` cascade.
    pub(super) thinking_scanner:
        Option<Box<dyn crate::reasoning_parser::ThinkingScanner>>,
    /// Set true once the first non-whitespace `content` byte has been
    /// emitted (or once Content state was entered without prior
    /// thinking). Until then, Content-state deltas are `trim_start`ed
    /// so the `</think>\n\n` boundary doesn't produce a leading blank
    /// line on the assistant bubble. Qwen3.5/3.6 emits `</think>` and
    /// the following `\n\n` as separate tokens, so trimming at the
    /// FSM transition isn't enough — we have to track across the
    /// first few Content steps too.
    pub(super) content_started: bool,
    /// Buffer used for stop-string matching across delta boundaries.
    pub(super) accumulated_content: String,
    /// Major FSM phase. Mutated via `enter_thinking` / `enter_content`
    /// / `mark_stopped`; queried via `is_thinking` / `is_stopped`.
    /// Helper functions that historically took `&mut stop_string_
    /// triggered: bool` (`bump_f12_tool_call_count`, `check_loop_
    /// watchdog`, etc.) operate on a local bool seeded from
    /// `is_stopped()`; the caller folds the post-call value back
    /// into `phase` via `mark_stopped()`. Local-bool bridge keeps
    /// the borrow checker happy on disjoint-field reborrows.
    phase: StreamPhase,
    /// Sanitiser state: suppressing content while waiting for a
    /// matching `</parameter>` close after an orphan `<parameter=`.
    pub(super) suppressing_param_leak: bool,
    /// Sanitiser state: currently inside a tool-call envelope opener
    /// (e.g. `<minimax:tool_call>`); inner `<invoke ...>` etc. are
    /// legitimate content while this is true.
    pub(super) inside_envelope: bool,
    /// Mirror of `inside_envelope` for the reasoning sanitiser.
    pub(super) reasoning_inside_envelope: bool,
    /// Tag-scan buffer for the content sanitiser.
    pub(super) tag_scan_buf: String,
    /// Sanitiser state for reasoning content (parallel to
    /// `suppressing_param_leak` above).
    pub(super) reasoning_suppressing_leak: bool,
    /// Tag-scan buffer for the reasoning sanitiser.
    pub(super) reasoning_tag_scan_buf: String,
    /// Repetition-loop watchdog: tail buffer for line-level
    /// duplicate detection.
    pub(super) loop_scan_buf: String,
    /// Set true when the watchdog or SimHash guard fires.
    pub(super) loop_watchdog_triggered: bool,
    /// F4: SimHash semantic-loop guard for paraphrased restarts.
    pub(super) simhash_guard: crate::loop_simhash::SimHashLoopGuard,
    /// F4: pending bytes accumulated until a sentence-boundary or
    /// 1KB force-flush triggers a `simhash_guard.check()`.
    pub(super) simhash_pending: String,
    /// F5: cross-flush tool-arg dedup (default thresholds).
    pub(super) tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup,
    /// F11: tighter within-response tool-arg dedup for the
    /// streaming `ToolCallEnd` path.
    pub(super) tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup,
    /// Per-streaming-toolcall accumulator keyed by `oa_idx`. Holds the
    /// upstream-minted id (so it can be replayed verbatim into the
    /// observability dump at end-of-stream, matching what the client
    /// saw on the SSE wire), the tool name, and args-so-far. The id
    /// flows through state from `handle_tool_call_start` (where the
    /// detector hands it to us via `tc_id`) to `handle_tool_call_end`
    /// (where the dump recorder reads it). Never re-derived from
    /// `idx` — re-derivation would diverge from the SSE-emitted id
    /// under interleaved requests, because the upstream id comes from
    /// a global counter while `idx` is per-request.
    pub(super) streaming_tool_args: HashMap<usize, StreamingToolCall>,
    /// F12: per-response total tool-call count.
    pub(super) tool_calls_emitted_count: usize,
    /// Bug-2 (OpenClaw 2026-05-08): per-tool-name consecutive-call
    /// guard. F11 keys on `(name, canonical_args)` and is defeated by
    /// runaway loops where the model varies args slightly each
    /// iteration (e.g. timestamps, sequence numbers, IDs). This
    /// counter trips whenever the same tool name fires in N
    /// successive `ToolCallEnd` events regardless of args drift,
    /// catching the `cron`+`exec` alternation pattern observed when
    /// the streaming detector did successfully classify the calls.
    /// `(last_name, run_length)`. `last_name = None` means the run
    /// was just broken by a different tool name.
    pub(super) name_run: Option<(String, u32)>,
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,

    // ── Observability accumulators ────────────────────────────────
    //
    // `dump_content` / `dump_reasoning_content` / `dump_tool_calls`
    // are a faithful, source-of-truth replay of what was emitted to
    // the client over SSE. They are populated by the `record_*`
    // methods at every SSE emit site, and assembled into a
    // `ChatCompletionResponse`-shaped body in `handle_done.rs` for
    // the `atlas::dump` observability event.
    //
    // This mirrors the non-streaming `chat_blocking.rs` path which
    // already dumps a full response — so the streaming and blocking
    // dumps now have the same JSON shape.

    /// Full content text emitted as SSE `content_chunk` events
    /// during this request. Populated by `record_content`.
    pub(super) dump_content: String,
    /// Full reasoning text emitted as SSE `reasoning_chunk` events
    /// during this request. Populated by `record_reasoning`.
    pub(super) dump_reasoning_content: String,
    /// Finalized tool calls emitted during this request, in emission
    /// order. Populated by `record_tool_call`.
    pub(super) dump_tool_calls: Vec<crate::tool_parser::ToolCall>,
}

impl StreamState {
    pub(super) fn new(tools_active: bool, enable_thinking: bool) -> Self {
        Self {
            decoder: None,
            thinking_scanner: None,
            // If thinking is disabled, the assistant turn opens
            // directly in Content state — there's no `</think>\n\n`
            // boundary to trim, so content_started begins `true`.
            content_started: !enable_thinking,
            accumulated_content: String::new(),
            phase: if enable_thinking {
                StreamPhase::Thinking
            } else {
                StreamPhase::Content
            },
            suppressing_param_leak: false,
            inside_envelope: false,
            reasoning_inside_envelope: false,
            tag_scan_buf: String::new(),
            reasoning_suppressing_leak: false,
            reasoning_tag_scan_buf: String::new(),
            loop_scan_buf: String::new(),
            loop_watchdog_triggered: false,
            simhash_guard: crate::loop_simhash::SimHashLoopGuard::new(),
            simhash_pending: String::new(),
            tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup::new(),
            tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup::with_params(4, 2, 3),
            streaming_tool_args: HashMap::new(),
            tool_calls_emitted_count: 0,
            name_run: None,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new())
            } else {
                None
            },
            dump_content: String::new(),
            dump_reasoning_content: String::new(),
            dump_tool_calls: Vec::new(),
        }
    }

    /// Record content bytes emitted on the `content` SSE channel.
    /// Called at every `content_chunk` SSE emit site.
    pub(super) fn record_content(&mut self, text: &str) {
        self.dump_content.push_str(text);
    }

    /// Record reasoning bytes emitted on the `reasoning_content`
    /// SSE channel. Called at every `reasoning_chunk` SSE emit site.
    pub(super) fn record_reasoning(&mut self, text: &str) {
        self.dump_reasoning_content.push_str(text);
    }

    /// Record a finalized tool call (emitted in full, or assembled
    /// from streaming `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd`
    /// events).
    pub(super) fn record_tool_call(&mut self, tc: crate::tool_parser::ToolCall) {
        self.dump_tool_calls.push(tc);
    }

    pub(super) fn phase(&self) -> StreamPhase {
        self.phase
    }

    pub(super) fn is_thinking(&self) -> bool {
        matches!(self.phase, StreamPhase::Thinking)
    }

    pub(super) fn is_stopped(&self) -> bool {
        matches!(self.phase, StreamPhase::Stopped)
    }

    /// Transition Thinking → Content. No-op from Content (idempotent
    /// re-entry on the same chunk) and from Stopped (terminal).
    /// Drops the thinking scanner (rebuilt lazily on the next
    /// Thinking re-entry).
    pub(super) fn enter_content(&mut self) {
        match self.phase {
            StreamPhase::Thinking => {
                self.phase = StreamPhase::Content;
                if let Some(det) = self.detector.as_mut() {
                    det.reset();
                }
                self.thinking_scanner = None;
            }
            StreamPhase::Content | StreamPhase::Stopped => {}
        }
    }

    /// Transition Content → Thinking, on a hallucinated `<think>`
    /// re-open mid-content. Resets the content-start trim; the
    /// thinking scanner is `None` here (cleared by `enter_content`)
    /// and gets lazily rebuilt on the next thinking chunk.
    pub(super) fn enter_thinking(&mut self) {
        match self.phase {
            StreamPhase::Content => {
                self.phase = StreamPhase::Thinking;
                self.content_started = false;
            }
            StreamPhase::Thinking | StreamPhase::Stopped => {}
        }
    }

    /// Move to the terminal Stopped phase. Subsequent emit-layer
    /// guards short-circuit; the `handle_done` arm still runs to
    /// emit the finish_reason and final usage block.
    pub(super) fn mark_stopped(&mut self) {
        self.phase = StreamPhase::Stopped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn tc(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: format!("call_{name}"),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[test]
    fn recorder_accumulates_content_across_calls() {
        let mut s = StreamState::new(false, false);
        s.record_content("Hello, ");
        s.record_content("world");
        s.record_content("!");
        assert_eq!(s.dump_content, "Hello, world!");
    }

    #[test]
    fn recorder_accumulates_reasoning_across_calls() {
        let mut s = StreamState::new(false, true);
        s.record_reasoning("first ");
        s.record_reasoning("second ");
        s.record_reasoning("third");
        assert_eq!(s.dump_reasoning_content, "first second third");
    }

    #[test]
    fn recorder_collects_tool_calls_in_emission_order() {
        let mut s = StreamState::new(true, false);
        s.record_tool_call(tc("a", "{\"x\":1}"));
        s.record_tool_call(tc("b", "{\"y\":2}"));
        s.record_tool_call(tc("c", "{}"));
        let names: Vec<&str> = s
            .dump_tool_calls
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        assert_eq!(s.dump_tool_calls[0].function.arguments, "{\"x\":1}");
        assert_eq!(s.dump_tool_calls[1].function.arguments, "{\"y\":2}");
        assert_eq!(s.dump_tool_calls[2].function.arguments, "{}");
    }

    #[test]
    fn recorders_start_empty() {
        let s = StreamState::new(false, false);
        assert!(s.dump_content.is_empty());
        assert!(s.dump_reasoning_content.is_empty());
        assert!(s.dump_tool_calls.is_empty());
    }
}
