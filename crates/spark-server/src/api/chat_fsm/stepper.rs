// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token FSM driving the chat-completion event stream.
//!
//! The Stepper owns the streaming decoder, the thinking scanner, the
//! tool-call detector, the stop-string predicate, and the per-tool-call
//! argument accumulator. Both the streaming and blocking adapters
//! construct one Stepper per response, feed it the scheduler-emitted
//! token ids via `step_token`, and drain remaining buffered state via
//! `flush` at end-of-stream.
//!
//! See `Stage 3` in the plan file
//! (`/Users/tscholak/.claude/plans/i-ve-deployed-the-heim-snazzy-swan.md`)
//! for the design rationale.

use std::collections::HashMap;

use crate::reasoning_parser::{ReasoningParser, ThinkingScanResult, ThinkingScanner};
use crate::tokenizer::{ChatTokenizer, StreamingDecoder};
use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

use super::events::{FsmEvent, StopReason};
use super::stop_predicate::{StopOutcome, StopPredicate};

/// Read-only Stepper construction inputs. Holds borrows of the
/// app-state's reasoning parser (when configured) and the user's
/// stop strings; the Stepper takes ownership of values it mutates.
pub struct StepperConfig {
    pub enable_thinking: bool,
    pub tools_active: bool,
    pub stop_strings: Vec<String>,
    /// Optional reasoning parser. When `None`, the Stepper enters the
    /// `Content` phase immediately and never instantiates a thinking
    /// scanner. When `Some`, the Stepper drives
    /// `parser.create_thinking_scanner()` until the first end-tag
    /// boundary, then transitions to Content.
    pub reasoning_parser: Option<&'static dyn ReasoningParser>,
}

/// Major FSM phase. Private — adapters consume `FsmEvent`s, not phases.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Phase {
    Thinking,
    Content,
}

/// Per-tool-call accumulator. Lives in `Stepper::tool_acc` between
/// `ToolCallStart` and `ToolCallEnd` so the `ToolCallEnd` event can
/// carry the verbatim id minted by the detector (B7 invariant).
#[derive(Debug, Clone)]
pub(crate) struct ToolCallAcc {
    pub id: String,
    pub name: String,
    pub args: String,
}

pub struct Stepper {
    decoder: StreamingDecoder<'static>,
    phase: Phase,
    content_started: bool,
    thinking_scanner: Option<Box<dyn ThinkingScanner>>,
    detector: Option<StreamingToolDetector>,
    stop_pred: Option<StopPredicate>,
    tool_acc: HashMap<usize, ToolCallAcc>,
    stopped: bool,
}

impl Stepper {
    /// Construct a fresh Stepper. The caller is responsible for
    /// extending the tokenizer borrow to `'static` (see
    /// `extend_tokenizer_lifetime` below). The Stepper holds the
    /// `'static` reference for the duration of its lifetime; the
    /// caller MUST ensure the underlying `Arc<AppState>` outlives the
    /// Stepper.
    pub fn new(tokenizer: &'static ChatTokenizer, cfg: StepperConfig) -> Self {
        // `skip_special_tokens=false`: protocol markers like
        // `</think>` reach the FSM as literal decoded text so the
        // thinking scanner can substring-match them. ChatML stop
        // specials (`<|im_start|>`, `<|im_end|>`) are registered as
        // `eos_tokens` upstream and never appear in the stream.
        let decoder = tokenizer.streaming_decoder(false);

        let (phase, thinking_scanner) =
            if cfg.enable_thinking && cfg.reasoning_parser.is_some() {
                let scanner = cfg
                    .reasoning_parser
                    .expect("checked is_some")
                    .create_thinking_scanner();
                (Phase::Thinking, Some(scanner))
            } else {
                (Phase::Content, None)
            };

        let detector = if cfg.tools_active {
            Some(StreamingToolDetector::new())
        } else {
            None
        };

        let stop_pred = if cfg.stop_strings.is_empty() {
            None
        } else {
            Some(StopPredicate::new(cfg.stop_strings))
        };

        Self {
            decoder,
            phase,
            // If thinking is disabled, the assistant turn opens directly
            // in Content state — no `</think>\n\n` boundary to trim.
            content_started: !matches!(phase, Phase::Thinking),
            thinking_scanner,
            detector,
            stop_pred,
            tool_acc: HashMap::new(),
            stopped: false,
        }
    }

    /// Feed one decoded token id. Returns zero or more events.
    /// Once the Stepper has emitted `Stopped`, subsequent calls return
    /// an empty vec.
    pub fn step_token(&mut self, tok: u32) -> Vec<FsmEvent> {
        let mut out = Vec::new();
        if self.stopped {
            return out;
        }
        let chunk = match self.decoder.step(tok) {
            Ok(Some(s)) => s,
            Ok(None) => return out,
            Err(e) => {
                tracing::warn!("Streaming decoder error: {e:?}");
                return out;
            }
        };
        self.feed_chunk(chunk, &mut out);
        out
    }

    /// End-of-stream: drain detector + thinking-scanner tail buffers,
    /// then emit `Stopped { Upstream(upstream_finish_reason) }`. If a
    /// stop string or grammar termination already fired during
    /// `step_token`, `flush` is a no-op (it returns empty because the
    /// `Stopped` event was already emitted there).
    pub fn flush(&mut self, upstream_finish_reason: String) -> Vec<FsmEvent> {
        let mut out = Vec::new();
        if self.stopped {
            return out;
        }
        // Drain the thinking scanner tail (only possible if we're
        // still in Thinking when EOS arrives — e.g. max_tokens hit
        // before `</think>`).
        if let Some(mut scanner) = self.thinking_scanner.take() {
            let tail = scanner.flush();
            if !tail.is_empty() {
                out.push(FsmEvent::ReasoningDelta(tail));
            }
        }
        // Drain the detector tail. Detector flush may emit any of the
        // Content / ToolCall* variants per its public contract.
        if let Some(det) = self.detector.as_mut() {
            let outputs = det.flush();
            self.dispatch_detector_outputs(outputs, &mut out);
            if self.stopped {
                return out;
            }
        }
        out.push(FsmEvent::Stopped {
            reason: StopReason::Upstream(upstream_finish_reason),
        });
        self.stopped = true;
        out
    }

    fn feed_chunk(&mut self, chunk: String, out: &mut Vec<FsmEvent>) {
        match self.phase {
            Phase::Thinking => self.feed_thinking(chunk, out),
            Phase::Content => self.feed_content(chunk, out),
        }
    }

    fn feed_thinking(&mut self, chunk: String, out: &mut Vec<FsmEvent>) {
        let Some(scanner) = self.thinking_scanner.as_mut() else {
            // Reasoning parser was None at construction; this branch
            // is unreachable because we'd already be in Content phase.
            // Defend with an explicit transition so future refactors
            // don't trip on it.
            self.phase = Phase::Content;
            self.feed_content(chunk, out);
            return;
        };
        match scanner.process(&chunk) {
            ThinkingScanResult::Continue { emit } => {
                if !emit.is_empty() {
                    out.push(FsmEvent::ReasoningDelta(emit));
                }
            }
            ThinkingScanResult::Transition {
                final_reasoning,
                content_start,
            } => {
                if !final_reasoning.is_empty() {
                    out.push(FsmEvent::ReasoningDelta(final_reasoning));
                }
                // Phase transition: drop the scanner, reset the
                // detector so any thinking-era tag fragments in its
                // buffer don't trigger spurious tool detection, reset
                // the content-leading-whitespace trim.
                self.thinking_scanner = None;
                if let Some(det) = self.detector.as_mut() {
                    det.reset();
                }
                self.content_started = false;
                self.phase = Phase::Content;
                if !content_start.is_empty() {
                    self.feed_content(content_start, out);
                }
            }
        }
    }

    fn feed_content(&mut self, chunk: String, out: &mut Vec<FsmEvent>) {
        // Trim leading whitespace until the first non-whitespace byte
        // arrives. Qwen3.x emits `</think>` and the trailing `\n\n` on
        // separate tokens, so the FSM-transition-time trim only catches
        // in-chunk whitespace; we keep trimming until we see content.
        let delta = if self.content_started {
            chunk
        } else {
            let trimmed = chunk.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.content_started = true;
            trimmed.to_string()
        };

        // Detector-active vs raw-content branches.
        if let Some(det) = self.detector.as_mut() {
            let outputs = det.process(&delta);
            self.dispatch_detector_outputs(outputs, out);
        } else {
            self.emit_content_with_stop_check(delta, out);
        }
    }

    fn dispatch_detector_outputs(
        &mut self,
        outputs: Vec<DetectorOutput>,
        out: &mut Vec<FsmEvent>,
    ) {
        for output in outputs {
            if self.stopped {
                return;
            }
            match output {
                DetectorOutput::Content(text) => {
                    if !text.is_empty() {
                        self.emit_content_with_stop_check(text, out);
                    }
                }
                DetectorOutput::ToolCall(tc, idx) => {
                    // Single-shot complete call (Mistral multi-`<invoke>`,
                    // bare-function salvage). Synthesise the
                    // Start/ArgDelta/End triple so the adapter has ONE
                    // assembly path.
                    out.push(FsmEvent::ToolCallStart {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        idx,
                    });
                    self.tool_acc.insert(
                        idx,
                        ToolCallAcc {
                            id: tc.id,
                            name: tc.function.name,
                            args: tc.function.arguments.clone(),
                        },
                    );
                    if !tc.function.arguments.is_empty() {
                        out.push(FsmEvent::ToolCallArgDelta {
                            args: tc.function.arguments,
                            idx,
                        });
                    }
                    out.push(FsmEvent::ToolCallEnd { idx });
                }
                DetectorOutput::ToolCallStart { id, name, idx } => {
                    self.tool_acc.insert(
                        idx,
                        ToolCallAcc {
                            id: id.clone(),
                            name: name.clone(),
                            args: String::new(),
                        },
                    );
                    out.push(FsmEvent::ToolCallStart { id, name, idx });
                }
                DetectorOutput::ToolCallDelta { args, idx } => {
                    if let Some(acc) = self.tool_acc.get_mut(&idx) {
                        acc.args.push_str(&args);
                    }
                    if !args.is_empty() {
                        out.push(FsmEvent::ToolCallArgDelta { args, idx });
                    }
                }
                DetectorOutput::ToolCallEnd { idx } => {
                    out.push(FsmEvent::ToolCallEnd { idx });
                }
            }
        }
    }

    fn emit_content_with_stop_check(&mut self, text: String, out: &mut Vec<FsmEvent>) {
        let Some(stop_pred) = self.stop_pred.as_mut() else {
            if !text.is_empty() {
                out.push(FsmEvent::ContentDelta(text));
            }
            return;
        };
        match stop_pred.feed(&text) {
            StopOutcome::Pass => {
                if !text.is_empty() {
                    out.push(FsmEvent::ContentDelta(text));
                }
            }
            StopOutcome::Fire {
                safe_prefix,
                matched,
            } => {
                if !safe_prefix.is_empty() {
                    out.push(FsmEvent::ContentDelta(safe_prefix));
                }
                out.push(FsmEvent::Stopped {
                    reason: StopReason::StopString { matched },
                });
                self.stopped = true;
            }
        }
    }

    /// Lookup the current accumulated args for a tool-call idx. The
    /// blocking adapter does NOT use this — it accumulates args from
    /// the event stream itself. Exposed for forensic / observability
    /// callers (e.g. log a tool call at `ToolCallEnd` from the args
    /// preview).
    pub(crate) fn tool_acc(&self, idx: usize) -> Option<&ToolCallAcc> {
        self.tool_acc.get(&idx)
    }
}

/// Extend the lifetime of a `&ChatTokenizer` borrow to `'static`.
///
/// SAFETY: the caller MUST ensure the underlying `Arc<AppState>` (or
/// whatever owns the tokenizer) outlives the `Stepper`. Both the
/// streaming adapter (which holds the `Arc` in the `flat_map` closure)
/// and the blocking adapter (which holds it across the per-choice
/// loop) satisfy this. Centralised in one helper so the unsafe is
/// auditable in one place.
pub fn extend_tokenizer_lifetime(t: &ChatTokenizer) -> &'static ChatTokenizer {
    unsafe { &*(t as *const ChatTokenizer) }
}
