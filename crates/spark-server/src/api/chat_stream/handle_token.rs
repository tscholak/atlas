// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the SSE events produced for this single token. Callers
// invoke `futures::stream::iter(...)` on the result to feed the
// `flat_map` output stream.

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::ctx::StreamCtx;
use super::state::{StreamPhase, StreamState};
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// Process one token. Returns the SSE events to forward to the
/// client (empty `Vec` is valid).
///
/// Single `DecodeStream` (created with `skip_special_tokens=false` so
/// `</end_tag>` arrives as literal text) feeds a two-state machine:
/// `Thinking` buffers decoded text and emits incremental
/// `reasoning_chunk` deltas, watching for the configured reasoning
/// parser's end-tag substring; on match, the pre-tag text is flushed,
/// the post-tag text is trimmed of leading whitespace, and the FSM
/// transitions to `Content` where the existing tool-detector +
/// sanitiser + watchdog pipeline runs unchanged. The thinking phase
/// uses `ReasoningParser::cleanup_reasoning_leaks` for any model-
/// specific quirk removal (stray protocol-tag fragments, role-word
/// repetition loops, etc.) — see the trait impl on
/// `QwenReasoningParser`.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> SseVec {
    let mut sse_events: SseVec = Vec::new();

    let decoder = state.decoder.get_or_insert_with(|| {
        // SAFETY: ctx.state (Arc<AppState>) is owned by the closure
        // and lives for its entire duration. The DecodeStream borrows
        // &Tokenizer from it. We extend the lifetime because the Arc
        // guarantees the tokenizer outlives the closure (and thus
        // the DecodeStream).
        let tokenizer_ref: &'static crate::tokenizer::ChatTokenizer =
            unsafe { &*(&ctx.state.tokenizer as *const crate::tokenizer::ChatTokenizer) };
        // `skip_special_tokens=false`: protocol markers like
        // `</think>` reach the state machine as literal text so we
        // can substring-match them deterministically. ChatML stop
        // specials (`<|im_start|>`, `<|im_end|>`) are registered in
        // `eos_tokens` at startup (see
        // `tokenizer_runtime.rs`) and terminate the response via the
        // sampler's regular EOS path — they never appear in the
        // decoded stream that reaches here.
        tokenizer_ref.streaming_decoder(false)
    });
    let chunk = match decoder.step(tok) {
        Ok(Some(s)) => s,
        Ok(None) => return sse_events,
        Err(e) => {
            tracing::warn!("Streaming decoder error: {e:?}");
            return sse_events;
        }
    };

    // ── Thinking state ───────────────────────────────────────────────
    // Buffer-and-substring-match the configured reasoning parser's
    // end-tag. The `cleanup_reasoning_leaks` + `sanitize_content_chunk`
    // pipeline runs on the SSE-emitted prefix only; the tail held back
    // for tag-straddle protection is re-considered on the next token.
    let delta = match state.phase() {
        StreamPhase::Content | StreamPhase::Stopped => chunk,
        StreamPhase::Thinking => match thinking_step(state, ctx, chunk, &mut sse_events) {
            Some(post_tag) => post_tag,
            None => return sse_events,
        },
    };

    let mut delta = delta;

    // ── Content phase ────────────────────────────────────────────────
    // Trim leading whitespace at the start of the Content phase. Qwen
    // emits `</think>` and the following `\n\n` as separate tokens, so
    // trimming at the FSM transition only catches the in-chunk case;
    // we have to keep trimming until the first non-whitespace delta
    // arrives.
    delta = trim_until_content_started(delta, &mut state.content_started);

    if delta.is_empty() {
        return sse_events;
    }

    // Multi-token stop sequences via string matching.
    if !ctx.stop_strings.is_empty() && !state.is_stopped() {
        state.accumulated_content.push_str(&delta);
        for stop_str in &ctx.stop_strings {
            if let Some(pos) = state.accumulated_content.find(stop_str.as_str()) {
                let content_before_stop = &state.accumulated_content[..pos];
                let already_emitted = state.accumulated_content.len() - delta.len();
                if pos > already_emitted {
                    delta = content_before_stop[already_emitted..].to_string();
                } else {
                    delta = String::new();
                }
                state.mark_stopped();
                break;
            }
        }
        if state.is_stopped() && delta.is_empty() {
            return sse_events;
        }
    }

    if state.is_stopped() {
        if !delta.is_empty() {
            state.record_content(&delta);
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, delta);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            sse_events.push(Ok(Event::default().data(json)));
        }
        return sse_events;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    emit_content(state, ctx, &text, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, idx);
                }
            }
        }
    } else {
        emit_content(state, ctx, &delta, &mut sse_events);
    }

    sse_events
}

/// Record + emit a `content_chunk` for the given text. No-op on empty.
fn emit_content(state: &mut StreamState, ctx: &StreamCtx, text: &str, sse_events: &mut SseVec) {
    if text.is_empty() {
        return;
    }
    state.record_content(text);
    let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, text.to_string());
    let json = serde_json::to_string(&chunk).unwrap_or_default();
    sse_events.push(Ok(Event::default().data(json)));
}

/// Process one decoder chunk while in Thinking phase.
///
/// Returns:
/// - `Some(post_tag_text)` when the end-tag was found and the FSM
///   transitioned to Content. The caller continues processing the
///   post-tag remainder as a content delta.
/// - `None` when nothing flows through to Content this token (either
///   the end-tag wasn't seen yet and we're still buffering, or there
///   was no post-tag text after the boundary).
///
/// `sse_events` accumulates `reasoning_content` SSE deltas emitted
/// during this step.
fn thinking_step(
    state: &mut StreamState,
    ctx: &StreamCtx,
    chunk: String,
    sse_events: &mut SseVec,
) -> Option<String> {
    // Lazily construct the model-specific thinking scanner. The
    // scanner owns end-tag detection AND model-specific leak-pattern
    // cleanup (Qwen3.5/3.6 hallucinated `<think>` re-opens, role-word
    // loops, stray tool-call XML, etc.), both with cross-chunk
    // awareness via the same safe-emit idiom used by
    // `StreamingToolDetector` in the Content phase.
    let Some(parser) = ctx.state.reasoning_parser.as_deref() else {
        // No reasoning parser configured. Treat the stream as
        // content-only from the start.
        state.enter_content();
        return Some(chunk);
    };
    let scanner = state
        .thinking_scanner
        .get_or_insert_with(|| parser.create_thinking_scanner());

    match scanner.process(&chunk) {
        crate::reasoning_parser::ThinkingScanResult::Continue { emit } => {
            if !emit.is_empty() && ctx.enable_thinking {
                emit_reasoning_sse(sse_events, state, ctx, &emit);
            }
            None
        }
        crate::reasoning_parser::ThinkingScanResult::Transition {
            final_reasoning,
            content_start,
        } => {
            if ctx.enable_thinking {
                if !final_reasoning.is_empty() {
                    emit_reasoning_sse(sse_events, state, ctx, &final_reasoning);
                }
            }
            state.enter_content();
            if content_start.is_empty() {
                None
            } else {
                Some(content_start)
            }
        }
    }
}

/// Thinking-phase SSE emit: record + push a `reasoning_chunk` event.
/// No-op on empty.
fn emit_reasoning_sse(
    sse_events: &mut SseVec,
    state: &mut StreamState,
    ctx: &StreamCtx,
    text: &str,
) {
    if text.is_empty() {
        return;
    }
    state.record_reasoning(text);
    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, text.to_string());
    let json = serde_json::to_string(&chunk).unwrap_or_default();
    sse_events.push(Ok(Event::default().data(json)));
}

/// Strip leading whitespace from the Content-phase delta until the
/// first non-whitespace byte is observed; flip `content_started` true
/// once that happens. This bridges the `</think>\n\n` boundary when
/// the tokenizer emits the special-token `</think>` and the trailing
/// `\n\n` on separate decoder steps — the FSM transition-time
/// `trim_start` only catches in-chunk whitespace.
fn trim_until_content_started(delta: String, content_started: &mut bool) -> String {
    if *content_started {
        return delta;
    }
    let trimmed = delta.trim_start();
    let out = if trimmed.len() < delta.len() {
        trimmed.to_string()
    } else {
        delta
    };
    if !out.is_empty() {
        *content_started = true;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_until_content_started_strips_leading_newlines_on_first_token() {
        let mut started = false;
        let out = trim_until_content_started("\n\nLet's".to_string(), &mut started);
        assert_eq!(out, "Let's");
        assert!(started, "first non-whitespace delta arms content_started");
    }

    #[test]
    fn trim_until_content_started_holds_when_chunk_is_pure_whitespace() {
        let mut started = false;
        let out = trim_until_content_started("\n".to_string(), &mut started);
        assert_eq!(out, "");
        assert!(
            !started,
            "pure-whitespace delta keeps the trim active for the next chunk"
        );
    }

    #[test]
    fn trim_until_content_started_passes_through_after_first_content() {
        let mut started = true;
        let out = trim_until_content_started("  spaces preserved".to_string(), &mut started);
        assert_eq!(
            out, "  spaces preserved",
            "interior whitespace must survive once content has started"
        );
    }

    #[test]
    fn trim_until_content_started_keeps_interior_whitespace_on_arming_delta() {
        let mut started = false;
        let out = trim_until_content_started("\nLet's\n  more".to_string(), &mut started);
        assert_eq!(out, "Let's\n  more");
        assert!(started);
    }
}
