// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Done { ... }` arm of the streaming `flat_map`
// closure (originally ~396 LoC).

use axum::response::sse::Event;

use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser;

use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    accepted_prediction_tokens: u32,
    rejected_prediction_tokens: u32,
) -> SseVec {
    let mut sse_events: SseVec = Vec::new();

    // ── Detector flush ──────────────────────────────────────────────
    if state.detector.is_some() {
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.flush()
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if !text.is_empty() {
                        state.record_content(&text);
                        let chunk =
                            ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, text);
                        sse_events.push(Ok(Event::default()
                            .data(serde_json::to_string(&chunk).unwrap_or_default())));
                    }
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
    }

    // ── Thinking scanner tail flush ─────────────────────────────────
    // Covers EOS during the Thinking phase (max_tokens hit before
    // `</think>` arrived). Drains any bytes the scanner was holding
    // back for safe-emit and emits them as the final reasoning delta.
    if let Some(mut scanner) = state.thinking_scanner.take() {
        let scanner_tail = scanner.flush();
        if !scanner_tail.is_empty() {
            state.record_reasoning(&scanner_tail);
            let chunk =
                ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, scanner_tail);
            sse_events.push(Ok(
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            ));
        }
    }

    // ── Usage block ─────────────────────────────────────────────────
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: accepted_prediction_tokens as usize,
            rejected_prediction_tokens: rejected_prediction_tokens as usize,
        }),
        time_to_first_token_ms,
        response_tokens_per_second: tps,
        request_id: ctx.request_id.as_str().to_string(),
    };

    let fr = if state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) {
        "tool_calls"
    } else {
        finish_reason.as_str()
    };

    // Usage emission strategy.
    let emit_separate_usage = ctx.req_stream_include_usage;
    let usage_for_dump = usage.clone();
    if emit_separate_usage {
        let usage_chunk = ChatCompletionChunk::usage_only_chunk(&ctx.model, &ctx.id, usage.clone());
        let json = serde_json::to_string(&usage_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
        let final_chunk = ChatCompletionChunk::final_chunk_no_usage(&ctx.model, &ctx.id, fr);
        let json = serde_json::to_string(&final_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    } else {
        let chunk = ChatCompletionChunk::done_chunk(&ctx.model, &ctx.id, fr, usage);
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    }

    // Metrics.
    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);

    // Rate-limit true-up.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // --dump response entry — emit only when the atlas::dump target
    // is enabled (avoids the body assembly when no subscriber wants
    // it). The body shape matches `chat_blocking.rs`'s response dump
    // (`ChatCompletionResponse`), populated from per-channel
    // accumulators on `state` (`dump_content`, `dump_reasoning_content`,
    // `dump_tool_calls`) that record every byte and structured value
    // emitted to the client over SSE. This makes the streaming dump
    // a faithful replay of what the client received, parity with the
    // blocking dump, and forensically inspectable in journald.
    if let Some(seq) = ctx.dump_seq
        && tracing::event_enabled!(target: "atlas::dump", tracing::Level::INFO)
    {
        let message = crate::openai::ChatMessage {
            role: "assistant".to_string(),
            reasoning_content: if state.dump_reasoning_content.is_empty() {
                None
            } else {
                Some(state.dump_reasoning_content.clone())
            },
            reasoning: None,
            content: if state.dump_content.is_empty() {
                None
            } else {
                Some(state.dump_content.clone())
            },
            tool_calls: if state.dump_tool_calls.is_empty() {
                None
            } else {
                Some(state.dump_tool_calls.clone())
            },
            annotations: None,
            refusal: None,
        };
        let response = crate::openai::ChatCompletionResponse {
            id: ctx.id.clone(),
            object: "chat.completion".to_string(),
            created: crate::openai::unix_timestamp(),
            model: ctx.model.clone(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![crate::openai::ChatChoice {
                index: 0,
                message,
                finish_reason: fr.to_string(),
                logprobs: None,
            }],
            usage: usage_for_dump,
            service_tier: None,
            metadata: None,
        };
        crate::request_dumper::dump_response(
            "/v1/chat/completions",
            seq,
            ctx.request_id.as_str(),
            &response,
            true,
        );
    }

    sse_events
}
