// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming `/v1/chat/completions` SSE handler.
//!
//! The scheduler emits `StreamEvent::{Token, TokenWithLogprobs, Done,
//! Error}` over a tokio mpsc channel. Token events are fed into a
//! `chat_fsm::Stepper` which emits a canonical `FsmEvent` stream;
//! `Done` triggers `Stepper::flush` to drain remaining state and emit
//! the terminal `Stopped` event. The `StreamingAdapter` translates
//! every event into OpenAI-compatible `ChatCompletionChunk` SSE
//! chunks and assembles the `--dump` body via a `ChoiceBuilder`.
//!
//! Sub-files:
//! - `ctx`          — minimal shared state for the error arm
//! - `adapter`      — `FsmEvent → SSE` translation + dump assembly
//! - `handle_error` — Error arm (unchanged)

mod adapter;
mod ctx;
mod handle_error;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::ChatCompletionChunk;

use super::chat_fsm::stepper::{Stepper, StepperConfig, extend_tokenizer_lifetime};
use super::inference_types::{GrammarSpec, InferenceRequest, StreamEvent};

use adapter::{StreamingAdapter, UsageInputs};
use ctx::StreamCtx;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn chat_completions_stream(
    state: Arc<AppState>,
    request_id: crate::request_id::RequestId,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    image_pixels: Vec<(Vec<f32>, usize, usize)>,
    max_tokens: usize,
    min_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    lz_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
    tool_choice_required: bool,
    stop_tokens: Vec<u32>,
    grammar_spec: Option<GrammarSpec>,
    seed: Option<u64>,
    top_logprobs: Option<u8>,
    timeout_at: Option<std::time::Instant>,
    stop_strings: Vec<String>,
    req_stream_include_usage: bool,
    req_service_tier: Option<String>,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    req_ctx: Option<crate::rate_limiter::RequestContext>,
    dump_seq: Option<u64>,
) -> Result<Response, (StatusCode, String)> {
    // service_tier + metadata are request echoes only; the chat-completion-
    // chunk schema doesn't carry them, but we surface them via the final
    // usage block when `include_usage=true`. Accept and retain for future
    // wiring — see gap #18/#19 in the OpenAI compat plan.
    let _ = (&req_service_tier, &req_metadata);
    // Channel capacity sized for ~30s of decode at 50 tok/s. The previous
    // 64-slot buffer would fill in <2s under any HTTP-flush stall and silently
    // drop the seq via emit_step.rs's `try_send().is_err()` (now fixed to
    // discriminate Full from Closed). Larger capacity = fewer Full→blocking_send
    // round-trips in the steady state.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();

    // Scheduler tracks thinking only when the template actually opens it.
    let scheduler_thinking = enable_thinking;
    let request = InferenceRequest::Streaming {
        request_id: request_id.as_str().to_string(),
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        min_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking: scheduler_thinking,
        thinking_budget,
        require_tool_call: tool_choice_required,
        disable_mtp: false,
        grammar_spec,
        seed,
        top_logprobs,
        timeout_at,
        token_tx,
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_chunk_id();
    let model_name = state.model_name.clone();

    // First chunk: role announcement
    let role_chunk = ChatCompletionChunk::role_chunk(&model_name, &chunk_id);
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();

    let stream_ctx = StreamCtx {
        state: state.clone(),
        req_ctx: req_ctx.clone(),
    };

    // Stepper owns the decoder / scanner / detector / stop predicate.
    let tokenizer = extend_tokenizer_lifetime(&state.tokenizer);
    let stepper_cfg = StepperConfig {
        enable_thinking,
        tools_active,
        stop_strings,
        reasoning_parser: state.reasoning_parser.as_deref().map(|p| {
            // Same lifetime extension trick as the tokenizer — the
            // adapter and stepper outlive the parser borrow because
            // `Arc<AppState>` is held by the flat_map closure.
            unsafe { std::mem::transmute::<&dyn crate::reasoning_parser::ReasoningParser, &'static dyn crate::reasoning_parser::ReasoningParser>(p) }
        }),
    };
    let mut stepper = Stepper::new(tokenizer, stepper_cfg);

    let mut adapter = StreamingAdapter::new(
        state.clone(),
        model_name.clone(),
        chunk_id.clone(),
        request_id,
        dump_seq,
        req_stream_include_usage,
        req_ctx,
        prompt_len,
    );

    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let mut sse_events: Vec<Result<Event, std::convert::Infallible>> = Vec::new();
        match event {
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                for ev in stepper.step_token(tok) {
                    adapter.translate(ev, &mut sse_events);
                }
            }
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                accepted_prediction_tokens,
                rejected_prediction_tokens,
            } => {
                adapter.set_usage_inputs(UsageInputs {
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                    accepted_prediction_tokens,
                    rejected_prediction_tokens,
                });
                for ev in stepper.flush(finish_reason) {
                    adapter.translate(ev, &mut sse_events);
                }
            }
            StreamEvent::Error(msg) => {
                sse_events.extend(handle_error::handle_error(&stream_ctx, msg));
            }
        }
        futures::stream::iter(sse_events)
    });

    // Prepend role chunk, append [DONE] sentinel
    let role_event = futures::stream::once(async move { Ok(Event::default().data(role_json)) });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = role_event.chain(token_stream).chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
