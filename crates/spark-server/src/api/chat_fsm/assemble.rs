// SPDX-License-Identifier: AGPL-3.0-only

//! Shared `ChatCompletionResponse` constructor + per-choice assembler.
//!
//! `assemble_chat_response` is the single point of truth for the
//! OpenAI-shaped chat-completion body. The streaming adapter calls it
//! to build the journald dump; the blocking adapter calls it to build
//! the HTTP response body.
//!
//! `assemble_choice` is the blocking-path entry point: it drives a
//! fresh `Stepper` over the scheduler-returned `InferenceResponse`'s
//! `output_tokens` and materialises one `ChatChoice` per call. It is
//! NOT used by the streaming path (streaming drives the Stepper
//! one-token-at-a-time from the mpsc receiver).

use std::collections::HashMap;

use crate::AppState;
use crate::openai::{ChatChoice, ChatCompletionResponse, Usage, unix_timestamp};

use super::choice_builder::ChoiceBuilder;
use super::events::{FsmEvent, StopReason};
use super::stepper::{Stepper, StepperConfig, extend_tokenizer_lifetime};

/// Canonical OpenAI `ChatCompletionResponse` constructor. Called by
/// both the streaming dump path and the blocking response path.
#[allow(clippy::too_many_arguments)]
pub fn assemble_chat_response(
    id: String,
    model: String,
    created: u64,
    choices: Vec<ChatChoice>,
    usage: Usage,
    service_tier: Option<String>,
    metadata: Option<HashMap<String, String>>,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model,
        system_fingerprint: Some("fp_atlas".to_string()),
        choices,
        usage,
        service_tier,
        metadata,
    }
}

/// Drive a fresh Stepper over `response.output_tokens` and build one
/// `ChatChoice`. The blocking path calls this once per `choice_idx`
/// in its `for choice_idx in 0..n` loop.
pub fn assemble_choice(
    state: &AppState,
    response: &super::super::inference_types::InferenceResponse,
    cfg: StepperConfig,
    choice_idx: usize,
    logprobs: Option<crate::openai::ChoiceLogprobs>,
) -> ChatChoice {
    let tokenizer = extend_tokenizer_lifetime(&state.tokenizer);
    let mut stepper = Stepper::new(tokenizer, cfg);
    let mut builder = ChoiceBuilder::new(choice_idx);

    let mut stopped = false;
    for &tok in &response.output_tokens {
        if stopped {
            break;
        }
        for ev in stepper.step_token(tok) {
            if matches!(ev, FsmEvent::Stopped { .. }) {
                stopped = true;
            }
            builder.apply(ev);
        }
    }
    if !stopped {
        for ev in stepper.flush(response.finish_reason.clone()) {
            builder.apply(ev);
        }
    }

    builder.into_chat_choice(logprobs)
}

/// Map an FSM `StopReason` (plus the assembled-choice fact "did any
/// tool calls fire?") onto the OpenAI-visible `finish_reason` string.
///
/// Precedence:
///   1. Any tool calls fired → `"tool_calls"` (regardless of stop_reason).
///   2. `StopReason::GrammarTerminated` → `"tool_calls"` (the grammar's
///      `stop_after_first` only fires under `tool_choice: required`,
///      where OpenAI expects `"tool_calls"`).
///   3. `StopReason::StopString { .. }` → `"stop"` (OpenAI doesn't
///      surface which stop matched).
///   4. `StopReason::Upstream(s)` → pass `s` through (`"stop"`,
///      `"length"`, `"timeout"`, etc.).
///   5. `None` is unreachable in practice (`flush()` always emits
///      `Stopped`); we return `"stop"` rather than panic so a
///      degenerate case still surfaces a valid OpenAI string.
pub fn compute_finish_reason(stop_reason: Option<&StopReason>, has_tool_calls: bool) -> String {
    if has_tool_calls {
        return "tool_calls".to_string();
    }
    match stop_reason {
        Some(StopReason::GrammarTerminated) => "tool_calls".to_string(),
        Some(StopReason::StopString { .. }) => "stop".to_string(),
        Some(StopReason::Upstream(s)) => s.clone(),
        None => "stop".to_string(),
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
pub fn build_usage(
    prompt_tokens: usize,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    accepted_prediction_tokens: u32,
    rejected_prediction_tokens: u32,
    request_id: String,
) -> Usage {
    let tps = if decode_time_ms > 0.0 && completion_tokens > 0 {
        (completion_tokens.saturating_sub(1)) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
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
        request_id,
    }
}

#[allow(dead_code)]
fn _unused_created() -> u64 {
    unix_timestamp()
}
