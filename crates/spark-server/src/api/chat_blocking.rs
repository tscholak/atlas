// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking (non-streaming) `/v1/chat/completions` path. Extracted from
//! `chat_completions_inner` (refactor wave-4e) to keep `chat.rs` under
//! the 500 LoC cap. Supports `n >= 1` (multiple choices per request) by
//! looping the scheduler send + decode + tool-parse pipeline once per
//! choice index.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, Usage};

use super::chat_fsm::assemble::assemble_choice;
use super::chat_fsm::stepper::StepperConfig;
use super::compact::openai_error_response;
use super::inference_types::{GrammarSpec, InferenceRequest};

pub(super) struct BlockingPathArgs {
    pub state: Arc<AppState>,
    pub req: ChatCompletionRequest,
    pub req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pub dump_seq: Option<u64>,
    /// Per-request identifier (Phase D) — threaded onto every
    /// `InferenceRequest::Blocking` we push to the scheduler so the
    /// resulting `ActiveSeq` carries it for per-tick log attribution.
    pub request_id: crate::request_id::RequestId,
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub tools_active: bool,
    pub tool_choice_required: bool,
    pub grammar_spec: Option<GrammarSpec>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<std::time::Instant>,
    pub prompt_len: usize,
}

pub(super) async fn run_blocking_path(args: BlockingPathArgs) -> Response {
    let BlockingPathArgs {
        state,
        req,
        req_ctx,
        dump_seq,
        request_id,
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
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
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        grammar_spec,
        top_logprobs,
        timeout_at,
        prompt_len,
    } = args;

    let n = req.n.max(1);
    let mut all_choices: Vec<crate::openai::ChatChoice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;
    let mut first_ttft = 0.0f64;
    let mut last_decode_time_ms = 0.0f64;
    let mut total_reasoning_tokens = 0u32;
    let mut total_cached_prompt_tokens = 0u32;
    // Phase C: accumulate MTP/spec accept+reject across all `n`
    // choices in the same request (n>1 with speculative decode emits
    // multiple sequences; each carries its own counters from the
    // scheduler).
    let mut total_accepted_prediction_tokens = 0u32;
    let mut total_rejected_prediction_tokens = 0u32;

    for choice_idx in 0..n {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            request_id: request_id.as_str().to_string(),
            prompt_tokens: prompt_tokens.clone(),
            session_hash,
            image_pixels: if choice_idx == 0 {
                image_pixels.clone()
            } else {
                Vec::new()
            },
            max_tokens,
            min_tokens: req.min_tokens,
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
            logit_bias: logit_bias.clone(),
            stop_tokens: stop_tokens.clone(),
            enable_thinking,
            thinking_budget,
            require_tool_call: tool_choice_required,
            disable_mtp: false,
            grammar_spec: grammar_spec.clone(),
            seed: req.seed.map(|s| s.wrapping_add(choice_idx as u64)),
            top_logprobs,
            timeout_at,
            response_tx: tx,
        };

        if state.request_tx.send(request).await.is_err() {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Scheduler queue full".to_string(),
            );
        }

        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {e}"),
                );
            }
            Err(_) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Inference cancelled".to_string(),
                );
            }
        };

        if choice_idx == 0 {
            first_ttft = response.time_to_first_token_ms;
        }
        last_decode_time_ms = response.decode_time_ms;

        let num_completion = response.output_tokens.len();
        total_completion_tokens += num_completion;
        total_reasoning_tokens += response.reasoning_tokens;
        // cached_prompt_tokens is a per-request prefix-cache hit count; for
        // n>1 we only charge once (same prompt reused).
        total_cached_prompt_tokens = total_cached_prompt_tokens.max(response.cached_prompt_tokens);
        total_accepted_prediction_tokens += response.accepted_prediction_tokens;
        total_rejected_prediction_tokens += response.rejected_prediction_tokens;

        // Drive the unified chat_fsm Stepper over this choice's
        // output_tokens. Reasoning split, stop-string trimming, and
        // tool-call extraction all flow through ONE FSM that the
        // streaming path also drives. The blocking-path-specific
        // `decode_response_text` / `strip_stop_sequences` /
        // `build_choice_message` / `parse_tool_calls` quartet is
        // replaced by `assemble_choice` + a `StepperConfig`.
        let stepper_cfg = StepperConfig {
            enable_thinking,
            tools_active,
            stop_strings: req.stop.clone(),
            reasoning_parser: state.reasoning_parser.as_deref().map(|p| {
                // SAFETY: the `Arc<AppState>` is held in the
                // surrounding `run_blocking_path` scope; it outlives
                // the per-choice Stepper. Centralised lifetime
                // extension lives in `chat_fsm::stepper`.
                unsafe {
                    std::mem::transmute::<
                        &dyn crate::reasoning_parser::ReasoningParser,
                        &'static dyn crate::reasoning_parser::ReasoningParser,
                    >(p)
                }
            }),
        };
        let logprobs = build_logprobs(&state, &response);
        let choice = assemble_choice(&state, &response, stepper_cfg, choice_idx, logprobs);
        all_choices.push(choice);
    }

    finalize_response(
        state,
        req,
        req_ctx,
        dump_seq,
        request_id,
        all_choices,
        total_completion_tokens,
        first_ttft,
        last_decode_time_ms,
        total_reasoning_tokens,
        total_cached_prompt_tokens,
        total_accepted_prediction_tokens,
        total_rejected_prediction_tokens,
        prompt_len,
    )
}

/// Convert internal logprobs to OpenAI `ChoiceLogprobs` format.
fn build_logprobs(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
) -> Option<crate::openai::ChoiceLogprobs> {
    if response.logprobs.is_empty() {
        return None;
    }
    Some(crate::openai::ChoiceLogprobs {
        content: response
            .logprobs
            .iter()
            .map(|lp| {
                let token_str = state.tokenizer.decode(&[lp.token_id]).unwrap_or_default();
                crate::openai::TokenLogprobInfo {
                    token: token_str,
                    logprob: lp.logprob,
                    bytes: None,
                    top_logprobs: lp
                        .top
                        .iter()
                        .map(|&(tid, lp_val)| crate::openai::TopLogprob {
                            token: state.tokenizer.decode(&[tid]).unwrap_or_default(),
                            logprob: lp_val,
                            bytes: None,
                        })
                        .collect(),
                }
            })
            .collect(),
    })
}

/// Build the final `ChatCompletionResponse` plus metrics, store, and
/// rate-limit refund. Returns the JSON-encoded HTTP response.
#[allow(clippy::too_many_arguments)]
fn finalize_response(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    dump_seq: Option<u64>,
    request_id: crate::request_id::RequestId,
    all_choices: Vec<crate::openai::ChatChoice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    total_accepted_prediction_tokens: u32,
    total_rejected_prediction_tokens: u32,
    prompt_len: usize,
) -> Response {
    let tokens_per_second = if last_decode_time_ms > 0.0 && total_completion_tokens > 0 {
        (total_completion_tokens.saturating_sub(1)) as f64 / (last_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        total_tokens: prompt_len + total_completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: total_cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: total_reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: total_accepted_prediction_tokens as usize,
            rejected_prediction_tokens: total_rejected_prediction_tokens as usize,
        }),
        time_to_first_token_ms: first_ttft,
        response_tokens_per_second: tokens_per_second,
        request_id: request_id.as_str().to_string(),
    };

    let completion_id = format!("chatcmpl-{}", crate::openai::uuid_v4());
    let created_at = crate::openai::unix_timestamp();
    let completion = ChatCompletionResponse {
        id: completion_id.clone(),
        object: "chat.completion".to_string(),
        created: created_at,
        model: state.model_name.clone(),
        system_fingerprint: Some("fp_atlas".to_string()),
        choices: all_choices,
        usage: usage.clone(),
        service_tier: req.service_tier.clone(),
        metadata: req.metadata.clone(),
    };

    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(first_ttft / 1000.0);

    // Completion-storage backend: when `store: true`, persist the
    // serialized body so a subsequent GET /v1/chat/completions/{id}
    // can return it. Bounded LRU + TTL in response_store.
    if req.store.unwrap_or(false)
        && let Ok(body) = serde_json::to_value(&completion)
    {
        state
            .response_store
            .insert(crate::response_store::StoredEntry {
                id: completion_id,
                kind: crate::response_store::StoredKind::ChatCompletion,
                model: state.model_name.clone(),
                created_at,
                messages: Vec::new(),
                body,
                last_access: std::time::Instant::now(),
            });
    }

    // Rate-limit true-up. Middleware admitted with a conservative
    // reservation of `max_seq_len` tokens; refund the difference.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    // --dump: emit the non-streaming response body, correlated with
    // the request via the shared seq number. The helper short-
    // circuits when the atlas::dump target is filtered out.
    if let Some(seq) = dump_seq {
        crate::request_dumper::dump_response(
            "/v1/chat/completions",
            seq,
            request_id.as_str(),
            &completion,
            false,
        );
    }

    Json(completion).into_response()
}
