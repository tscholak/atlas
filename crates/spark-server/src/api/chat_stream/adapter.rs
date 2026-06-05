// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming SSE adapter for the chat-completion FSM.
//!
//! Translates the canonical `FsmEvent` stream emitted by `chat_fsm::Stepper`
//! into OpenAI-compatible `ChatCompletionChunk` SSE events. Also maintains
//! the per-request `ChoiceBuilder` used by the `--dump` observability path
//! at end-of-stream.
//!
//! See `Stage 3` in the plan file
//! (`/Users/tscholak/.claude/plans/i-ve-deployed-the-heim-snazzy-swan.md`)
//! §C4 for the design rationale.

use std::sync::Arc;

use axum::response::sse::Event;

use crate::AppState;
use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser::{FunctionCall, ToolCall};

use super::super::chat_fsm::assemble::{assemble_chat_response, build_usage};
use super::super::chat_fsm::choice_builder::ChoiceBuilder;
use super::super::chat_fsm::events::FsmEvent;

pub(super) type SseVec = Vec<Result<Event, std::convert::Infallible>>;

pub(super) struct StreamingAdapter {
    // ── Wire identity ───────────────────────────────────────────────
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) id: String,
    pub(super) request_id: crate::request_id::RequestId,
    pub(super) dump_seq: Option<u64>,
    pub(super) req_stream_include_usage: bool,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
    pub(super) prompt_len: usize,

    // ── Dump assembly ───────────────────────────────────────────────
    /// Builder that mirrors what the client sees: every FsmEvent we
    /// translate is also applied here so the dump body matches the
    /// blocking response shape exactly (§C6).
    pub(super) choice: ChoiceBuilder,

    // ── Usage scratch ───────────────────────────────────────────────
    /// Filled by `set_usage_inputs` when `StreamEvent::Done` arrives,
    /// consumed when `Stopped` is translated.
    pub(super) usage_inputs: Option<UsageInputs>,

    // ── return_token_ids (vLLM-compatible extension) ────────────────
    /// When true, every sampled token id passed to `buffer_token_id`
    /// is held until the next client-visible chunk drains it onto
    /// `choices[0].token_ids`. Default false — opt-in only.
    pub(super) return_token_ids: bool,
    /// Tokens sampled since the last drain. Always empty when
    /// `return_token_ids = false` (buffer_token_id is a no-op).
    /// Invariant: Σ drained == usage.completion_tokens exactly.
    pub(super) pending_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(super) struct UsageInputs {
    pub completion_tokens: usize,
    pub time_to_first_token_ms: f64,
    pub decode_time_ms: f64,
    pub reasoning_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

impl StreamingAdapter {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        state: Arc<AppState>,
        model: String,
        id: String,
        request_id: crate::request_id::RequestId,
        dump_seq: Option<u64>,
        req_stream_include_usage: bool,
        req_ctx: Option<crate::rate_limiter::RequestContext>,
        prompt_len: usize,
        return_token_ids: bool,
    ) -> Self {
        Self {
            state,
            model,
            id,
            request_id,
            dump_seq,
            req_stream_include_usage,
            req_ctx,
            prompt_len,
            choice: ChoiceBuilder::new(0),
            usage_inputs: None,
            return_token_ids,
            pending_token_ids: Vec::new(),
        }
    }

    /// Record one sampled token id. No-op when `return_token_ids=false`.
    /// Called from the streaming entry point BEFORE `stepper.step_token`,
    /// so the id buffered here is exactly the one whose effects (deltas,
    /// tool-call events, stop) translate into the next emitted chunk.
    pub(super) fn buffer_token_id(&mut self, tok: u32) {
        if self.return_token_ids {
            self.pending_token_ids.push(tok);
        }
    }

    /// Drain the pending buffer. Returns `Vec::new()` when the request
    /// did not opt in (buffer is always empty in that case).
    fn take_pending_ids(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.pending_token_ids)
    }

    /// Cache the per-`StreamEvent::Done` usage data so the terminal
    /// `Stopped` translation can build the usage chunk(s).
    pub(super) fn set_usage_inputs(&mut self, u: UsageInputs) {
        self.usage_inputs = Some(u);
    }

    /// Translate one FsmEvent into zero or more SSE chunks, and apply
    /// it to the dump-builder. `Stopped` is the terminal event and
    /// drives the usage chunk(s), the dump emit, and the rate-limit
    /// true-up.
    pub(super) fn translate(&mut self, ev: FsmEvent, out: &mut SseVec) {
        // Emit the wire chunk(s) first. Each client-visible chunk
        // drains the pending sampled-token-id buffer onto its
        // `choices[0].token_ids` (no-op when return_token_ids=false).
        match &ev {
            FsmEvent::ReasoningDelta(text) => {
                let chunk = ChatCompletionChunk::reasoning_chunk(
                    &self.model,
                    &self.id,
                    text.clone(),
                )
                .with_token_ids(self.take_pending_ids());
                push(out, &chunk);
            }
            FsmEvent::ContentDelta(text) => {
                let chunk = ChatCompletionChunk::content_chunk(
                    &self.model,
                    &self.id,
                    text.clone(),
                )
                .with_token_ids(self.take_pending_ids());
                push(out, &chunk);
            }
            FsmEvent::ToolCallStart { id, name, idx } => {
                let tc = ToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: String::new(),
                    },
                };
                let chunk =
                    ChatCompletionChunk::tool_call_start_chunk(&self.model, &self.id, &tc, *idx)
                        .with_token_ids(self.take_pending_ids());
                push(out, &chunk);
            }
            FsmEvent::ToolCallArgDelta { args, idx } => {
                let chunk = ChatCompletionChunk::tool_call_args_fragment(
                    &self.model,
                    &self.id,
                    *idx,
                    args,
                )
                .with_token_ids(self.take_pending_ids());
                push(out, &chunk);
            }
            FsmEvent::ToolCallEnd { .. } | FsmEvent::Stopped { .. } => {
                // No direct wire chunk. ToolCallEnd → log+metric below
                // (after apply, so the builder has the assembled
                // call). Stopped → terminal cascade below (drains
                // residual token_ids onto the final chunk).
            }
        }
        // Mirror onto the dump builder.
        let is_stopped = matches!(ev, FsmEvent::Stopped { .. });
        let is_tool_call_end = matches!(ev, FsmEvent::ToolCallEnd { .. });
        self.choice.apply(ev);
        // Observability log + metric for completed tool calls. Mirrors
        // the historical sites at tool_handlers.rs:32-33,142-143.
        if is_tool_call_end {
            if let Some(tc) = self.choice.tool_calls().last() {
                log_tool_call(&tc.function.name, &tc.function.arguments);
            }
        }
        // Handle the terminal cascade after apply so finish_reason
        // computation sees the up-to-date builder state.
        if is_stopped {
            self.emit_terminal_and_dump(out);
        }
    }

    /// Build the usage chunk(s) at end-of-stream + emit the dump.
    fn emit_terminal_and_dump(&mut self, out: &mut SseVec) {
        let usage_inputs = self.usage_inputs.clone();
        let completion_tokens = usage_inputs
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(0);
        let usage = build_usage_from_adapter(self, usage_inputs.as_ref());

        // Finish reason is computed by ChoiceBuilder::into_chat_choice;
        // we need the same value here for the wire chunks BEFORE we
        // consume the builder for the dump. Compute it from the
        // builder snapshot.
        let has_tool_calls = !self.choice.tool_calls().is_empty();
        let fr = super::super::chat_fsm::assemble::compute_finish_reason(
            self.choice.stop_reason(),
            has_tool_calls,
        );

        // Terminal SSE chunk(s): either usage_only + final-no-usage,
        // or done-chunk with embedded usage. Residual sampled-token-ids
        // (tokens emitted since the last client-visible delta — e.g.
        // </think>, the EOS marker, a stop token) attach to the chunk
        // that carries `choices` so the per-request Σ token_ids equals
        // usage.completion_tokens exactly. `usage_only_chunk` has
        // `choices: []`, so the residual goes on `final_chunk_no_usage`.
        let residual_ids = self.take_pending_ids();
        if self.req_stream_include_usage {
            let usage_chunk =
                ChatCompletionChunk::usage_only_chunk(&self.model, &self.id, usage.clone());
            push(out, &usage_chunk);
            let final_chunk =
                ChatCompletionChunk::final_chunk_no_usage(&self.model, &self.id, &fr)
                    .with_token_ids(residual_ids);
            push(out, &final_chunk);
        } else {
            let done_chunk =
                ChatCompletionChunk::done_chunk(&self.model, &self.id, &fr, usage.clone())
                    .with_token_ids(residual_ids);
            push(out, &done_chunk);
        }

        // Metrics.
        crate::metrics::REQUESTS_ACTIVE.dec();
        crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(self.prompt_len as u64);
        crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
        if let Some(u) = usage_inputs.as_ref() {
            crate::metrics::TTFT_SECONDS.observe(u.time_to_first_token_ms / 1000.0);
        }

        // Rate-limit true-up.
        if let Some(ref rctx) = self.req_ctx {
            let actual = (self.prompt_len + completion_tokens) as u64;
            let refund = rctx.reserved_tokens.saturating_sub(actual);
            if refund > 0 {
                self.state.rate_limiter.refund_tokens(&rctx.identity, refund);
            }
        }

        // Dump emit. Consumes the choice builder via std::mem::take.
        if let Some(seq) = self.dump_seq
            && tracing::event_enabled!(target: "atlas::dump", tracing::Level::INFO)
        {
            let choice = std::mem::replace(&mut self.choice, ChoiceBuilder::new(0))
                .into_chat_choice(None);
            let response = assemble_chat_response(
                self.id.clone(),
                self.model.clone(),
                crate::openai::unix_timestamp(),
                vec![choice],
                usage,
                None,
                None,
            );
            crate::request_dumper::dump_response(
                "/v1/chat/completions",
                seq,
                self.request_id.as_str(),
                &response,
                true,
            );
        }
    }

}

fn push(out: &mut SseVec, chunk: &ChatCompletionChunk) {
    let json = serde_json::to_string(chunk).unwrap_or_default();
    out.push(Ok(Event::default().data(json)));
}

fn log_tool_call(name: &str, args_json: &str) {
    let preview: String = args_json.chars().take(120).collect();
    let s = if args_json.len() > preview.len() {
        "…"
    } else {
        ""
    };
    tracing::info!("Tool call: {name}({preview}{s})");
    crate::metrics::TOOL_CALLS_TOTAL.inc();
}

fn build_usage_from_adapter(
    adapter: &StreamingAdapter,
    inputs: Option<&UsageInputs>,
) -> Usage {
    if let Some(u) = inputs {
        build_usage(
            adapter.prompt_len,
            u.completion_tokens,
            u.time_to_first_token_ms,
            u.decode_time_ms,
            u.reasoning_tokens,
            u.cached_prompt_tokens,
            u.accepted_prediction_tokens,
            u.rejected_prediction_tokens,
            adapter.request_id.as_str().to_string(),
        )
    } else {
        // Degenerate path: should not happen in practice because the
        // Done arm always calls set_usage_inputs before flush. Build
        // a zero-filled usage block so the SSE chunk shape stays
        // valid and the request still terminates cleanly.
        build_usage(
            adapter.prompt_len,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            adapter.request_id.as_str().to_string(),
        )
    }
}
