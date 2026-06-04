// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the four `DetectorOutput` variants emitted by the
// streaming tool-call detector. Shared by both `handle_token` (mid-
// stream `process()` outputs) and `handle_done` (end-of-stream
// `flush()` outputs).

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::ctx::StreamCtx;
use super::state::StreamState;

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// `DetectorOutput::ToolCall(tc, idx)`: complete tool call.
pub(super) fn handle_complete_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc: &mut tool_parser::ToolCall,
    tc_idx: usize,
    sse_events: &mut SseVec,
) {
    let preview: String = tc.function.arguments.chars().take(120).collect();
    let s = if tc.function.arguments.len() > preview.len() {
        "…"
    } else {
        ""
    };
    tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
    crate::metrics::TOOL_CALLS_TOTAL.inc();
    state.record_tool_call(tc.clone());
    let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, tc_idx);
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&start).unwrap_or_default())
    ));
    let frag = ChatCompletionChunk::tool_call_args_fragment(
        &ctx.model,
        &ctx.id,
        tc_idx,
        &tc.function.arguments,
    );
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
    ));
}

/// `DetectorOutput::ToolCallStart` — incremental: emit header now.
pub(super) fn handle_tool_call_start(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc_id: String,
    name: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    state.streaming_tool_args.insert(
        idx,
        super::state::StreamingToolCall {
            id: tc_id.clone(),
            name: name.clone(),
            args: String::new(),
        },
    );
    let tc = tool_parser::ToolCall {
        id: tc_id,
        call_type: "function".to_string(),
        function: tool_parser::FunctionCall {
            name,
            arguments: String::new(),
        },
    };
    let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &tc, idx);
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&start).unwrap_or_default())
    ));
}

/// `DetectorOutput::ToolCallDelta` — incremental: append args.
///
/// For qwen3_coder XML the streaming detector emits a single Delta with
/// the full parsed-and-canonicalised JSON arguments at the `</tool_call>`
/// boundary (see `streaming_impl.rs::process` line ~67 — args can't be
/// streamed character-by-character because XML parameter blocks must
/// finish before they convert to JSON). This is the natural spot to run
/// the same `backfill_required_params` + `validate_single_tool_call`
/// chain that the complete-tool-call path runs at `handle_complete_tool_call`,
/// so that streaming and non-streaming responses behave identically.
///
/// Without this, a model that emits `<function=NAME></function>` with no
/// `<parameter=>` blocks (observed under qwen3_coder + multi-turn agentic
/// loops with 21 tools, OpenClaw 2026.5.7) streams literal `"{}"` to the
/// client even when required parameters are declared in the schema —
/// while the non-streaming path would have backfilled `{"required_key": ""}`
/// and at least logged a warning. Issue #40 (iromu) called out this
/// "Opencode breaks tool calling more often" symptom.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        entry.args.push_str(&args);
    }
    if !args.is_empty() {
        let frag =
            ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &args);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    }
}

/// `DetectorOutput::ToolCallEnd` — assemble the streaming tool call
/// into a record for the observability dump and emit the per-call
/// log + metric to match the blocking and complete-call paths.
pub(super) fn handle_tool_call_end(state: &mut StreamState, idx: usize) {
    if let Some(super::state::StreamingToolCall {
        id,
        name,
        args: args_json,
    }) = state.streaming_tool_args.remove(&idx)
    {
        state.record_tool_call(tool_parser::ToolCall {
            id,
            call_type: "function".to_string(),
            function: tool_parser::FunctionCall {
                name: name.clone(),
                arguments: args_json.clone(),
            },
        });
        let preview: String = args_json.chars().take(120).collect();
        let s = if args_json.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {name}({preview}{s})");
        crate::metrics::TOOL_CALLS_TOTAL.inc();
    }
}
