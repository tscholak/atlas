// SPDX-License-Identifier: AGPL-3.0-only

//! Canonical event stream emitted by the chat-completion FSM. Both the
//! streaming SSE adapter and the blocking response adapter translate
//! this stream into their output formats.

/// Reason the FSM transitioned to its terminal state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Upstream `StreamEvent::Done.finish_reason` (the scheduler's
    /// verbatim string: "stop", "length", "timeout", etc.). The
    /// assembler maps this to the OpenAI-visible finish_reason via
    /// `assemble::compute_finish_reason`.
    Upstream(String),
    /// A configured stop string was matched in the content channel.
    /// `matched` is the exact stop string that fired (for dump +
    /// metrics; OpenAI's public finish_reason for this is "stop").
    StopString { matched: String },
    /// xgrammar `stop_after_first` reached an accepting state for a
    /// tool-call envelope under `tool_choice: required`. Carried
    /// separately because the scheduler reports the EOS path's
    /// `"stop"`, not `"tool_calls"`; the assembler promotes.
    GrammarTerminated,
}

/// One event emitted by the chat-completion FSM.
///
/// Invariants:
/// * `ReasoningDelta(text)` and `ContentDelta(text)` are never empty.
/// * `ToolCallArgDelta { args, .. }` is never empty.
/// * `ToolCallStart { idx, .. }` always precedes any `ToolCallArgDelta`
///   and `ToolCallEnd` with the same `idx`.
/// * `Stopped { .. }` is emitted exactly once per FSM lifetime, as the
///   final event of either `step_token` or `flush`. No further events
///   are emitted after `Stopped`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsmEvent {
    ReasoningDelta(String),
    ContentDelta(String),
    ToolCallStart { id: String, name: String, idx: usize },
    ToolCallArgDelta { args: String, idx: usize },
    ToolCallEnd { idx: usize },
    Stopped { reason: StopReason },
}
