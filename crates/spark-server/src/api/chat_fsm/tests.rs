// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the chat_fsm module. These test the building blocks
//! (`StopPredicate`, `ChoiceBuilder`, `compute_finish_reason`) and
//! event-shape invariants. Full Stepper tests that drive over real
//! token sequences require a tokenizer and live in integration tests
//! that the streaming/blocking adapters carry once they're wired in
//! commits 2 and 3.

use super::choice_builder::ChoiceBuilder;
use super::events::{FsmEvent, StopReason};
use crate::tool_parser::{FunctionCall, ToolCall};

fn tc(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: args.to_string(),
        },
    }
}

#[test]
fn choice_builder_accumulates_content_and_reasoning() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ReasoningDelta("First thought. ".to_string()));
    b.apply(FsmEvent::ReasoningDelta("Second thought.".to_string()));
    b.apply(FsmEvent::ContentDelta("Hello, ".to_string()));
    b.apply(FsmEvent::ContentDelta("world!".to_string()));
    b.apply(FsmEvent::Stopped {
        reason: StopReason::Upstream("stop".to_string()),
    });

    assert_eq!(b.reasoning_str(), "First thought. Second thought.");
    assert_eq!(b.content_str(), "Hello, world!");
    assert!(b.tool_calls().is_empty());

    let choice = b.into_chat_choice(None);
    assert_eq!(choice.index, 0);
    assert_eq!(choice.finish_reason, "stop");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("First thought. Second thought.")
    );
    assert_eq!(choice.message.reasoning.as_deref(), Some("First thought. Second thought."));
    assert_eq!(choice.message.content.as_deref(), Some("Hello, world!"));
    assert!(choice.message.tool_calls.is_none());
}

#[test]
fn choice_builder_assembles_tool_call_from_start_delta_end() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ToolCallStart {
        id: "call_xyz".to_string(),
        name: "Write".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: "{\"path\":\"/tmp/a.txt\"".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: ",\"content\":\"hi\"}".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallEnd { idx: 0 });
    b.apply(FsmEvent::Stopped {
        reason: StopReason::Upstream("stop".to_string()),
    });

    assert_eq!(b.tool_calls().len(), 1);
    assert_eq!(b.tool_calls()[0].id, "call_xyz");
    assert_eq!(b.tool_calls()[0].function.name, "Write");
    assert_eq!(
        b.tool_calls()[0].function.arguments,
        "{\"path\":\"/tmp/a.txt\",\"content\":\"hi\"}"
    );

    let choice = b.into_chat_choice(None);
    // tool_calls present → finish_reason promoted regardless of upstream.
    assert_eq!(choice.finish_reason, "tool_calls");
    // content empty + tool_calls present → content is None per OpenAI spec.
    assert!(choice.message.content.is_none());
    assert_eq!(choice.message.tool_calls.as_ref().unwrap().len(), 1);
}

#[test]
fn choice_builder_keeps_content_when_present_alongside_tool_calls() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ContentDelta("Calling Write: ".to_string()));
    b.apply(FsmEvent::ToolCallStart {
        id: "call_a".to_string(),
        name: "Write".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: "{}".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallEnd { idx: 0 });
    b.apply(FsmEvent::Stopped {
        reason: StopReason::Upstream("stop".to_string()),
    });

    let choice = b.into_chat_choice(None);
    assert_eq!(choice.finish_reason, "tool_calls");
    assert_eq!(choice.message.content.as_deref(), Some("Calling Write: "));
    assert!(choice.message.tool_calls.is_some());
}

#[test]
fn choice_builder_stop_string_reports_stop_finish_reason() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ContentDelta("hello".to_string()));
    b.apply(FsmEvent::Stopped {
        reason: StopReason::StopString {
            matched: "</end>".to_string(),
        },
    });
    let choice = b.into_chat_choice(None);
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn choice_builder_grammar_terminated_promotes_to_tool_calls() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ToolCallStart {
        id: "call_g".to_string(),
        name: "Bash".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: "{\"command\":\"ls\"}".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallEnd { idx: 0 });
    b.apply(FsmEvent::Stopped {
        reason: StopReason::GrammarTerminated,
    });

    let choice = b.into_chat_choice(None);
    assert_eq!(choice.finish_reason, "tool_calls");
}

#[test]
fn choice_builder_assembles_multiple_tool_calls_in_emit_order() {
    let mut b = ChoiceBuilder::new(0);
    b.apply(FsmEvent::ToolCallStart {
        id: "call_0".to_string(),
        name: "a".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: "{\"x\":1}".to_string(),
        idx: 0,
    });
    b.apply(FsmEvent::ToolCallEnd { idx: 0 });
    b.apply(FsmEvent::ToolCallStart {
        id: "call_1".to_string(),
        name: "b".to_string(),
        idx: 1,
    });
    b.apply(FsmEvent::ToolCallArgDelta {
        args: "{\"y\":2}".to_string(),
        idx: 1,
    });
    b.apply(FsmEvent::ToolCallEnd { idx: 1 });
    b.apply(FsmEvent::Stopped {
        reason: StopReason::Upstream("stop".to_string()),
    });

    let calls = b.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "a");
    assert_eq!(calls[1].function.name, "b");
}

#[test]
fn finish_reason_precedence() {
    use super::assemble::compute_finish_reason;
    // Tool calls present → always tool_calls.
    assert_eq!(
        compute_finish_reason(
            Some(&StopReason::Upstream("length".to_string())),
            true
        ),
        "tool_calls"
    );
    assert_eq!(
        compute_finish_reason(
            Some(&StopReason::StopString {
                matched: "x".to_string()
            }),
            true
        ),
        "tool_calls"
    );
    // Grammar terminated → tool_calls (regardless of upstream).
    assert_eq!(
        compute_finish_reason(Some(&StopReason::GrammarTerminated), false),
        "tool_calls"
    );
    // StopString → stop.
    assert_eq!(
        compute_finish_reason(
            Some(&StopReason::StopString {
                matched: "x".to_string()
            }),
            false
        ),
        "stop"
    );
    // Upstream passes through verbatim.
    assert_eq!(
        compute_finish_reason(Some(&StopReason::Upstream("length".to_string())), false),
        "length"
    );
    assert_eq!(
        compute_finish_reason(Some(&StopReason::Upstream("timeout".to_string())), false),
        "timeout"
    );
    // None → "stop" (degenerate; flush() always emits Stopped).
    assert_eq!(compute_finish_reason(None, false), "stop");
}

#[test]
fn fsm_event_equality_round_trip() {
    let a = FsmEvent::ToolCallStart {
        id: "call_a".to_string(),
        name: "Write".to_string(),
        idx: 0,
    };
    let b = FsmEvent::ToolCallStart {
        id: "call_a".to_string(),
        name: "Write".to_string(),
        idx: 0,
    };
    assert_eq!(a, b);

    let c = FsmEvent::Stopped {
        reason: StopReason::StopString {
            matched: "END".to_string(),
        },
    };
    let d = FsmEvent::Stopped {
        reason: StopReason::StopString {
            matched: "END".to_string(),
        },
    };
    assert_eq!(c, d);
}

#[test]
fn tool_call_helper_does_not_warn() {
    // Anchor a use of the helper so unused-import warnings don't fire
    // for the fixture; the helper is here for future Stepper tests.
    let t = tc("call_x", "X", "{}");
    assert_eq!(t.id, "call_x");
    assert_eq!(t.function.name, "X");
}
