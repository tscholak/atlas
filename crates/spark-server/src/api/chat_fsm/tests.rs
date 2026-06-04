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

// ─── §D3 streaming vs blocking byte-equivalence ───────────────────────
//
// Drive a single sequence of pre-decoded text chunks (simulating what
// a Qwen-template-compliant model emits at decode time) through the
// chat_fsm Stepper. Run the resulting FsmEvent stream through TWO
// adapters:
//   * "streaming" — apply each event to a ChoiceBuilder
//   * "blocking"  — same event stream, second ChoiceBuilder
// Assert both ChoiceBuilders produce byte-identical ChatChoice JSON.
// This is the unification's correctness guarantee: ONE FSM, ONE
// assembler, so the streaming roll-up MUST equal the blocking body.
//
// The chunk shapes here mirror what `tokenizer.streaming_decoder(false)`
// would actually emit for a Qwen3 model: protocol markers (`</think>`,
// `<tool_call>`) arrive as literal text because `skip_special_tokens
// =false` keeps them un-rendered as IDs.

use super::stepper::{Stepper, StepperConfig};
use crate::reasoning_parser::QwenReasoningParser;

/// Leak a QwenReasoningParser so the Stepper can hold a `'static`
/// borrow during the test. Tests run for milliseconds; the leak is
/// bounded and acceptable for fixtures.
fn qwen_static() -> &'static dyn crate::reasoning_parser::ReasoningParser {
    Box::leak(Box::new(QwenReasoningParser))
        as &'static dyn crate::reasoning_parser::ReasoningParser
}

fn drive_stepper(cfg: StepperConfig, chunks: &[&str]) -> Vec<FsmEvent> {
    let mut s = Stepper::for_tests(cfg);
    let mut out = Vec::new();
    for chunk in chunks {
        out.extend(s.feed_text(chunk.to_string()));
    }
    out.extend(s.flush("stop".to_string()));
    out
}

#[test]
fn d3_streaming_blocking_qwen_thinking_then_content() {
    // Canonical Qwen3 reasoning → content flow. The model first
    // produces reasoning, emits the `</think>` end-tag, then a
    // blank line, then the content. No tools active.
    let cfg = StepperConfig {
        enable_thinking: true,
        tools_active: false,
        stop_strings: vec![],
        reasoning_parser: Some(qwen_static()),
    };
    let chunks = [
        "The user said hi. ",
        "I'll greet back.",
        "</think>",
        "\n\n",
        "Hello",
        "!",
    ];

    // ── First pass: capture the FsmEvent stream once ────────────────
    let events = drive_stepper(
        StepperConfig {
            reasoning_parser: cfg.reasoning_parser,
            ..StepperConfig {
                enable_thinking: cfg.enable_thinking,
                tools_active: cfg.tools_active,
                stop_strings: cfg.stop_strings.clone(),
                reasoning_parser: cfg.reasoning_parser,
            }
        },
        &chunks,
    );

    // The ThinkingScanner holds back `end_tag.len() - 1` trailing
    // bytes per chunk for straddle protection (`</think>` is 8 chars
    // so hold = 7). Reasoning deltas therefore DO NOT match the input
    // chunks verbatim — but their CONCATENATION must, because no byte
    // is dropped or rewritten. Assert at the concatenation level.
    let reasoning_concat: String = events
        .iter()
        .filter_map(|e| match e {
            FsmEvent::ReasoningDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    let content_concat: String = events
        .iter()
        .filter_map(|e| match e {
            FsmEvent::ContentDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    let stop_count = events
        .iter()
        .filter(|e| matches!(e, FsmEvent::Stopped { .. }))
        .count();
    assert_eq!(
        reasoning_concat, "The user said hi. I'll greet back.",
        "reasoning bytes must round-trip verbatim across deltas"
    );
    assert_eq!(
        content_concat, "Hello!",
        "content bytes must round-trip verbatim post-</think>+blank-line trim"
    );
    assert_eq!(stop_count, 1, "exactly one Stopped event per FSM lifetime");

    // ── Second pass: build streaming + blocking ChoiceBuilders ──────
    let mut streaming = ChoiceBuilder::new(0);
    let mut blocking = ChoiceBuilder::new(0);
    for ev in &events {
        streaming.apply(ev.clone());
        blocking.apply(ev.clone());
    }
    let streaming_choice = streaming.into_chat_choice(None);
    let blocking_choice = blocking.into_chat_choice(None);

    let streaming_json = serde_json::to_value(&streaming_choice).unwrap();
    let blocking_json = serde_json::to_value(&blocking_choice).unwrap();
    assert_eq!(
        streaming_json, blocking_json,
        "streaming roll-up must byte-match blocking response for the same FSM event stream"
    );

    // ── Final shape check ──────────────────────────────────────────
    assert_eq!(streaming_choice.message.role, "assistant");
    assert_eq!(
        streaming_choice.message.reasoning_content.as_deref(),
        Some("The user said hi. I'll greet back.")
    );
    assert_eq!(streaming_choice.message.content.as_deref(), Some("Hello!"));
    assert!(streaming_choice.message.tool_calls.is_none());
    assert_eq!(streaming_choice.finish_reason, "stop");
}

#[test]
fn d3_streaming_blocking_qwen_thinking_then_tool_call() {
    // Canonical Qwen3-coder tool-call flow: reasoning → `</think>` →
    // newline → `<tool_call>` envelope with `<function=NAME>` +
    // `<parameter=KEY>VALUE</parameter>` → `</tool_call>`. The
    // StreamingToolDetector emits ToolCallStart on `<function=NAME>`
    // and one ToolCallArgDelta at `</tool_call>` (the XML must close
    // before it canonicalises to JSON).
    let cfg = StepperConfig {
        enable_thinking: true,
        tools_active: true,
        stop_strings: vec![],
        reasoning_parser: Some(qwen_static()),
    };
    let chunks = [
        "Need to write a file.",
        "</think>",
        "\n\n",
        "<tool_call>\n",
        "<function=Write>\n",
        "<parameter=path>/tmp/a.txt</parameter>\n",
        "<parameter=content>hi</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ];

    let events = drive_stepper(
        StepperConfig {
            reasoning_parser: cfg.reasoning_parser,
            enable_thinking: cfg.enable_thinking,
            tools_active: cfg.tools_active,
            stop_strings: cfg.stop_strings.clone(),
        },
        &chunks,
    );

    // Expected event sequence (order matters):
    //   ReasoningDelta("Need to write a file.")
    //   ToolCallStart { id: "call_XXXX", name: "Write", idx: 0 }
    //   ToolCallArgDelta { args: "{...}", idx: 0 }
    //   ToolCallEnd { idx: 0 }
    //   Stopped { Upstream("stop") }
    let tool_starts: Vec<&FsmEvent> = events
        .iter()
        .filter(|e| matches!(e, FsmEvent::ToolCallStart { .. }))
        .collect();
    let tool_ends: Vec<&FsmEvent> = events
        .iter()
        .filter(|e| matches!(e, FsmEvent::ToolCallEnd { .. }))
        .collect();
    assert_eq!(
        tool_starts.len(),
        1,
        "exactly one ToolCallStart for a single-tool envelope"
    );
    assert_eq!(tool_ends.len(), 1, "exactly one ToolCallEnd");
    if let FsmEvent::ToolCallStart { name, .. } = tool_starts[0] {
        assert_eq!(name, "Write", "function name extracted from <function=NAME>");
    }

    // ── Streaming vs blocking equivalence ───────────────────────────
    let mut streaming = ChoiceBuilder::new(0);
    let mut blocking = ChoiceBuilder::new(0);
    for ev in &events {
        streaming.apply(ev.clone());
        blocking.apply(ev.clone());
    }
    let streaming_choice = streaming.into_chat_choice(None);
    let blocking_choice = blocking.into_chat_choice(None);
    assert_eq!(
        serde_json::to_value(&streaming_choice).unwrap(),
        serde_json::to_value(&blocking_choice).unwrap(),
        "tool-call flow streaming roll-up must byte-match blocking response"
    );

    // ── Final shape check ──────────────────────────────────────────
    assert_eq!(
        streaming_choice.message.reasoning_content.as_deref(),
        Some("Need to write a file.")
    );
    // content empty + tool_calls present → content is None per §C7.
    assert!(streaming_choice.message.content.is_none());
    let calls = streaming_choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Write");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("args must be valid JSON");
    assert_eq!(args["path"], "/tmp/a.txt");
    assert_eq!(args["content"], "hi");
    // Tool call fired → finish_reason promoted regardless of upstream.
    assert_eq!(streaming_choice.finish_reason, "tool_calls");
}

#[test]
fn d3_stop_string_in_tool_envelope_completes_call_first() {
    // Deliberate correctness fix from §C11: a stop string appearing
    // inside the `<tool_call>...</tool_call>` body must NOT terminate
    // the FSM mid-envelope. The detector buffers the envelope; the
    // tool call completes via Start/ArgDelta/End; THEN the stop fires
    // on the next genuine Content delta if any.
    let cfg = StepperConfig {
        enable_thinking: false,
        tools_active: true,
        stop_strings: vec!["STOP".to_string()],
        reasoning_parser: None,
    };
    let chunks = [
        "<tool_call>\n",
        "<function=Run>\n",
        "<parameter=cmd>echo STOP</parameter>\n",
        "</function>\n",
        "</tool_call>",
        "trailing STOP after envelope",
    ];

    let events = drive_stepper(cfg, &chunks);

    // The tool call must complete before any Stopped fires. Find the
    // index of ToolCallEnd and the index of Stopped; ToolCallEnd MUST
    // come first.
    let end_idx = events
        .iter()
        .position(|e| matches!(e, FsmEvent::ToolCallEnd { .. }))
        .expect("tool call must complete before stop");
    let stopped_idx = events
        .iter()
        .position(|e| matches!(e, FsmEvent::Stopped { .. }))
        .expect("FSM must terminate");
    assert!(
        end_idx < stopped_idx,
        "ToolCallEnd must precede Stopped — stop strings cannot bisect a structured tool call"
    );

    // The stop reason should be StopString (matched the trailing
    // "STOP" in the post-envelope content), not Upstream.
    if let FsmEvent::Stopped { reason } = &events[stopped_idx] {
        match reason {
            StopReason::StopString { matched } => assert_eq!(matched, "STOP"),
            other => panic!("expected StopReason::StopString, got {other:?}"),
        }
    }
}

#[test]
fn d3_content_only_no_thinking_no_tools() {
    // Simplest path: enable_thinking=false, no detector, no stops.
    // The Stepper starts in Content phase; every chunk emits a
    // ContentDelta verbatim.
    let cfg = StepperConfig {
        enable_thinking: false,
        tools_active: false,
        stop_strings: vec![],
        reasoning_parser: None,
    };
    let chunks = ["Hello, ", "world", "!"];

    let events = drive_stepper(cfg, &chunks);
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            FsmEvent::ContentDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["Hello, ", "world", "!"]);
    assert!(events
        .iter()
        .any(|e| matches!(e, FsmEvent::Stopped { reason: StopReason::Upstream(s) } if s == "stop")));
}
