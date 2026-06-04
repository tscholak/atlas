// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;


#[test]
fn parse_qwen3_coder_call_direct_two_funcs_returns_first_only_no_bleed() {
    // Direct test of `parse_qwen3_coder_call`: when fed text that
    // contains two consecutive `<function=...>` blocks, it must
    // return ONLY the first call's parameters — no fields from
    // the second block leak into the first's args. This is the
    // exact contract the BareFunctionAttrPass relies on (it
    // advances past `</function>` between calls).
    let input = "<function=write>\n\
            <parameter=filePath>\n/tmp/y.txt\n</parameter>\n\
            <parameter=content>\nhello\n</parameter>\n\
            </function>\n\
            <function=bash>\n\
            <parameter=command>\nrm -rf /\n</parameter>\n\
            </function>";
    let tc = parse_qwen3_coder_call(input, 0).expect("must parse first function");
    assert_eq!(tc.function.name, "write");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["filePath"], "/tmp/y.txt");
    assert_eq!(args["content"], "hello");
    assert!(
        args.get("command").is_none(),
        "param loop must NOT cross `</function>` boundary: {args:?}"
    );
}

#[test]
fn streaming_detector_qwen3_coder() {
    let mut det = StreamingToolDetector::new();
    let out = det.process(
        "Hi <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    );
    assert!(out.len() >= 2);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Hi")));
    assert!(matches!(&out[1], DetectorOutput::ToolCall(tc, 0) if tc.function.name == "f"));
    assert!(det.has_tool_calls());
}

#[test]
fn tool_call_format_from_str() {
    assert!("hermes".parse::<ToolCallFormat>().is_ok());
    assert!("qwen3_coder".parse::<ToolCallFormat>().is_ok());
    assert!("unknown".parse::<ToolCallFormat>().is_err());
}

#[test]
fn into_parser_returns_correct_name() {
    let h = ToolCallFormat::Hermes.into_parser();
    assert_eq!(h.name(), "hermes");
    let q = ToolCallFormat::Qwen3Coder.into_parser();
    assert_eq!(q.name(), "qwen3_coder");
}

#[test]
fn hermes_parser_system_prompt_contains_json() {
    let parser = HermesParser;
    let tools = vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "test".into(),
            description: None,
            parameters: None,
        },
    }];
    let prompt = parser.system_prompt(&tools, &ToolChoice::Mode("auto".into()));
    assert!(prompt.contains("\"name\":\"test\""));
    assert!(prompt.contains("<tools>"));
}

#[test]
fn qwen3_coder_parser_system_prompt_contains_xml() {
    let parser = Qwen3CoderParser;
    let tools = vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "test".into(),
            description: Some("A test function".into()),
            parameters: None,
        },
    }];
    let prompt = parser.system_prompt(&tools, &ToolChoice::Mode("auto".into()));
    assert!(prompt.contains("\"name\":\"test\""));
    assert!(prompt.contains("A test function"));
    assert!(prompt.contains("<function=example_function_name>"));
}

#[test]
fn format_tool_response_default() {
    let parser = HermesParser;
    let resp = parser.format_tool_response("{\"temp\": 20}");
    assert_eq!(resp, "<tool_response>\n{\"temp\": 20}\n</tool_response>");
}

#[test]
fn streaming_bare_function_flush() {
    let mut det = StreamingToolDetector::new();
    // Feed complete content including bare function tag
    let out1 = det.process(
        "Hello <function>test</function><parameters><name>x</name><value>1</value></parameters>",
    );
    // Flush triggers bare function detection on buffered content
    let out2 = det.flush();
    let all: Vec<_> = out1.into_iter().chain(out2).collect();
    let has_content = all
        .iter()
        .any(|o| matches!(o, DetectorOutput::Content(s) if s.contains("Hello")));
    let has_tool = all
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCall(tc, _) if tc.function.name == "test"));
    assert!(has_content, "Should have content before function tag");
    assert!(has_tool, "Should detect bare function tag on flush");
}

