// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar-shape inspection tool. Constructs the structural_tag JSON that
// `compile_qwen3_coder_tool_grammar` produces for the deployed heim path
// (Qwen3.6 + qwen3_coder + bash tool + enable_thinking={true,false}),
// then asks xgrammar to parse + render the EBNF via `to_string_ebnf`.
//
// Purpose: experimental verification that the high-level structural_tag
// DSL atlas uses actually compiles down to the response-shape grammar we
// intend it to. Run on Spark via the documented atlas-dev workflow:
//
//   cd ~/atlas-dev
//   git checkout heim/main
//   devenv shell
//   cargo run --release --example dump_grammar -p spark-server
//
// The output prints the input JSON and the rendered EBNF for both
// enable_thinking=true and enable_thinking=false, plus a side-by-side
// comparison against the formal grammar in /tmp/grammar-experiment.md.

use serde_json::json;
use xgrammar::Grammar;

/// JSON schema for the heim bash tool. Mirrors
/// `harness/heim_agent/agent.py::BASH_TOOL["parameters"]` with
/// `enforce_min_length_on_required_strings` already applied (the deployed
/// path adds `minLength: 1` to `command`).
fn bash_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
                "minLength": 1
            },
            "background": {
                "type": "boolean",
                "description": "Run in background. Returns immediately.",
                "default": false
            }
        },
        "required": ["command"]
    })
}

/// Build the qwen3_coder tag entry for a tool. Mirrors
/// `compile_qwen3_coder_tool_grammar`'s tag construction (compile_tools.rs:211-219).
fn bash_tag_entry() -> serde_json::Value {
    json!({
        "type": "tag",
        "begin": "<tool_call>\n<function=bash>\n",
        "content": {"type": "qwen_xml_parameter", "json_schema": bash_schema()},
        "end": "\n</function>\n</tool_call>"
    })
}

/// The qwen3_coder LATE trigger for `tool_choice="auto"` (per
/// compile_tools.rs:249-253).
fn bash_trigger() -> &'static str {
    "<tool_call>\n<function=bash"
}

/// Build the structural_tag JSON for `enable_thinking=true`. Mirrors
/// `compile_thinking_wrapped_structural_tag` (compile_misc.rs:57-99,
/// after A3 added `excludes` to the post-think triggered_tags).
fn thinking_wrapped_json() -> String {
    json!({
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "any_text",
                    "excludes": [
                        "<think>",
                        "<function=",
                        "<tool_call>",
                        "<parameter=",
                        "</function>",
                        "</tool_call>",
                        "</parameter>"
                    ]
                },
                {"type": "const_string", "value": "</think>"},
                {
                    "type": "triggered_tags",
                    "triggers": [bash_trigger()],
                    "tags": [bash_tag_entry()],
                    "at_least_one": false,
                    "stop_after_first": false,
                    "excludes": ["<think>", "</think>"]
                }
            ]
        }
    })
    .to_string()
}

/// Build the structural_tag JSON for `enable_thinking=false`. Mirrors
/// `compile_structural_tag_raw` (compile_misc.rs:26-49, after A3).
fn non_thinking_json() -> String {
    json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": [bash_trigger()],
            "tags": [bash_tag_entry()],
            "at_least_one": false,
            "stop_after_first": false,
            "excludes": ["<think>", "</think>"]
        }
    })
    .to_string()
}

fn dump_section(label: &str, st_json: &str) {
    println!("\n{}", "=".repeat(72));
    println!("== {label}");
    println!("{}", "=".repeat(72));
    println!("\n--- structural_tag JSON input ---\n");
    // Pretty-print the input for readability.
    match serde_json::from_str::<serde_json::Value>(st_json) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap_or_else(|_| st_json.to_string())),
        Err(_) => println!("{st_json}"),
    }
    println!("\n--- compiled EBNF (Grammar::to_string_ebnf) ---\n");
    match Grammar::from_structural_tag(st_json) {
        Ok(g) => println!("{}", g.to_string_ebnf()),
        Err(e) => eprintln!("xgrammar rejected structural_tag: {e}"),
    }
}

fn main() {
    dump_section("enable_thinking=true (compile_thinking_wrapped_structural_tag)", &thinking_wrapped_json());
    dump_section("enable_thinking=false (compile_structural_tag_raw)", &non_thinking_json());

    println!("\n{}", "=".repeat(72));
    println!("== End of dump. Compare against /tmp/grammar-experiment.md");
    println!("{}", "=".repeat(72));
}
