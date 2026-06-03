// SPDX-License-Identifier: AGPL-3.0-only

//! Misc grammar compilation: structural-tag, JSON schema, JSON, EBNF.

use xgrammar::CompiledGrammar;

use super::engine::{GrammarEngine, GrammarError};

impl GrammarEngine {
    /// Build and compile a structural tag grammar from raw JSON components.
    /// This bypasses xgrammar-rs's `compile_structural_tag` wrapper to access
    /// `at_least_one` and `stop_after_first` parameters.
    ///
    /// The free-text portion of the resulting `triggered_tags` excludes
    /// `<think>` and `</think>` substrings. These are structural invariants
    /// of the response shape:
    ///   * Under `enable_thinking=false` (the deployed chat template renders
    ///     an empty `<think>\n\n</think>\n\n` block in the prompt and the
    ///     response shape is content-only), the model must not re-emit a
    ///     thinking block in its generated output.
    ///   * Under `enable_thinking=true`, the post-think content phase
    ///     (this `triggered_tags`, after the const-string `</think>`
    ///     transition in `compile_thinking_wrapped_structural_tag`) must
    ///     not contain a fresh `<think>` re-open or a stray duplicate
    ///     `</think>` — exactly one thinking block is allowed.
    pub(super) fn compile_structural_tag_raw(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let structural_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
                "excludes": ["<think>", "</think>"],
            }
        })
        .to_string();

        let grammar = xgrammar::Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        self.compiler
            .compile_grammar(&grammar)
            .map_err(GrammarError::Compilation)
    }

    /// Thinking-aware structural-tag wrapper.
    ///
    /// Wraps a `triggered_tags` body in a `sequence` that first matches
    /// thinking content via `any_text` with structural-leak excludes,
    /// then `</think>`, then the post-think triggered_tags. The matcher
    /// engages from the first generated token (immediately after the
    /// chat-template's `<think>\n`), so the excluded structural openers
    /// and closers are masked at the bitmask level throughout thinking.
    ///
    /// Excludes — substrings of decoded bytes the model may not produce
    /// inside the thinking region. These are STRUCTURAL invariants only
    /// (openers/closers that would corrupt the post-think state). Model
    /// degeneration patterns (role-word stuttering, template echo) are
    /// NOT excluded here — they are model-quality / chat-template issues
    /// to address at their source, never patched at the grammar layer.
    ///   - `<think>`  — re-open
    ///   - `<function=`, `<tool_call>`, `<parameter=` — tool-call openers
    ///   - `</function>`, `</tool_call>`, `</parameter>` — stray closers
    pub(super) fn compile_thinking_wrapped_structural_tag(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let structural_tag_json = serde_json::json!({
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
                            "</parameter>",
                        ],
                    },
                    {"type": "const_string", "value": "</think>"},
                    {
                        "type": "triggered_tags",
                        "triggers": triggers,
                        "tags": tags,
                        "at_least_one": at_least_one,
                        "stop_after_first": stop_after_first,
                        // Exactly one thinking block. The post-think content
                        // phase must not re-emit `<think>` (which would imply
                        // a second thinking block) nor `</think>` (the unique
                        // closer was consumed by the const_string above).
                        "excludes": ["<think>", "</think>"],
                    },
                ],
            }
        })
        .to_string();

        let grammar = xgrammar::Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        self.compiler
            .compile_grammar(&grammar)
            .map_err(GrammarError::Compilation)
    }

    // ── JSON schema grammar ──

    /// Compile a grammar that enforces a JSON schema.
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_json_schema(
                schema,
                true,                 // any_whitespace
                None,                 // indent
                None::<(&str, &str)>, // separators
                true,                 // strict_mode
                None,                 // max_whitespace_cnt
            )
            .map_err(GrammarError::Compilation)
    }

    /// Compile the built-in JSON grammar (any valid JSON).
    pub fn compile_json_grammar(&mut self) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_builtin_json_grammar()
            .map_err(GrammarError::Compilation)
    }

    /// Compile a grammar from an EBNF string.
    pub fn compile_ebnf(
        &mut self,
        ebnf: &str,
        root_rule: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_grammar_from_ebnf(ebnf, root_rule)
            .map_err(GrammarError::Compilation)
    }
}
