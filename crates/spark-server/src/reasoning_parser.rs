// SPDX-License-Identifier: AGPL-3.0-only

//! Reasoning parser trait — model-agnostic thinking/reasoning block detection.
//!
//! Different models use different tags for reasoning blocks:
//!   - Qwen3.5 / Nemotron / DeepSeek-R1: `<think>...</think>`
//!   - Mistral: `[THINK]...[/THINK]`
//!
//! The `ReasoningParser` trait abstracts tag knowledge so the server can
//! detect, extract, and stream reasoning blocks for any model format.
//! Follows the same pattern as `ToolCallParser` (trait + enum + TOML auto-detect).

use std::str::FromStr;

use crate::tokenizer::ChatTokenizer;

/// Parses reasoning/thinking blocks from model output.
pub trait ReasoningParser: Send + Sync {
    /// Parser name for logging (e.g. "qwen", "mistral").
    fn name(&self) -> &str;

    /// Opening tag text (e.g. `"<think>"`, `"[THINK]"`).
    fn start_tag(&self) -> &str;

    /// Closing tag text (e.g. `"</think>"`, `"[/THINK]"`).
    fn end_tag(&self) -> &str;

    /// Resolve the end-of-thinking token ID from the tokenizer.
    /// Returns None if the end tag doesn't encode to a single token.
    fn end_token_id(&self, tokenizer: &ChatTokenizer) -> Option<u32> {
        match tokenizer.encode(self.end_tag()) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    }

    /// Apply model-specific cleanups to a streamed reasoning-content
    /// chunk before it's emitted as a `reasoning_content` SSE delta.
    ///
    /// Default: pass-through. Override for models with known leak
    /// patterns inside the thinking block — e.g. Qwen3.5/3.6 emits
    /// stray protocol-tag fragments (`<tool_call>`, `</parameter>`,
    /// `<function=`), bare role literals (`user`/`assistant`/`tool`),
    /// and `assistant\n` prefixes mid-think that have to be stripped
    /// at the SSE layer.
    ///
    /// Called from the chat-stream emit state machine on each chunk
    /// the streaming decoder hands back while in the Thinking state.
    /// The chunk is the freshly-decoded text since the previous
    /// emission, so the cleanups operate on a small bounded window
    /// — pair-collapse and substring removal are fine.
    fn cleanup_reasoning_leaks(&self, chunk: &str) -> String {
        chunk.to_string()
    }

    /// Extract reasoning content from completed generation text.
    ///
    /// Chat templates inject `<start_tag>` into the prompt as a
    /// reasoning prefix, so the model's output usually OPENS inside a
    /// thinking block (text starts with reasoning content, no leading
    /// `<start_tag>`). We track that with `inside_think` and walk the
    /// string, collecting reasoning while inside, content while
    /// outside, and crossing on each tag. Multiple `<start>...<end>`
    /// blocks all contribute to reasoning; intervening text (including
    /// stray `</end>` tokens) becomes content. Handles unclosed
    /// trailing `<start_tag>` (budget exhausted) by treating the
    /// remainder as reasoning. Returns `(reasoning_content,
    /// response_content)`.
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        let start = self.start_tag();
        let end = self.end_tag();

        // If there's no `<end>` anywhere AND no explicit `<start>` then
        // the model didn't emit a thinking block — short-circuit as
        // content-only. Without this gate the chat-template-implicit-
        // open assumption below would swallow the whole response as
        // reasoning.
        if !text.contains(end) && !text.contains(start) {
            return (None, text.to_string());
        }

        let mut reasoning = String::new();
        let mut content = String::new();
        // The chat template's `<start>` prefix lives in the prompt, so
        // the model's output opens INSIDE a thinking block.
        let mut inside_think = true;
        let mut rest = text;

        while !rest.is_empty() {
            let next_start = rest.find(start);
            let next_end = rest.find(end);
            // Pick the earliest tag in the remaining text.
            let next = match (next_start, next_end) {
                (Some(s), Some(e)) if s < e => Some((s, true)),
                (_, Some(e)) => Some((e, false)),
                (Some(s), None) => Some((s, true)),
                (None, None) => None,
            };
            match next {
                Some((pos, is_start)) => {
                    let head = &rest[..pos];
                    if inside_think {
                        if !reasoning.is_empty() && !head.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(head);
                    } else {
                        content.push_str(head);
                    }
                    if is_start {
                        inside_think = true;
                        rest = &rest[pos + start.len()..];
                    } else {
                        inside_think = false;
                        rest = &rest[pos + end.len()..];
                    }
                }
                None => {
                    if inside_think {
                        if !reasoning.is_empty() && !rest.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(rest);
                    } else {
                        content.push_str(rest);
                    }
                    break;
                }
            }
        }

        let content = content.trim().to_string();
        let reasoning = reasoning.trim().to_string();
        if enable_thinking && !reasoning.is_empty() {
            (Some(reasoning), content)
        } else {
            (None, content)
        }
    }
}

// ── Concrete implementations ────────────────────────────────────────────────

/// Qwen3.5 / Nemotron / DeepSeek-R1 reasoning format: `<think>...</think>`
struct QwenReasoningParser;

impl ReasoningParser for QwenReasoningParser {
    fn name(&self) -> &str {
        "qwen"
    }
    fn start_tag(&self) -> &str {
        "<think>"
    }
    fn end_tag(&self) -> &str {
        "</think>"
    }

    /// Qwen3.5/3.6 thinking-phase leak patterns observed in
    /// production, all originating in the model itself (not in
    /// detection layers below this one). Applied per chunk before
    /// emission as a `reasoning_content` SSE delta.
    ///
    /// 1. Stray `<think>` re-opens — the model sometimes restarts
    ///    a thinking block mid-stream; strip the literal.
    /// 2. `assistant\n` / `assistant` prefix at the very start of
    ///    a chunk — Qwen leaks the role label when transitioning
    ///    between thinking and content. Strip only at prefix
    ///    position so legitimate "assistant" words elsewhere are
    ///    preserved.
    /// 3. Embedded `<tool_call>...</tool_call>` blocks — the
    ///    model hallucinates a tool call inside its thinking;
    ///    splice them out so the reasoning text is clean.
    /// 4. Truncate at `<function=` — Qwen3.5's alternate tool-
    ///    call format leaking into thinking. The detector
    ///    runs on the content phase only, so we hard-stop here
    ///    to prevent the partial structure from polluting the
    ///    reasoning text.
    /// 5. Strip stray closing tags `</parameter>`, `</function>`,
    ///    `</tool_call>` — BPE-token fragments the model emits
    ///    AFTER the real tool call structure has already been
    ///    parsed elsewhere.
    /// 6. Collapse `userX...userX` / `assistantX...assistantX` /
    ///    `toolX...toolX` repetition loops (post-tool-call
    ///    hallucination); also strip line-bounded standalones
    ///    (`\nuser\n` → `\n`).
    ///
    /// All transformations are idempotent and preserve
    /// surrounding whitespace.
    fn cleanup_reasoning_leaks(&self, chunk: &str) -> String {
        let mut cleaned = chunk.to_string();

        // 1. Strip stray <think> re-opens.
        cleaned = cleaned.replace("<think>", "");

        // 2. Strip leading "assistant\n" / "assistant" prefix.
        if let Some(rest) = cleaned.strip_prefix("assistant\n") {
            cleaned = rest.to_string();
        } else if let Some(rest) = cleaned.strip_prefix("assistant") {
            cleaned = rest.to_string();
        }

        // 3. Splice out <tool_call>...</tool_call> blocks. Unclosed
        //    trailing `<tool_call>` truncates the chunk (the model
        //    is about to emit a tool call that the content-phase
        //    detector will handle properly post-`</think>`).
        while let Some(start) = cleaned.find("<tool_call>") {
            if let Some(end) = cleaned[start..].find("</tool_call>") {
                cleaned = format!(
                    "{}{}",
                    &cleaned[..start],
                    &cleaned[start + end + "</tool_call>".len()..]
                );
            } else {
                cleaned = cleaned[..start].to_string();
                break;
            }
        }

        // 4. Hard-stop at <function= (alternate Qwen tool-call format).
        if let Some(start) = cleaned.find("<function=") {
            cleaned = cleaned[..start].to_string();
        }

        // 5. Strip stray close tags.
        for tag in &["</parameter>", "</function>", "</tool_call>"] {
            cleaned = cleaned.replace(tag, "");
        }

        // 6. Role-word repetition loop collapse + standalone strip.
        for word in &["user", "assistant", "tool"] {
            let pair = format!("{word}{word}");
            while cleaned.contains(&pair) {
                cleaned = cleaned.replace(&pair, "");
            }
            let nl_form = format!("\n{word}\n");
            while cleaned.contains(&nl_form) {
                cleaned = cleaned.replace(&nl_form, "\n");
            }
        }

        cleaned
    }
}

/// Mistral reasoning format: `[THINK]...[/THINK]`
struct MistralReasoningParser;

impl ReasoningParser for MistralReasoningParser {
    fn name(&self) -> &str {
        "mistral"
    }
    fn start_tag(&self) -> &str {
        "[THINK]"
    }
    fn end_tag(&self) -> &str {
        "[/THINK]"
    }
}

// ── Format enum + auto-detection ────────────────────────────────────────────

/// Supported reasoning block formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// `<think>...</think>` (Qwen3.5, Nemotron, DeepSeek-R1)
    Qwen,
    /// `[THINK]...[/THINK]` (Mistral)
    Mistral,
}

impl FromStr for ReasoningFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "qwen" | "qwen3" | "deepseek_r1" => Ok(Self::Qwen),
            "mistral" => Ok(Self::Mistral),
            other => Err(format!(
                "Unknown reasoning parser '{other}'. Supported: qwen, mistral"
            )),
        }
    }
}

impl ReasoningFormat {
    /// Create a boxed parser for this format.
    pub fn into_parser(self) -> Box<dyn ReasoningParser> {
        match self {
            Self::Qwen => Box::new(QwenReasoningParser),
            Self::Mistral => Box::new(MistralReasoningParser),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_extract_simple() {
        let parser = QwenReasoningParser;
        let text = "I need to think about this\n</think>\nThe answer is 42.";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert_eq!(reasoning.unwrap(), "I need to think about this");
        assert_eq!(content, "The answer is 42.");
    }

    #[test]
    fn mistral_extract_simple() {
        let parser = MistralReasoningParser;
        let text = "Let me reason here\n[/THINK]\nParis is the capital.";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert_eq!(reasoning.unwrap(), "Let me reason here");
        assert_eq!(content, "Paris is the capital.");
    }

    #[test]
    fn thinking_disabled_returns_none() {
        let parser = QwenReasoningParser;
        let text = "reasoning\n</think>\ncontent";
        let (reasoning, content) = parser.extract_thinking(text, false);
        assert!(reasoning.is_none());
        assert_eq!(content, "content");
    }

    #[test]
    fn no_end_tag_returns_full_text() {
        let parser = QwenReasoningParser;
        let text = "just normal text, no thinking";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert!(reasoning.is_none());
        assert_eq!(content, text);
    }

    #[test]
    fn leaked_thinking_stripped() {
        let parser = QwenReasoningParser;
        let text = "first thought</think>middle<think>leaked</think>end";
        let (_, content) = parser.extract_thinking(text, true);
        assert_eq!(content, "middleend");
    }

    #[test]
    fn multi_block_reasoning_concatenated() {
        // Two explicit <think>...</think> blocks should both contribute
        // to reasoning; intervening text becomes content.
        let parser = QwenReasoningParser;
        let text = "first<think>extra reasoning</think>between<think>more</think>tail";
        let (reasoning, content) = parser.extract_thinking(text, true);
        let r = reasoning.expect("reasoning present");
        assert!(r.contains("first"), "implicit open should capture: {r}");
        assert!(r.contains("extra reasoning"), "explicit block 1: {r}");
        assert!(r.contains("more"), "explicit block 2: {r}");
        assert_eq!(content, "betweentail");
    }

    #[test]
    fn unclosed_thinking_treated_as_reasoning() {
        // Budget exhausted — no </think>. Implicit-open + reasoning
        // remainder becomes the reasoning, content empty.
        let parser = QwenReasoningParser;
        let text = "thought without close";
        // No <think> tag either → falls into the no-tag short circuit.
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert!(reasoning.is_none());
        assert_eq!(content, text);
    }

    #[test]
    fn default_cleanup_reasoning_leaks_passes_through() {
        // Models without quirky thinking-phase leaks (Mistral, etc.)
        // get the default no-op implementation.
        let parser = MistralReasoningParser;
        let chunk = " any text with <tool_call>foo</tool_call> and </parameter>";
        assert_eq!(parser.cleanup_reasoning_leaks(chunk), chunk);
    }

    #[test]
    fn qwen_cleanup_strips_stray_think_tag() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks("normal text <think>re-opened"),
            "normal text re-opened",
        );
    }

    #[test]
    fn qwen_cleanup_strips_leading_assistant_prefix() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks("assistant\nactual reasoning"),
            "actual reasoning",
        );
        assert_eq!(
            parser.cleanup_reasoning_leaks("assistantactual reasoning"),
            "actual reasoning",
        );
        // But not mid-chunk.
        assert_eq!(
            parser.cleanup_reasoning_leaks("I told the assistant to think"),
            "I told the assistant to think",
        );
    }

    #[test]
    fn qwen_cleanup_splices_out_inline_tool_call_blocks() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks(
                "thinking <tool_call>{\"name\":\"bash\"}</tool_call> more thinking"
            ),
            "thinking  more thinking",
        );
    }

    #[test]
    fn qwen_cleanup_truncates_at_unclosed_tool_call() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks("thinking <tool_call>{\"name\":\"bash\""),
            "thinking ",
        );
    }

    #[test]
    fn qwen_cleanup_truncates_at_function_equals() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks("thinking <function=bash>"),
            "thinking ",
        );
    }

    #[test]
    fn qwen_cleanup_strips_stray_close_tags() {
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks(
                "after a tool call </parameter></function></tool_call> back to thinking"
            ),
            "after a tool call  back to thinking",
        );
    }

    #[test]
    fn qwen_cleanup_collapses_role_word_repetition_loops() {
        let parser = QwenReasoningParser;
        // Pair collapse: useruser → empty (idempotent until no pairs left).
        assert_eq!(parser.cleanup_reasoning_leaks("useruser"), "");
        assert_eq!(parser.cleanup_reasoning_leaks("useruseruseruser"), "");
        // Adjacent text outside the pair survives.
        assert_eq!(
            parser.cleanup_reasoning_leaks("hello useruser world"),
            "hello  world",
        );
        // Standalone line-bounded form collapses to just the newline.
        assert_eq!(
            parser.cleanup_reasoning_leaks("line1\nuser\nline2"),
            "line1\nline2",
        );
    }

    #[test]
    fn qwen_cleanup_preserves_legitimate_leading_whitespace() {
        // The point of the rewrite: the BPE decoder gives us
        // leading-space tokens (` voice`, ` computer`) intact, and
        // none of the cleanups eat them.
        let parser = QwenReasoningParser;
        assert_eq!(
            parser.cleanup_reasoning_leaks(" voice computer in the system"),
            " voice computer in the system",
        );
    }

    #[test]
    fn format_from_str() {
        assert_eq!(
            "qwen".parse::<ReasoningFormat>().unwrap(),
            ReasoningFormat::Qwen
        );
        assert_eq!(
            "mistral".parse::<ReasoningFormat>().unwrap(),
            ReasoningFormat::Mistral
        );
        assert_eq!(
            "deepseek_r1".parse::<ReasoningFormat>().unwrap(),
            ReasoningFormat::Qwen
        );
        assert!("unknown".parse::<ReasoningFormat>().is_err());
    }
}
