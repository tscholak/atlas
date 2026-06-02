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
//!
//! ## Architecture: streaming = non-streaming, both via `ThinkingScanner`
//!
//! The chat-stream emit pipeline runs over a special-token-aware decoder
//! (`skip_special_tokens=false`), so the FSM sees protocol markers as
//! literal text and demuxes by substring match. The per-phase parsers
//! are stateful with a safe-emit idiom:
//!
//!   - **Content phase**: `tool_parser::StreamingToolDetector` +
//!     `api::sanitizer::sanitize_content_chunk`. Both carry buffer
//!     state across calls; both compute `tag_max - 1` as the held-back
//!     tail size so partial tags straddling chunk boundaries can fuse
//!     with the next chunk before the rules look for them.
//!
//!   - **Thinking phase**: `ThinkingScanner` (this file). Same stateful
//!     safe-emit idiom. Model-specific leak patterns (Qwen3.5/3.6
//!     hallucinated `<think>` re-opens, role-word loops, stray
//!     tool-call XML fragments) are recognized in the buffered
//!     accumulator, not per-chunk.
//!
//! The non-streaming `extract_thinking` API drives the SAME scanner —
//! it just feeds the entire generated text in one `process` call,
//! then `flush`es. Streaming and non-streaming therefore produce
//! byte-equivalent reasoning output for the same input text.

use std::str::FromStr;

use crate::tokenizer::ChatTokenizer;

// ── ThinkingScanner: stateful streaming parser for the Thinking phase ──────
//
// Mirrors `StreamingToolDetector`'s shape: process chunks, hold a
// tail buffer for safe-emit, emit incrementally; flush at end-of-stream.

/// Result of feeding one decoder chunk to a `ThinkingScanner`.
#[derive(Debug)]
pub enum ThinkingScanResult {
    /// Continue in the Thinking phase. `emit` is the bytes safe to
    /// send as a `reasoning_content` SSE delta (may be empty if the
    /// scanner is holding bytes pending tag straddle).
    Continue { emit: String },

    /// The reasoning-block end-tag (e.g. `</think>`) was found in
    /// the buffered stream. The FSM transitions to Content phase.
    /// `final_reasoning` is the last reasoning_content delta
    /// (post-cleanup, includes any held buffer contents up to the
    /// tag). `content_start` is the post-tag remainder, with leading
    /// whitespace trimmed — the FSM feeds it to Content-phase
    /// processing.
    Transition { final_reasoning: String, content_start: String },
}

/// Stateful streaming scanner for the Thinking phase. Implementations
/// own model-specific leak-pattern handling and an internal buffer
/// large enough to detect those patterns across chunk boundaries.
pub trait ThinkingScanner: Send + Sync {
    /// Feed one decoded chunk. Returns the safe-emit bytes plus, if
    /// the end-tag was found, a transition signal.
    fn process(&mut self, chunk: &str) -> ThinkingScanResult;

    /// Drain any held bytes after rules are applied one last time.
    /// Called when the stream ends mid-thinking (e.g. `max_tokens`
    /// hit before `</think>`).
    fn flush(&mut self) -> String;

    /// Reset state. Called when the FSM re-enters Thinking from
    /// Content (hallucinated `<think>` re-open).
    fn reset(&mut self);
}

// ── Generic NoopThinkingScanner ────────────────────────────────────────────
//
// Used by models without model-specific leak quirks (e.g. Mistral).
// Handles end-tag detection only; the chunk's bytes flow through
// unchanged.

struct NoopThinkingScanner {
    buf: String,
    end_tag: &'static str,
}

impl NoopThinkingScanner {
    fn new(end_tag: &'static str) -> Self {
        Self { buf: String::new(), end_tag }
    }

    /// Hold-back size: anything shorter than the end-tag could still
    /// be a partial prefix waiting to complete.
    fn hold(&self) -> usize {
        self.end_tag.len().saturating_sub(1)
    }
}

impl ThinkingScanner for NoopThinkingScanner {
    fn process(&mut self, chunk: &str) -> ThinkingScanResult {
        self.buf.push_str(chunk);
        if let Some(pos) = self.buf.find(self.end_tag) {
            let after_start = pos + self.end_tag.len();
            let final_reasoning = self.buf[..pos].to_string();
            let content_start = self.buf[after_start..].trim_start().to_string();
            self.buf.clear();
            return ThinkingScanResult::Transition { final_reasoning, content_start };
        }
        let hold = self.hold();
        let total = self.buf.len();
        if total <= hold {
            return ThinkingScanResult::Continue { emit: String::new() };
        }
        let mut split = total - hold;
        while split > 0 && !self.buf.is_char_boundary(split) {
            split -= 1;
        }
        let emit: String = self.buf.drain(..split).collect();
        ThinkingScanResult::Continue { emit }
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    fn reset(&mut self) {
        self.buf.clear();
    }
}

// ── QwenThinkingScanner ────────────────────────────────────────────────────
//
// Qwen3.5/3.6 thinking-phase leak patterns observed in production,
// all originating in the model itself (not in detection layers below
// this one). The scanner applies these rules on a buffer that
// accumulates across chunks, so a leak pattern that arrives split
// across multiple decoder steps (e.g. `<t` + `hink>`) is still
// caught when the assembled bytes appear in the buffer.
//
//   1. Stray `<think>` re-opens — the model sometimes restarts a
//      thinking block mid-stream; strip the literal.
//   2. `assistant\n` / `assistant` prefix at the start of thinking —
//      Qwen leaks the role label. Strip only at the absolute start
//      (gated by `started: bool`).
//   3. Embedded `<tool_call>...</tool_call>` blocks — hallucinated
//      tool calls inside thinking; splice out completed blocks.
//      Unclosed `<tool_call>` is held back by the safe-emit boundary
//      until a close arrives; if the stream ends with an unclosed
//      tool_call, `flush` drops the unclosed remainder.
//   4. Hard-stop at `<function=` — alternate Qwen tool-call format.
//      Once seen, the scanner enters a permanent drop state until
//      `reset()`.
//   5. Strip stray closing tags `</parameter>`, `</function>`,
//      `</tool_call>` — BPE-token fragments the model emits after
//      the real tool call has been parsed elsewhere.
//   6. Collapse role-word repetition loops: `useruser` / `\nuser\n`
//      (same for `assistant`, `tool`).
//
// All rules are idempotent and operate on `self.buf` in place.

/// Longest leak-pattern tag the rules look for. Determines the
/// safe-emit hold size (tag_max - 1 bytes retained as potential
/// partial-prefix tail). `</parameter>` and `</tool_call>` are both
/// 12 bytes (the longest).
const QWEN_LEAK_TAG_MAX: usize = 12;

struct QwenThinkingScanner {
    /// Accumulated decoded text since last emit. The safe-emit drains
    /// a prefix and retains the tail bounded by `QWEN_LEAK_TAG_MAX - 1`.
    buf: String,
    /// True once any non-whitespace byte has been emitted. Used by
    /// rule 2 (leading `assistant` strip) which only fires before
    /// content has started.
    started: bool,
    /// True once rule 4's `<function=` hard-stop has fired. All
    /// subsequent process calls return empty emits until `reset()`.
    hard_stopped: bool,
    /// End-tag for transition detection.
    end_tag: &'static str,
}

impl QwenThinkingScanner {
    fn new() -> Self {
        Self {
            buf: String::new(),
            started: false,
            hard_stopped: false,
            end_tag: "</think>",
        }
    }

    /// Apply the 6 leak rules to `self.buf` in place.
    fn apply_rules_in_place(&mut self) {
        // Rule 1: strip `<think>` re-opens.
        while let Some(pos) = self.buf.find("<think>") {
            self.buf.replace_range(pos..pos + "<think>".len(), "");
        }

        // Rule 2: strip leading `assistant\n` / `assistant` prefix.
        // Only at absolute start of thinking.
        if !self.started {
            if let Some(rest) = self.buf.strip_prefix("assistant\n") {
                self.buf = rest.to_string();
            } else if let Some(rest) = self.buf.strip_prefix("assistant") {
                self.buf = rest.to_string();
            }
        }

        // Rule 3: splice out `<tool_call>...</tool_call>` blocks.
        // Unclosed trailing `<tool_call>` left in buf — safe-emit
        // hold keeps it from being emitted until `</tool_call>` arrives
        // or `flush` runs.
        while let Some(start) = self.buf.find("<tool_call>") {
            if let Some(end_rel) = self.buf[start..].find("</tool_call>") {
                let end = start + end_rel + "</tool_call>".len();
                self.buf.replace_range(start..end, "");
            } else {
                // Unclosed at end-of-buffer — leave alone for now.
                break;
            }
        }

        // Rule 4: hard-stop at `<function=`. Drop the tail and latch
        // hard_stopped so future chunks are silently dropped.
        if let Some(pos) = self.buf.find("<function=") {
            self.buf.truncate(pos);
            self.hard_stopped = true;
        }

        // Rule 5: strip stray close tags.
        for tag in ["</parameter>", "</function>", "</tool_call>"] {
            while let Some(pos) = self.buf.find(tag) {
                self.buf.replace_range(pos..pos + tag.len(), "");
            }
        }

        // Rule 6: collapse role-word loops.
        for word in ["user", "assistant", "tool"] {
            let pair: String = [word, word].concat();
            while let Some(pos) = self.buf.find(pair.as_str()) {
                self.buf.replace_range(pos..pos + pair.len(), "");
            }
            let nl_form = format!("\n{word}\n");
            while let Some(pos) = self.buf.find(nl_form.as_str()) {
                self.buf.replace_range(pos..pos + nl_form.len(), "\n");
            }
        }
    }
}

impl ThinkingScanner for QwenThinkingScanner {
    fn process(&mut self, chunk: &str) -> ThinkingScanResult {
        if self.hard_stopped {
            return ThinkingScanResult::Continue { emit: String::new() };
        }
        self.buf.push_str(chunk);

        // (a) End-tag detection. The `</think>` substring search
        //     runs FIRST so a real end-of-thinking transition is
        //     recognized even if leak patterns appear before it.
        if let Some(pos) = self.buf.find(self.end_tag) {
            let pre_tag: String = self.buf[..pos].to_string();
            let after_tag: String =
                self.buf[pos + self.end_tag.len()..].trim_start().to_string();
            self.buf.clear();
            // Run rules on pre_tag (one-shot, no safe-emit hold).
            std::mem::swap(&mut self.buf, &mut { pre_tag });
            self.apply_rules_in_place();
            let final_reasoning = std::mem::take(&mut self.buf);
            return ThinkingScanResult::Transition {
                final_reasoning,
                content_start: after_tag,
            };
        }

        // (b) Apply leak rules in-place on the buffer.
        self.apply_rules_in_place();
        if self.hard_stopped {
            // Rule 4 fired during this call — buf was truncated.
            // Emit whatever's left (may be empty), then future
            // processes drop everything.
            let emit = std::mem::take(&mut self.buf);
            if !emit.is_empty() && !emit.chars().all(char::is_whitespace) {
                self.started = true;
            }
            return ThinkingScanResult::Continue { emit };
        }

        // (c) Safe-emit boundary. Two sources of hold contribute:
        //
        //   - `QWEN_LEAK_TAG_MAX - 1` bytes for any leak-pattern
        //     prefix at the tail that might complete on the next
        //     chunk.
        //   - If rule 3 found an unclosed `<tool_call>` (left in
        //     place because `</tool_call>` hasn't arrived yet),
        //     hold from THAT position to the end of buf — the
        //     close could arrive in any subsequent chunk, possibly
        //     after a long argument body. Emitting any partial
        //     `<tool_call>` text would leak XML fragments into the
        //     reasoning stream.
        let total = self.buf.len();
        let hold_for_pattern = QWEN_LEAK_TAG_MAX - 1;
        let hold_for_unclosed_tool_call = self
            .buf
            .find("<tool_call>")
            .map(|pos| total - pos)
            .unwrap_or(0);
        let hold = hold_for_pattern.max(hold_for_unclosed_tool_call);
        if total <= hold {
            return ThinkingScanResult::Continue { emit: String::new() };
        }
        let mut split = total - hold;
        while split > 0 && !self.buf.is_char_boundary(split) {
            split -= 1;
        }
        let emit: String = self.buf.drain(..split).collect();
        if !emit.is_empty() && !emit.chars().all(char::is_whitespace) {
            self.started = true;
        }
        ThinkingScanResult::Continue { emit }
    }

    fn flush(&mut self) -> String {
        if self.hard_stopped {
            self.buf.clear();
            return String::new();
        }
        // Drop any unclosed `<tool_call>` tail (rule 3 didn't truncate
        // because a close might have arrived; flush means no more
        // chunks coming, so it won't).
        if let Some(start) = self.buf.find("<tool_call>") {
            self.buf.truncate(start);
        }
        self.apply_rules_in_place();
        std::mem::take(&mut self.buf)
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.started = false;
        self.hard_stopped = false;
    }
}

// ── ReasoningParser trait ───────────────────────────────────────────────────

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

    /// Build a stateful scanner for incremental thinking-phase
    /// parsing. The scanner owns any model-specific leak-pattern
    /// handling AND the end-tag detection logic. The chat-stream
    /// FSM drives it per-chunk; `extract_thinking` (non-streaming)
    /// drives it with a single chunk + flush.
    fn create_thinking_scanner(&self) -> Box<dyn ThinkingScanner>;

    /// Extract reasoning content from completed (non-streaming)
    /// generation text. Drives `create_thinking_scanner` with the
    /// whole text in one call — yields byte-equivalent results to
    /// the streaming path for the same input.
    ///
    /// Chat templates inject `<start_tag>` into the prompt as a
    /// reasoning prefix, so the model's output usually OPENS inside
    /// a thinking block (text starts with reasoning content, no
    /// leading `<start_tag>`). Returns `(reasoning_content,
    /// response_content)`.
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        // Short-circuit: no thinking markers anywhere → content-only.
        // Without this gate, the implicit-open assumption below would
        // swallow the whole response as reasoning.
        if !text.contains(self.end_tag()) && !text.contains(self.start_tag()) {
            return (None, text.to_string());
        }

        let mut scanner = self.create_thinking_scanner();
        let mut reasoning_buf = String::new();
        let mut content_buf = String::new();

        match scanner.process(text) {
            ThinkingScanResult::Continue { emit } => {
                // No `</think>` in the text → model never closed the
                // thinking block (budget exhausted or template
                // didn't open). The implicit-open assumption holds:
                // entire (cleaned) text is reasoning.
                reasoning_buf.push_str(&emit);
                reasoning_buf.push_str(&scanner.flush());
            }
            ThinkingScanResult::Transition { final_reasoning, content_start } => {
                reasoning_buf.push_str(&final_reasoning);
                content_buf.push_str(&content_start);
                // The scanner has no held state after Transition;
                // explicit flush for symmetry.
                let _ = scanner.flush();
            }
        }

        // Some models emit multiple `<think>...</think>` blocks. The
        // current scanner-based path handles the FIRST end-tag; for
        // subsequent blocks we re-scan the content_buf. This matches
        // the prior `extract_thinking` semantics.
        loop {
            let start = self.start_tag();
            let end = self.end_tag();
            let Some(s_pos) = content_buf.find(start) else { break };
            let after_start = s_pos + start.len();
            let Some(e_rel) = content_buf[after_start..].find(end) else {
                // Unclosed inner block; treat the trailing tail as
                // reasoning, drop the open tag.
                let pre = content_buf[..s_pos].to_string();
                let extra_reasoning = content_buf[after_start..].to_string();
                if !reasoning_buf.is_empty() && !extra_reasoning.is_empty() {
                    reasoning_buf.push('\n');
                }
                reasoning_buf.push_str(extra_reasoning.as_str());
                content_buf = pre;
                break;
            };
            let e_pos = after_start + e_rel;
            // Inner reasoning block: append its contents (cleaned)
            // to reasoning_buf; drop the tag pair from content_buf.
            let inner = content_buf[after_start..e_pos].to_string();
            let mut sub_scanner = self.create_thinking_scanner();
            let cleaned_inner = match sub_scanner.process(&inner) {
                ThinkingScanResult::Continue { emit } => {
                    let mut acc = emit;
                    acc.push_str(&sub_scanner.flush());
                    acc
                }
                ThinkingScanResult::Transition { final_reasoning, .. } => {
                    // Inner block contained another `</think>` —
                    // unusual; take the first split as the inner
                    // reasoning.
                    final_reasoning
                }
            };
            if !reasoning_buf.is_empty() && !cleaned_inner.is_empty() {
                reasoning_buf.push('\n');
            }
            reasoning_buf.push_str(&cleaned_inner);
            // Remove the consumed `<think>...</think>` from content_buf.
            let after_end = e_pos + end.len();
            content_buf = format!(
                "{}{}",
                &content_buf[..s_pos],
                &content_buf[after_end..],
            );
        }

        let content = content_buf.trim().to_string();
        let reasoning = reasoning_buf.trim().to_string();
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
    fn create_thinking_scanner(&self) -> Box<dyn ThinkingScanner> {
        Box::new(QwenThinkingScanner::new())
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
    fn create_thinking_scanner(&self) -> Box<dyn ThinkingScanner> {
        Box::new(NoopThinkingScanner::new("[/THINK]"))
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: drive a scanner with a sequence of chunks and return
    /// the joined `Continue` emits PLUS any final flush content.
    /// Stops at the first `Transition` and asserts no later chunks
    /// were provided (use `drive_until_transition` for that case).
    fn drive_to_end(
        scanner: &mut Box<dyn ThinkingScanner>,
        chunks: &[&str],
    ) -> String {
        let mut joined = String::new();
        for chunk in chunks {
            match scanner.process(chunk) {
                ThinkingScanResult::Continue { emit } => joined.push_str(&emit),
                ThinkingScanResult::Transition { .. } => {
                    panic!("unexpected Transition during drive_to_end");
                }
            }
        }
        joined.push_str(&scanner.flush());
        joined
    }

    /// Helper: drive a scanner with chunks until a Transition fires.
    /// Returns (joined Continue emits, final_reasoning, content_start).
    fn drive_until_transition(
        scanner: &mut Box<dyn ThinkingScanner>,
        chunks: &[&str],
    ) -> (String, String, String) {
        let mut continued = String::new();
        for chunk in chunks {
            match scanner.process(chunk) {
                ThinkingScanResult::Continue { emit } => continued.push_str(&emit),
                ThinkingScanResult::Transition { final_reasoning, content_start } => {
                    return (continued, final_reasoning, content_start);
                }
            }
        }
        panic!("drive_until_transition: no Transition reached in {chunks:?}");
    }

    // ── Group A: single-chunk rule coverage ────────────────────────

    #[test]
    fn a01_single_chunk_no_quirks_passes_through() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["hello world"]);
        assert_eq!(joined, "hello world");
    }

    #[test]
    fn a02_strip_think_reopen_within_chunk() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["text<think>more"]);
        assert_eq!(joined, "textmore");
    }

    #[test]
    fn a03_strip_leading_assistant_newline() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["assistant\nactual thinking"]);
        assert_eq!(joined, "actual thinking");
    }

    #[test]
    fn a04_strip_leading_assistant_no_newline() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["assistantactual"]);
        assert_eq!(joined, "actual");
    }

    #[test]
    fn a05_splice_complete_tool_call_block_in_chunk() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["before<tool_call>x</tool_call>after"]);
        assert_eq!(joined, "beforeafter");
    }

    #[test]
    fn a06_unclosed_tool_call_dropped_at_flush() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["before<tool_call>fragment"]);
        assert_eq!(joined, "before");
    }

    #[test]
    fn a07_hard_stop_at_function_eq_truncates() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["before<function=foo>baz", "more"]);
        assert_eq!(joined, "before");
    }

    #[test]
    fn a08_strip_stray_close_tag_parameter() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["a</parameter>b"]);
        assert_eq!(joined, "ab");
    }

    #[test]
    fn a09_strip_stray_close_tag_function() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["a</function>b"]);
        assert_eq!(joined, "ab");
    }

    #[test]
    fn a10_strip_stray_close_tag_tool_call() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["a</tool_call>b"]);
        assert_eq!(joined, "ab");
    }

    #[test]
    fn a11_collapse_role_word_pair() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["xuseruserx"]);
        assert_eq!(joined, "xx");
    }

    #[test]
    fn a12_strip_newline_bounded_role_word() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["a\nassistant\nb"]);
        assert_eq!(joined, "a\nb");
    }

    #[test]
    fn a13_preserves_legitimate_leading_whitespace() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        // The previous bug: cleanups eating space tokens. The scanner
        // must NOT touch ordinary whitespace.
        let joined = drive_to_end(&mut s, &[" voice computer in the system"]);
        assert_eq!(joined, " voice computer in the system");
    }

    // ── Group B: cross-chunk leak regression ───────────────────────

    #[test]
    fn b01_cross_chunk_think_split_at_tag_boundary() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["text<t", "hink>more"]);
        assert_eq!(joined, "textmore");
    }

    #[test]
    fn b02_cross_chunk_think_arrives_split_via_tokens() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(
            &mut s,
            &["text", "<", "th", "in", "k", ">", "more"],
        );
        assert_eq!(joined, "textmore");
    }

    #[test]
    fn b03_cross_chunk_assistant_newline_split() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["text\nass", "istant\nmore"]);
        assert_eq!(joined, "text\nmore");
    }

    #[test]
    fn b04_cross_chunk_function_eq_hard_stop() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(&mut s, &["text<func", "tion=name>more"]);
        assert_eq!(joined, "text");
    }

    #[test]
    fn b05_cross_chunk_tool_call_block_complete() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let joined = drive_to_end(
            &mut s,
            &["text<tool_", "call>fragment</tool_", "call>after"],
        );
        assert_eq!(joined, "textafter");
    }

    #[test]
    fn b06_screenshot_leak_pattern_regression() {
        // The actual bug captured in
        // `Desktop/Screenshot 2026-06-02 at 11.02.02 AM.png`.
        // Tokens roughly approximate the decoder's chunking of the
        // model's hallucinated `assistant<think>` sequence followed
        // by the real `</think>`. The last chunk fuses `</think>`
        // with the first content token so the Transition's
        // `content_start` is non-empty (asserts the post-tag flow
        // through to Content phase).
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let (continued, final_reasoning, content_start) = drive_until_transition(
            &mut s,
            &[
                "Timer", " set", ".", "\n\n", "assistant", "<think>", "\n\n",
                "</think>rest",
            ],
        );
        let joined_reasoning = format!("{continued}{final_reasoning}");
        // Both leaks must be gone:
        assert!(
            !joined_reasoning.contains("<think>"),
            "joined reasoning must not contain `<think>` literal: {joined_reasoning:?}",
        );
        assert!(
            !joined_reasoning.contains("\nassistant\n"),
            "joined reasoning must not contain bare \\nassistant\\n: {joined_reasoning:?}",
        );
        // The real text still survives:
        assert!(
            joined_reasoning.contains("Timer set."),
            "expected `Timer set.` in joined reasoning: {joined_reasoning:?}",
        );
        // Post-tag content flows to Content phase:
        assert_eq!(content_start, "rest");
    }

    // ── Group C: flush + reset behavior ────────────────────────────

    #[test]
    fn c01_flush_drains_held_safe_tail() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        // 5 bytes < hold (11) → entirely held during process.
        let res = s.process("hello");
        match res {
            ThinkingScanResult::Continue { emit } => {
                assert_eq!(emit, "", "5-byte input should be entirely held");
            }
            ThinkingScanResult::Transition { .. } => panic!("unexpected transition"),
        }
        assert_eq!(s.flush(), "hello");
    }

    #[test]
    fn c02_flush_applies_rules_one_last_time() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        // Neither chunk alone forms `\nassistant\n` but the buffer
        // assembled across them does.
        let _ = s.process("text\n");
        let _ = s.process("assistant\n");
        // flush() applies rules then drains; rule 6's `\nassistant\n`
        // → `\n` should fire.
        let tail = s.flush();
        assert_eq!(tail, "text\n");
    }

    #[test]
    fn c03_reset_clears_state() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let _ = s.process("text");
        s.reset();
        let joined = drive_to_end(&mut s, &["other"]);
        assert_eq!(joined, "other");
    }

    #[test]
    fn c04_transition_with_post_tag_leading_whitespace_trim() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let (continued, final_reasoning, content_start) = drive_until_transition(
            &mut s,
            &["text</think>\n\nmore"],
        );
        assert_eq!(continued, "");
        assert_eq!(final_reasoning, "text");
        assert_eq!(content_start, "more");
    }

    // ── Group D: transition edge cases ─────────────────────────────

    #[test]
    fn d01_transition_at_chunk_boundary() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let (continued, final_reasoning, content_start) = drive_until_transition(
            &mut s,
            &["text</", "think>more"],
        );
        let joined = format!("{continued}{final_reasoning}");
        assert_eq!(joined, "text");
        assert_eq!(content_start, "more");
    }

    #[test]
    fn d02_transition_with_no_post_tag_text() {
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let (continued, final_reasoning, content_start) = drive_until_transition(
            &mut s,
            &["text</think>"],
        );
        let joined = format!("{continued}{final_reasoning}");
        assert_eq!(joined, "text");
        assert_eq!(content_start, "");
    }

    #[test]
    fn d03_end_tag_takes_priority_over_rules() {
        // Leading `<think>` literal AND a closing `</think>`.
        // The transition is detected first; rules clean the pre-tag.
        let parser = QwenReasoningParser;
        let mut s = parser.create_thinking_scanner();
        let (continued, final_reasoning, content_start) = drive_until_transition(
            &mut s,
            &["<think>middle</think>after"],
        );
        let joined = format!("{continued}{final_reasoning}");
        assert_eq!(joined, "middle");
        assert_eq!(content_start, "after");
    }

    // ── Non-streaming surface (extract_thinking via scanner) ───────

    #[test]
    fn e01_extract_thinking_qwen_simple() {
        let parser = QwenReasoningParser;
        let text = "I need to think about this\n</think>\nThe answer is 42.";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert_eq!(reasoning.unwrap(), "I need to think about this");
        assert_eq!(content, "The answer is 42.");
    }

    #[test]
    fn e02_extract_thinking_mistral_simple() {
        let parser = MistralReasoningParser;
        let text = "Let me reason here\n[/THINK]\nParis is the capital.";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert_eq!(reasoning.unwrap(), "Let me reason here");
        assert_eq!(content, "Paris is the capital.");
    }

    #[test]
    fn e03_extract_thinking_disabled_returns_none() {
        let parser = QwenReasoningParser;
        let text = "reasoning\n</think>\ncontent";
        let (reasoning, content) = parser.extract_thinking(text, false);
        assert!(reasoning.is_none());
        assert_eq!(content, "content");
    }

    #[test]
    fn e04_extract_thinking_no_tags_returns_content_only() {
        let parser = QwenReasoningParser;
        let text = "just normal text, no thinking";
        let (reasoning, content) = parser.extract_thinking(text, true);
        assert!(reasoning.is_none());
        assert_eq!(content, text);
    }

    #[test]
    fn e05_extract_thinking_multi_block_concatenated() {
        // Two explicit `<think>...</think>` blocks both contribute
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
    fn e06_extract_thinking_uses_scanner_cleanup() {
        // Non-streaming Qwen path applies the SAME quirk cleanup as
        // streaming (the bug we're closing here on the streaming
        // side was previously silently present in non-streaming as
        // well — `extract_thinking` did substring split only).
        let parser = QwenReasoningParser;
        let text = "Timer set.\n\nassistant<think>\n</think>answer";
        let (reasoning, content) = parser.extract_thinking(text, true);
        let r = reasoning.expect("reasoning present");
        assert!(
            !r.contains("<think>"),
            "non-streaming extract_thinking must clean `<think>` literals: {r:?}",
        );
        assert!(
            !r.contains("\nassistant\n"),
            "non-streaming extract_thinking must collapse role-word loops: {r:?}",
        );
        assert!(r.contains("Timer set."), "real text survives: {r:?}");
        assert_eq!(content, "answer");
    }

    // ── Format parsing ─────────────────────────────────────────────

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
