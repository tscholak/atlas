// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Track CURRENT tool-body phase (P3.1, 2026-04-25). Set on the
    // open token, clear on the close. The flag drives sampler
    // scoping: when true, the main decode path zeroes
    // repetition/presence/frequency/DRY so legitimate JSON
    // micro-repetition (`":"`, `","`, key names) is not penalised.
    if !a.inside_thinking {
        if a.tool_call_start_token == Some(tok) {
            a.inside_tool_body = true;
        } else if a.tool_call_end_token == Some(tok) {
            a.inside_tool_body = false;
        }
    }

    // Advance the grammar matcher on every emitted token, including
    // those inside `<think>...</think>`. The Qwen3 grammar's
    // `compile_thinking_wrapped_structural_tag` wraps the thinking
    // phase in an `any_text` block that allows free text but excludes
    // structural markers (tool-call openers, stray closers,
    // `<think>` re-opens). The matcher engages from token 0 of
    // generation (immediately after the prompt's `<think>\n`).
    if let Some(ref mut gs) = a.grammar_state {
        gs.accept_token(tok);
    }

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);

    // Thinking tokens are "free" (don't decrement remaining).
    // Detect </think> transition. Track thinking token count for budget enforcement.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.think_ended = true;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                tracing::info!("Thinking budget exhausted ({budget} tokens), forcing </think>");
            }
        }
    } else {
        a.remaining -= 1;
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
    }

    // EOS handling: grammar-based, legacy, or min_tokens suppression.
    let grammar_suppresses_eos = a
        .grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated());
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    let suppress_eos = grammar_suppresses_eos || legacy_suppresses_eos || min_tokens_suppresses;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.finished = true;
        return;
    }
    if a.eos_tokens.contains(&tok) && suppress_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // or min_tokens not reached. Don't stop — let the model continue generating.
        return;
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
            enable_thinking,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers, *enable_thinking) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    Some(state)
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}
