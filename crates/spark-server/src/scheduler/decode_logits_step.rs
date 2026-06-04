// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;
use rayon::prelude::*;

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    let needs_host_logits = active
        .iter()
        .any(|a| a.inside_thinking || a.think_ended || a.grammar_state.is_some())
        || any_logprobs
        || model_logits_fp32;

    let lockprof_sample_t0 = std::time::Instant::now();
    if n >= 2 {
        let temps: Vec<f32> = active.iter().map(|a| a.temperature).collect();
        let thinks: Vec<bool> = active
            .iter()
            .map(|a| a.inside_thinking || a.think_ended)
            .collect();
        tracing::info!(
            target: "atlas::lockprof",
            "decode_gate n={n} all_t0={} any_grammar={} any_logprobs={} fp32={} any_think_state={} temps={:?} thinks={:?}",
            active.iter().all(|a| a.temperature == 0.0),
            any_grammar,
            any_logprobs,
            model_logits_fp32,
            active.iter().any(|a| a.inside_thinking || a.think_ended),
            temps,
            thinks,
        );
    }
    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> =
        if active.iter().all(|a| a.temperature == 0.0) && !any_grammar && !needs_host_logits {
            // Fast path: all greedy, no grammar, no thinking — GPU argmax for the full batch.
            match model.argmax_batch(logits, n, 0) {
                Ok(t) => {
                    tracing::info!(
                        target: "atlas::lockprof",
                        "process_decode_logits n={n} fast_path argmax={:.2}ms",
                        lockprof_sample_t0.elapsed().as_micros() as f64 / 1000.0,
                    );
                    t.into_iter().map(|tok| (tok, None)).collect()
                }
                Err(e) => {
                    tracing::error!("argmax_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
            }
        } else {
            // Host-side path: copy all batch logits to host, sample per-sequence.
            // Required when any sequence has temperature > 0 or grammar constraints.
            let vocab_size = model.vocab_size();
            // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
            // When the model writes FP32 logits to its decode-logits buffer, we
            // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
            // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
            // had no effect on Gemma-4 because this dispatch was never wired —
            // the scheduler always read the (stale) BF16 logits buffer.
            // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
            // on, the per-token decode lm_head writes 4 bytes/element. The
            // passed `logits` pointer is whatever the most-recent forward
            // returned — that's already the correct buffer (prefill or decode).
            // We just need to read it with the matching width.
            let logits_fp32 = model.decode_logits_fp32();
            let elem_bytes = if logits_fp32 { 4 } else { 2 };
            let lockprof_d2h_t0 = std::time::Instant::now();
            let mut buf = vec![0u8; n * vocab_size * elem_bytes];
            if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
                tracing::error!("copy_logits_to_host error: {e:#}");
                for mut a in active.drain(..) {
                    send_error(model, &mut a, &format!("{e:#}"));
                }
                return;
            }
            let lockprof_d2h_ms = lockprof_d2h_t0.elapsed().as_micros() as f64 / 1000.0;
            let lockprof_sample_loop_t0 = std::time::Instant::now();
            // Parallelize per-sequence sampling across rayon's pool. Each
            // iteration owns a disjoint `&mut ActiveSeq` and reads a disjoint
            // slice of `buf`; the model reference and reflection_suppress_ids
            // are shared `&_`. At N=4 this collapses ~25 ms sequential into
            // a single ~11 ms parallel pass.
            let r: Vec<(u32, Option<crate::api::TokenLogprobs>)> = active
                .par_iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        think_end_token,
                        think_start_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        reflection_suppress_ids,
                        adaptive_sampling,
                    )
                })
                .collect();
            let lockprof_sample_loop_ms =
                lockprof_sample_loop_t0.elapsed().as_micros() as f64 / 1000.0;
            tracing::info!(
                target: "atlas::lockprof",
                "process_decode_logits n={n} host_path d2h={:.2}ms sample_loop={:.2}ms total={:.2}ms",
                lockprof_d2h_ms,
                lockprof_sample_loop_ms,
                lockprof_sample_t0.elapsed().as_micros() as f64 / 1000.0,
            );
            r
        };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let lockprof_per_tok_t0 = std::time::Instant::now();
    let now = Instant::now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        a.last_token = tok;
        a.last_token_time = now;

        // Advance grammar state with the sampled token — but only
        // once thinking is finished, because thinking tokens are
        // stripped from the API output and should not consume grammar
        // slots (matches the bitmask-skip in the sampler above).
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
            gs.accept_token(tok);
        }

        // Thinking tokens don't count toward remaining (thinking is "free").
        if a.inside_thinking {
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                a.consecutive_confident = 0;
                a.think_ended = true;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Set force_end_thinking when budget exhausted (picked up next iteration)
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
            a.content_started = true;
            a.content_tokens = a.content_tokens.saturating_add(1);
            // think_just_ended is a one-shot: it was set when the prior
            // token was `</think>`; clear it now that we've emitted the
            // first content token.
            a.think_just_ended = false;

            // F2 (2026-04-26): bounded inter-tool prose budget.
            // Counts only free-text tokens (not inside tool body,
            // not inside grammar-constrained emission). When the
            // budget trips we end the response cleanly so the next
            // turn can re-plan with fresh context, instead of
            // letting the model emit prose↔tool↔prose↔tool
            // forever (the `tool_choice="auto"` grammar never
            // self-terminates — see grammar.rs:461-462).
            if !a.inside_tool_body && a.grammar_state.is_some() {
                a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
                if a.prose_tokens_since_last_tool > MAX_INTER_TOOL_PROSE {
                    tracing::warn!(
                        prose_tokens = a.prose_tokens_since_last_tool,
                        max = MAX_INTER_TOOL_PROSE,
                        "Inter-tool prose budget exhausted, ending response"
                    );
                    a.finished = true;
                }
            }
        }

        // Track <tool_call> opener for the inside-tool-body phase
        // (drives sampler scoping in emit_step.rs).
        if tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.tool_call_opened = true;
            a.prose_tokens_since_last_tool = 0;
        }

        // Accumulate logprobs data for blocking responses.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }

        // </tool_call> stop: in legacy mode (no grammar), stop after first tool call.
        // When grammar is active, allow the model to generate multiple tool calls —
        // the grammar controls when EOS is valid.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            a.output_tokens.push(tok);
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!(
                            "Streaming receiver dropped during tool_call_end, finishing sequence"
                        );
                        a.finished = true;
                        continue;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during tool_call_end backpressure: {e}"
                            );
                            a.finished = true;
                            continue;
                        }
                    }
                }
            }
            if a.grammar_state.is_none() {
                // Legacy mode: one tool call per response
                a.finished = true;
            }
            // Mirror finish_sequence (lines ~3445-3448): keep
            // `inside_tool_body` and the grammar FSM in sync with the
            // emitted token stream. The `continue;` below skips the
            // `emit_token()` path that would normally do this, so
            // without these two lines the flag stays `true` for all
            // subsequent prose tokens — sampler penalties stay
            // disabled for the rest of the response, and the grammar
            // bitmask drifts out of sync with the actual emission.
            // Root-caused 2026-04-26 (8-agent sweep, F1).
            a.inside_tool_body = false;
            if let Some(ref mut gs) = a.grammar_state {
                gs.accept_token(tok);
            }
            // F9 companion (2026-04-26): clear `think_ended` at every
            // </tool_call> boundary so legitimate post-tool
            // re-thinking is allowed. F9 masks <think> when
            // `think_ended=true`, but between tool calls the model
            // SHOULD be allowed to re-think (MiniMax-M2 / Qwen3.6
            // pattern per project_minimax_m27_final.md). F10's
            // watchdog-fire counter still applies — repeated
            // re-thinking that loops will decay its budget.
            a.think_ended = false;
            continue;
        }

        // EOS handling — two principled mechanisms only:
        //   * Grammar: when xgrammar's NPDA isn't at a terminating
        //     state, the EOS token cannot satisfy the grammar.
        //   * min_tokens: OpenAI-spec request parameter; suppress EOS
        //     until output_tokens.len() >= min_tokens.
        // Posthoc quality patches (thinking_suppresses_eos,
        // post_think_suppresses_eos, legacy_suppresses_eos via the
        // deleted require_tool_call) have been removed per Stage 5.2.
        let grammar_suppresses_eos = a
            .grammar_state
            .as_ref()
            .is_some_and(|gs| !gs.is_terminated());
        let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
        let suppress_eos = grammar_suppresses_eos || min_tokens_suppresses;

        if a.eos_tokens.contains(&tok) && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            a.output_tokens.push(tok);
            a.finished = true;
        } else if a.eos_tokens.contains(&tok) && suppress_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
        } else {
            a.output_tokens.push(tok);
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow MUST also be kept off the
            // wire — otherwise opencode persists them as `assistant.content` and
            // on the next turn the model sees its own past garbage (fake
            // `<function=…>`, fake `<tool_response>`) as a "format example" and
            // continues the pattern. Tokens stay in `output_tokens` for the
            // blocking response path's reasoning_content extraction.
            let suppress_stream = a.inside_thinking && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink
                && !suppress_stream
            {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::debug!(
                            "Streaming receiver dropped (decode_logits), finishing seq"
                        );
                        a.finished = true;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during backpressure (decode_logits): {e}"
                            );
                            a.finished = true;
                        }
                    }
                }
            }
            if a.remaining == 0 {
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }

            // Check request timeout.
            if !a.finished
                && let Some(deadline) = a.timeout_at
                && Instant::now() >= deadline
            {
                tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
                a.finished = true;
            }
        }
    }
    tracing::info!(
        target: "atlas::lockprof",
        "process_decode_logits n={n} per_tok_loop={:.2}ms",
        lockprof_per_tok_t0.elapsed().as_micros() as f64 / 1000.0,
    );
}
