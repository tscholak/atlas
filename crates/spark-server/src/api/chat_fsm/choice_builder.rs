// SPDX-License-Identifier: AGPL-3.0-only

//! Buffer FSM events into the inputs for one OpenAI `ChatChoice`.
//!
//! Used by both the blocking adapter (one builder per choice index)
//! and the streaming adapter (one builder for the dump). The builder
//! does NO posthoc cleanup or rewriting — it just accumulates the
//! event stream into typed buckets and emits a `ChatChoice` via
//! `into_chat_choice`.

use std::collections::HashMap;

use crate::citation;
use crate::openai::{ChatChoice, ChatMessage};
use crate::tool_parser::{FunctionCall, ToolCall};

use super::events::{FsmEvent, StopReason};

#[derive(Debug)]
pub struct ChoiceBuilder {
    choice_idx: usize,
    content_buf: String,
    reasoning_buf: String,
    tool_calls: Vec<ToolCall>,
    pending: HashMap<usize, (String, String, String)>, // (id, name, args)
    stop_reason: Option<StopReason>,
}

impl ChoiceBuilder {
    pub fn new(choice_idx: usize) -> Self {
        Self {
            choice_idx,
            content_buf: String::new(),
            reasoning_buf: String::new(),
            tool_calls: Vec::new(),
            pending: HashMap::new(),
            stop_reason: None,
        }
    }

    pub fn apply(&mut self, ev: FsmEvent) {
        match ev {
            FsmEvent::ReasoningDelta(text) => self.reasoning_buf.push_str(&text),
            FsmEvent::ContentDelta(text) => self.content_buf.push_str(&text),
            FsmEvent::ToolCallStart { id, name, idx } => {
                self.pending.insert(idx, (id, name, String::new()));
            }
            FsmEvent::ToolCallArgDelta { args, idx } => {
                if let Some((_, _, args_buf)) = self.pending.get_mut(&idx) {
                    args_buf.push_str(&args);
                }
            }
            FsmEvent::ToolCallEnd { idx } => {
                if let Some((id, name, arguments)) = self.pending.remove(&idx) {
                    self.tool_calls.push(ToolCall {
                        id,
                        call_type: "function".to_string(),
                        function: FunctionCall { name, arguments },
                    });
                }
            }
            FsmEvent::Stopped { reason } => {
                self.stop_reason = Some(reason);
            }
        }
    }

    pub fn content_str(&self) -> &str {
        &self.content_buf
    }

    pub fn reasoning_str(&self) -> &str {
        &self.reasoning_buf
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Materialise the OpenAI `ChatChoice`. Logprobs (which depend on
    /// scheduler-side `InferenceResponse.logprobs`) are supplied by
    /// the caller — the builder doesn't know about per-token
    /// logprobs.
    pub fn into_chat_choice(
        self,
        logprobs: Option<crate::openai::ChoiceLogprobs>,
    ) -> ChatChoice {
        let reasoning_content = if self.reasoning_buf.is_empty() {
            None
        } else {
            Some(self.reasoning_buf.clone())
        };
        let annotations = citation::merged_annotations(&self.content_buf);
        let content = if !self.tool_calls.is_empty() && self.content_buf.is_empty() {
            None
        } else {
            Some(self.content_buf)
        };
        let tool_calls = if self.tool_calls.is_empty() {
            None
        } else {
            Some(self.tool_calls)
        };
        let finish_reason = super::assemble::compute_finish_reason(
            self.stop_reason.as_ref(),
            tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()),
        );
        ChatChoice {
            index: self.choice_idx,
            message: ChatMessage {
                role: "assistant".to_string(),
                reasoning_content: reasoning_content.clone(),
                reasoning: reasoning_content,
                content,
                tool_calls,
                annotations,
                refusal: None,
            },
            finish_reason,
            logprobs,
        }
    }
}
