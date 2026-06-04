// SPDX-License-Identifier: AGPL-3.0-only

//! Chat-completion FSM shared by streaming and blocking adapters.
//!
//! The FSM consumes scheduler-emitted token ids one at a time, drives
//! a `StreamingDecoder` + a `ThinkingScanner` (in the Thinking phase) +
//! a `StreamingToolDetector` (in the Content phase), and emits a flat
//! sequence of `FsmEvent`s. The streaming adapter (`chat_stream/`)
//! translates each event to an SSE `ChatCompletionChunk`. The blocking
//! adapter (`chat_fsm::assemble::assemble_choice`) buffers events into
//! a `ChoiceBuilder` and materialises a `ChatChoice`. Both adapters
//! call `assemble::assemble_chat_response` to build the final
//! `ChatCompletionResponse` JSON.
//!
//! See `/Users/tscholak/.claude/plans/i-ve-deployed-the-heim-snazzy-swan.md`
//! Stage 3 for the design rationale and migration sequence.

#![allow(dead_code, unused_imports)]

pub mod assemble;
pub mod choice_builder;
pub mod events;
pub mod stepper;
pub mod stop_predicate;

#[cfg(test)]
mod tests;

pub use assemble::{assemble_chat_response, assemble_choice};
pub use choice_builder::ChoiceBuilder;
pub use events::{FsmEvent, StopReason};
pub use stepper::{Stepper, StepperConfig};
