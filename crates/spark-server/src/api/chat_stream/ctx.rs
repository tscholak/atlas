// SPDX-License-Identifier: AGPL-3.0-only
//
// Read-only context shared by the streaming `flat_map` closure. The
// adapter (see `adapter.rs`) owns the bulk of this; `ctx` survives
// only for the error-arm path (`handle_error.rs`), which needs the
// app state for the rate-limit refund.

use std::sync::Arc;

use crate::AppState;

pub(super) struct StreamCtx {
    pub(super) state: Arc<AppState>,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
}
