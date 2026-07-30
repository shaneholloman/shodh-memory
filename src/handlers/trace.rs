//! Witnessed-operation capture middleware (traceability slice 1, Task 3).
//!
//! One choke point wraps every protected route: the middleware inserts an
//! [`OpTrace`] extension, the handler enriches it with body-derived identity
//! and evidence, and after the response is produced exactly one
//! [`OpRecordDraft`] is appended to the per-user operation log
//! (spec `docs/superpowers/specs/2026-07-30-agent-traceability-design.md` §3.1).
//!
//! Capture policy (audit amendments 9–13):
//! - Mounted INSIDE `build_protected_routes` so both transports — HTTP (with
//!   auth) and local IPC (`server.rs`, no auth layer) — are covered
//!   structurally.
//! - Excluded: `/api/trace/*` (capture must not observe itself), `/metrics`
//!   (a scrape is not an agent op), and the four SSE/WS streaming routes
//!   (long-lived connections are not discrete ops; their capture story is a
//!   future slice).
//! - Identity-less requests: some protected routes carry no user identity by
//!   construction (see [`NO_IDENTITY_OPS`]); for those, and for any request
//!   whose handler never set an identity, the record is DROPPED and the
//!   [`crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL`] counter says so loudly.
//!   We never append under a synthetic user: creating a user store on miss
//!   costs up to ~750 ms of retry sleeps and materializes junk user dirs.
//! - Capture failure never fails the operation and is never silent: counter
//!   increment + best-effort `oplog_mark_incomplete`.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::handlers::state::MultiUserMemoryManager;
use crate::memory::oplog::{OpRecordDraft, ATTESTATION_WITNESSED};

/// Application state type alias (matches the rest of `handlers/`).
pub type AppState = Arc<MultiUserMemoryManager>;

/// Request extension enriched by handlers with body-derived identity and
/// evidence. The middleware owns the record's envelope (op, timing, outcome);
/// handlers own what only they can see (who asked, what memories were touched).
#[derive(Clone, Default, Debug)]
pub struct OpTrace(pub Arc<parking_lot::Mutex<OpTraceInner>>);

#[derive(Default, Debug)]
pub struct OpTraceInner {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub evidence_refs: Vec<String>,
    pub payload_summary: Option<String>,
}

impl OpTrace {
    /// Handler-side enrichment helper: set identity once, append evidence ids.
    pub fn set_identity(&self, user_id: &str, session_id: Option<&str>) {
        let mut inner = self.0.lock();
        inner.user_id = Some(user_id.to_string());
        if let Some(sid) = session_id {
            inner.session_id = Some(sid.to_string());
        }
    }

    pub fn push_evidence<I: IntoIterator<Item = String>>(&self, ids: I) {
        self.0.lock().evidence_refs.extend(ids);
    }

    pub fn set_summary(&self, summary: String) {
        self.0.lock().payload_summary = Some(summary);
    }
}

/// Path prefixes never captured. `/api/trace` covers the read API and the
/// reported-event ingest (Task 4) — capture observing capture would
/// self-amplify. `/metrics` would mint one permanent record per Prometheus
/// scrape, forever.
const EXCLUDED_PREFIXES: &[&str] = &["/api/trace", "/metrics"];

/// Long-lived streaming routes (SSE/WS) — connections, not discrete ops.
const EXCLUDED_EXACT: &[&str] = &[
    "/api/context/monitor",
    "/api/events/sse",
    "/api/events",
    "/api/stream",
];

/// Protected routes that carry NO user identity by construction (verified in
/// the slice-1 substrate audit): their requests have no `user_id` anywhere,
/// so a dropped record here is a known, named property — not a capture hole.
/// Task 5's completeness sweep consumes this list (audit amendment 18).
pub const NO_IDENTITY_OPS: &[&str] = &["/api/users", "/api/mif/adapters", "/api/ab"];

fn is_excluded(path: &str) -> bool {
    EXCLUDED_PREFIXES.iter().any(|p| path.starts_with(p))
        || EXCLUDED_EXACT.iter().any(|e| path == *e)
}

fn is_known_no_identity(path: &str) -> bool {
    NO_IDENTITY_OPS.iter().any(|p| path.starts_with(p))
}

/// Derive the op name from method + path: `POST /api/recall` → `recall`,
/// `GET /api/memories` → `get:memories`. Non-GET keeps the bare tail (agent
/// verbs are POST-shaped in this API); GET is prefixed to distinguish
/// read-surface ops from the verb of the same name.
fn op_name(method: &axum::http::Method, path: &str) -> String {
    let tail = path.strip_prefix("/api/").unwrap_or(path).replace('/', ":");
    if method == axum::http::Method::GET {
        format!("get:{tail}")
    } else {
        tail
    }
}

/// The capture choke point. Mounted inside `build_protected_routes` — see the
/// layer-order note at the mount site.
pub async fn capture_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if is_excluded(&path) {
        return next.run(req).await;
    }
    let method = req.method().clone();

    let trace = OpTrace::default();
    let mut req = req;
    req.extensions_mut().insert(trace.clone());

    let response = next.run(req).await;
    let status = response.status();

    // Post-handler: build and append the record. Everything below is
    // best-effort by contract — the response is returned unchanged no matter
    // what happens here.
    let inner = trace.0.lock();
    let Some(user_id) = inner.user_id.clone() else {
        // No identity: known-anonymous routes are an accepted drop; anything
        // else is a capture gap worth counting loudly (a handler forgot to
        // enrich, or a new identity-less route appeared unnamed).
        if !is_known_no_identity(&path) {
            crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL.inc();
            tracing::warn!(
                path = %path,
                "trace capture dropped: no identity set by handler (unnamed identity-less route?)"
            );
        }
        drop(inner);
        return response;
    };

    // Session fallback: the store the rest of the codebase already uses —
    // never an invented namespace (audit amendment 12).
    let session_id = match inner.session_id.clone() {
        Some(sid) => sid,
        None => state
            .session_store()
            .get_or_create_session(&user_id)
            .to_string(),
    };
    let evidence_refs = inner.evidence_refs.clone();
    let payload_summary = inner.payload_summary.clone().unwrap_or_default();
    drop(inner);

    // Cache-only lookup (audit amendment 11): the handler that just ran has
    // already materialized the user's MemorySystem on any op that touches
    // memory; if the user is not cached, we DROP rather than construct a
    // store from the capture path.
    let Some(system) = state.cached_user_memory(&user_id) else {
        crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL.inc();
        tracing::warn!(user_id = %user_id, path = %path, "trace capture dropped: user not cached");
        return response;
    };

    let draft = OpRecordDraft {
        ts: chrono::Utc::now(),
        session_id: session_id.clone(),
        user_id: user_id.clone(),
        op: op_name(&method, &path),
        attestation: ATTESTATION_WITNESSED.to_string(),
        payload_summary,
        evidence_refs,
        outcome: if status.is_success() {
            "ok".to_string()
        } else {
            format!("error:{}", status.as_u16())
        },
        reported_ts: None,
        source: None,
    };

    let append_result = { system.read().storage().oplog_append(draft) };
    if let Err(e) = append_result {
        crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL.inc();
        tracing::error!(user_id = %user_id, session_id = %session_id, error = %e,
            "oplog append failed — session trace marked incomplete");
        let _ = { system.read().storage().oplog_mark_incomplete(&session_id) };
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_name_shapes() {
        use axum::http::Method;
        assert_eq!(op_name(&Method::POST, "/api/recall"), "recall");
        assert_eq!(
            op_name(&Method::POST, "/api/remember/batch"),
            "remember:batch"
        );
        assert_eq!(op_name(&Method::GET, "/api/memories"), "get:memories");
    }

    #[test]
    fn exclusions_cover_amendment_10() {
        for p in ["/api/trace/report", "/api/trace/s1", "/metrics"] {
            assert!(is_excluded(p), "{p} must be excluded");
        }
        for p in EXCLUDED_EXACT {
            assert!(is_excluded(p), "{p} must be excluded");
        }
        assert!(!is_excluded("/api/recall"));
    }

    #[test]
    fn no_identity_ops_named() {
        assert!(is_known_no_identity("/api/users"));
        assert!(is_known_no_identity("/api/ab/summary"));
        assert!(!is_known_no_identity("/api/recall"));
    }
}
