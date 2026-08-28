//! C ABI FFI for PhoneBuddy SDK.
//!
//! Exposes the engine as a `cdylib`/`staticlib` for iOS and Android.
//! The interface is intentionally a thin, stable C surface:
//! - opaque engine handle;
//! - configuration and events passed as JSON strings (host parses them);
//! - streaming delivered through a single C callback receiving event JSON.
//!
//! This keeps the ABI robust across Swift / Objective-C++ / Kotlin (JNI) /
//! React-Native native modules without binding-codegen.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use phone_buddy::engine::PhoneBuddyEngine;
use phone_buddy::events::{AgentEvent, AgentObserver};
use phone_buddy::llm::router::{LlmRoutingConfig, MAIN_POOL_ID};
use phone_buddy::runtime::{GenerateTextRequest, GenerateTextResult, PhoneBuddyRuntime};

/// Streaming event callback.
///
/// `event_json` is a UTF-8 JSON string describing one [`AgentEvent`].
/// It is valid only for the duration of the call; copy it if you keep it.
/// The callback may be invoked from a background engine thread.
pub type PbEventCallback =
    Option<unsafe extern "C" fn(event_json: *const c_char, user_data: *mut c_void)>;

/// Host LLM request callback.
///
/// Fired when the engine needs a completion. `request_id` and `request_json`
/// are valid only for the duration of the call. Respond with
/// [`pb_engine_llm_push_chunk`] / [`pb_engine_llm_finish`] /
/// [`pb_engine_llm_fail`].
pub type PbLlmRequestCallback = Option<
    unsafe extern "C" fn(
        request_id: *const c_char,
        request_json: *const c_char,
        user_data: *mut c_void,
    ),
>;

/// Host tool request callback.
///
/// Fired when a host-registered tool must run. Respond with
/// [`pb_engine_host_tool_result`].
pub type PbHostToolCallback = Option<
    unsafe extern "C" fn(
        call_id: *const c_char,
        name: *const c_char,
        arguments_json: *const c_char,
        user_data: *mut c_void,
    ),
>;

/// Host system WebView fetch callback.
///
/// Fired when `web_search` wants DuckDuckGo Lite loaded in WKWebView /
/// Android WebView. `request_json` is a [`phone_buddy::tools::WebViewFetchRequest`].
/// Respond with [`pb_engine_webview_result`]. Desktop / C hosts should leave
/// this unset so search skips scraping and uses the LLM API.
pub type PbWebViewFetchCallback = Option<
    unsafe extern "C" fn(
        call_id: *const c_char,
        request_json: *const c_char,
        user_data: *mut c_void,
    ),
>;

/// Native logging callback.
/// `level`: 1 = ERROR, 2 = WARN, 3 = INFO, 4 = DEBUG, 5 = TRACE.
/// `target`: Logger target / module name (C-string).
/// `message`: Formatted log text (C-string).
pub type PbLogCallback =
    Option<unsafe extern "C" fn(level: i32, target: *const c_char, message: *const c_char)>;

static GLOBAL_LOG_CALLBACK: std::sync::RwLock<PbLogCallback> = std::sync::RwLock::new(None);
static LOG_INIT: std::sync::Once = std::sync::Once::new();

struct HostLogLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for HostLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let cb_opt = if let Ok(guard) = GLOBAL_LOG_CALLBACK.read() {
            *guard
        } else {
            None
        };
        let Some(cb) = cb_opt else {
            return;
        };

        let meta = event.metadata();
        let level: i32 = match *meta.level() {
            tracing::Level::ERROR => 1,
            tracing::Level::WARN => 2,
            tracing::Level::INFO => 3,
            tracing::Level::DEBUG => 4,
            tracing::Level::TRACE => 5,
        };

        struct Visitor(String);
        impl tracing::field::Visit for Visitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    use std::fmt::Write;
                    let _ = write!(self.0, "{:?}", value);
                } else {
                    if !self.0.is_empty() {
                        self.0.push(' ');
                    }
                    use std::fmt::Write;
                    let _ = write!(self.0, "{}={:?}", field.name(), value);
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.0.push_str(value);
                } else {
                    if !self.0.is_empty() {
                        self.0.push(' ');
                    }
                    self.0.push_str(field.name());
                    self.0.push('=');
                    self.0.push_str(value);
                }
            }
        }

        let mut visitor = Visitor(String::new());
        event.record(&mut visitor);

        let Ok(target_c) = CString::new(meta.target()) else {
            return;
        };
        let Ok(msg_c) = CString::new(visitor.0) else {
            return;
        };

        unsafe {
            cb(level, target_c.as_ptr(), msg_c.as_ptr());
        }
    }
}

/// Opaque engine handle.
pub struct PbEngine {
    inner: Arc<PhoneBuddyEngine>,
    /// Kept so C callbacks stay valid while the engine lives.
    host_user_data: *mut c_void,
    webview_user_data: *mut c_void,
}

/// Opaque long-lived routing runtime handle.
pub struct PbRuntime {
    inner: Arc<PhoneBuddyRuntime>,
}

/// One-shot `generate_text` completion callback.
///
/// `envelope_json` is a versioned JSON object valid only for the duration of
/// the call. Copy it if you keep it. Invoked from a background worker thread.
/// Do not reuse chat session event callbacks for this.
pub type PbOperationCallback =
    Option<unsafe extern "C" fn(envelope_json: *const c_char, user_data: *mut c_void)>;

// user_data is an opaque host pointer; the host owns lifetime/thread safety.
unsafe impl Send for PbEngine {}
unsafe impl Sync for PbEngine {}

/// Adapter that forwards [`AgentEvent`]s to the C callback.
struct FfiObserver {
    callback: PbEventCallback,
    user_data: *mut c_void,
}

// The host guarantees the callback + user_data are safe to use from any
// thread (events fire on the engine runtime thread).
unsafe impl Send for FfiObserver {}
unsafe impl Sync for FfiObserver {}

impl AgentObserver for FfiObserver {
    fn on_event(&self, event: AgentEvent) {
        let Some(cb) = self.callback else {
            return;
        };
        let Ok(json) = serde_json::to_string(&event) else {
            return;
        };
        let Ok(cstr) = CString::new(json) else {
            return;
        };
        unsafe { cb(cstr.as_ptr(), self.user_data) };
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

unsafe fn str_from<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

fn to_cstring(s: String) -> *mut c_char {
    CString::new(s)
        .map(|c| c.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

unsafe fn set_err(err_out: *mut *mut c_char, msg: String) {
    if !err_out.is_null() {
        *err_out = to_cstring(msg);
    }
}

fn catch_ptr<T, F: FnOnce() -> *mut T>(f: F) -> *mut T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(std::ptr::null_mut())
}

fn catch_i32<F: FnOnce() -> i32>(f: F) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(-100)
}

fn catch_void<F: FnOnce()>(f: F) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

fn generate_ok_envelope(operation_id: &str, result: &GenerateTextResult) -> String {
    serde_json::json!({
        "version": 1,
        "ok": true,
        "operation_id": operation_id,
        "result": result,
    })
    .to_string()
}

fn generate_err_envelope(operation_id: &str, err: &phone_buddy::error::EngineError) -> String {
    let mut error = serde_json::json!({
        "kind": err.kind(),
        "message": err.to_string(),
    });
    if let serde_json::Value::Object(extra) = err.envelope_fields() {
        if let Some(map) = error.as_object_mut() {
            for (k, v) in extra {
                map.insert(k, v);
            }
        }
    }
    serde_json::json!({
        "version": 1,
        "ok": false,
        "operation_id": operation_id,
        "error": error,
    })
    .to_string()
}

fn invoke_operation_callback(
    callback: PbOperationCallback,
    user_data: *mut c_void,
    envelope: String,
) {
    let Some(cb) = callback else {
        return;
    };
    let Ok(cstr) = CString::new(envelope) else {
        return;
    };
    unsafe { cb(cstr.as_ptr(), user_data) };
}

// ── exported API ─────────────────────────────────────────────────────────

/// Library version string. Do not free.
#[no_mangle]
pub extern "C" fn pb_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    let v = VERSION.get_or_init(|| CString::new(phone_buddy::VERSION).unwrap());
    v.as_ptr()
}

/// Create an engine from a JSON configuration.
///
/// `config_json` must match [`phone_buddy::config::EngineConfig`].
/// On success returns a handle and writes null to `err_out`; on failure
/// returns null and writes an error message to `err_out` (caller frees with
/// [`pb_string_free`]).
///
/// # Safety
/// `config_json` must be a valid C string. `err_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_new(
    config_json: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut PbEngine {
    let cfg_str = match str_from(config_json) {
        Some(s) => s,
        None => {
            set_err(err_out, "config_json is null or invalid UTF-8".into());
            return std::ptr::null_mut();
        }
    };
    let cfg: phone_buddy::config::EngineConfig = match serde_json::from_str(cfg_str) {
        Ok(c) => c,
        Err(e) => {
            set_err(err_out, format!("invalid config JSON: {e}"));
            return std::ptr::null_mut();
        }
    };
    match PhoneBuddyEngine::new(cfg) {
        Ok(engine) => {
            // Success: write null to err_out as documented.
            if !err_out.is_null() {
                *err_out = std::ptr::null_mut();
            }
            Box::into_raw(Box::new(PbEngine {
                inner: engine,
                host_user_data: std::ptr::null_mut(),
                webview_user_data: std::ptr::null_mut(),
            }))
        }
        Err(e) => {
            set_err(err_out, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Free an engine handle.
///
/// # Safety
/// `engine` must be a handle returned by [`pb_engine_new`], or null.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_free(engine: *mut PbEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Create a long-lived routing runtime from a JSON [`LlmRoutingConfig`].
///
/// # Safety
/// `routing_config_json` and `root_dir` must be valid C strings. `err_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn pb_runtime_new(
    routing_config_json: *const c_char,
    root_dir: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut PbRuntime {
    catch_ptr(|| {
        let cfg_str = match str_from(routing_config_json) {
            Some(s) => s,
            None => {
                set_err(
                    err_out,
                    "routing_config_json is null or invalid UTF-8".into(),
                );
                return std::ptr::null_mut();
            }
        };
        let root = match str_from(root_dir) {
            Some(s) => s,
            None => {
                set_err(err_out, "root_dir is null or invalid UTF-8".into());
                return std::ptr::null_mut();
            }
        };
        let routing: LlmRoutingConfig = match serde_json::from_str(cfg_str) {
            Ok(c) => c,
            Err(e) => {
                set_err(err_out, format!("invalid routing config JSON: {e}"));
                return std::ptr::null_mut();
            }
        };
        match PhoneBuddyRuntime::new(routing, root) {
            Ok(runtime) => {
                if !err_out.is_null() {
                    *err_out = std::ptr::null_mut();
                }
                Box::into_raw(Box::new(PbRuntime { inner: runtime }))
            }
            Err(e) => {
                set_err(err_out, e.to_string());
                std::ptr::null_mut()
            }
        }
    })
}

/// Replace routing on an existing runtime. In-flight operations may finish
/// on a previously captured snapshot.
///
/// Returns 0 on success.
///
/// # Safety
/// `runtime` must be a handle from [`pb_runtime_new`]. `routing_config_json` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn pb_runtime_update_routing(
    runtime: *mut PbRuntime,
    routing_config_json: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    catch_i32(|| {
        let Some(runtime) = runtime.as_ref() else {
            set_err(err_out, "runtime is null".into());
            return -1;
        };
        let Some(cfg_str) = str_from(routing_config_json) else {
            set_err(
                err_out,
                "routing_config_json is null or invalid UTF-8".into(),
            );
            return -2;
        };
        let routing: LlmRoutingConfig = match serde_json::from_str(cfg_str) {
            Ok(c) => c,
            Err(e) => {
                set_err(err_out, format!("invalid routing config JSON: {e}"));
                return -3;
            }
        };
        match runtime.inner.update_routing(routing) {
            Ok(()) => 0,
            Err(e) => {
                set_err(err_out, e.to_string());
                -4
            }
        }
    })
}

/// Create an engine bound to `runtime`. `main_pool_id` defaults to `"main"` when null.
///
/// # Safety
/// `runtime` must outlive the returned engine. `agent_config_json` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_new_with_runtime(
    runtime: *mut PbRuntime,
    agent_config_json: *const c_char,
    main_pool_id: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut PbEngine {
    catch_ptr(|| {
        let Some(runtime) = runtime.as_ref() else {
            set_err(err_out, "runtime is null".into());
            return std::ptr::null_mut();
        };
        let Some(cfg_str) = str_from(agent_config_json) else {
            set_err(err_out, "agent_config_json is null or invalid UTF-8".into());
            return std::ptr::null_mut();
        };
        let cfg: phone_buddy::config::EngineConfig = match serde_json::from_str(cfg_str) {
            Ok(c) => c,
            Err(e) => {
                set_err(err_out, format!("invalid config JSON: {e}"));
                return std::ptr::null_mut();
            }
        };
        let pool = str_from(main_pool_id)
            .filter(|s| !s.is_empty())
            .unwrap_or(MAIN_POOL_ID);
        match runtime.inner.create_engine(cfg, pool) {
            Ok(engine) => {
                if !err_out.is_null() {
                    *err_out = std::ptr::null_mut();
                }
                Box::into_raw(Box::new(PbEngine {
                    inner: engine,
                    host_user_data: std::ptr::null_mut(),
                    webview_user_data: std::ptr::null_mut(),
                }))
            }
            Err(e) => {
                set_err(err_out, e.to_string());
                std::ptr::null_mut()
            }
        }
    })
}

/// Start tool-free one-shot generation. Returns `operation_id` immediately.
/// Completion is delivered through `callback` as a versioned JSON envelope.
///
/// # Safety
/// `runtime` and `request_json` must be valid. `callback` may be null (result discarded).
/// The callback and `user_data` must remain valid until the envelope is delivered.
#[no_mangle]
pub unsafe extern "C" fn pb_runtime_generate_text_async(
    runtime: *mut PbRuntime,
    request_json: *const c_char,
    callback: PbOperationCallback,
    user_data: *mut c_void,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    catch_ptr(|| {
        let Some(runtime) = runtime.as_ref() else {
            set_err(err_out, "runtime is null".into());
            return std::ptr::null_mut();
        };
        let Some(req_str) = str_from(request_json) else {
            set_err(err_out, "request_json is null or invalid UTF-8".into());
            return std::ptr::null_mut();
        };
        let request: GenerateTextRequest = match serde_json::from_str(req_str) {
            Ok(r) => r,
            Err(e) => {
                set_err(err_out, format!("invalid generate_text request JSON: {e}"));
                return std::ptr::null_mut();
            }
        };
        let ud = user_data as usize;
        match runtime
            .inner
            .generate_text_async(request, move |op, result| {
                let envelope = match result {
                    Ok(ok) => generate_ok_envelope(&op, &ok),
                    Err(e) => generate_err_envelope(&op, &e),
                };
                invoke_operation_callback(callback, ud as *mut c_void, envelope);
            }) {
            Ok(op) => {
                if !err_out.is_null() {
                    *err_out = std::ptr::null_mut();
                }
                to_cstring(op)
            }
            Err(e) => {
                set_err(err_out, e.to_string());
                std::ptr::null_mut()
            }
        }
    })
}

/// Cancel a one-shot operation started by [`pb_runtime_generate_text_async`].
///
/// # Safety
/// `runtime` and `operation_id` may be null (no-op).
#[no_mangle]
pub unsafe extern "C" fn pb_runtime_cancel_operation(
    runtime: *mut PbRuntime,
    operation_id: *const c_char,
) {
    catch_void(|| {
        let Some(runtime) = runtime.as_ref() else {
            return;
        };
        let Some(op) = str_from(operation_id) else {
            return;
        };
        runtime.inner.cancel_operation(op);
    });
}

/// Free a runtime handle. Outstanding one-shot operations are cancelled.
///
/// # Safety
/// `runtime` must be a handle returned by [`pb_runtime_new`], or null.
#[no_mangle]
pub unsafe extern "C" fn pb_runtime_free(runtime: *mut PbRuntime) {
    catch_void(|| {
        if !runtime.is_null() {
            let boxed = Box::from_raw(runtime);
            boxed.inner.cancel_all();
            drop(boxed);
        }
    });
}

/// Run one chat turn to completion (blocking). Call from a background
/// thread; streaming events are delivered via `callback` as they happen.
///
/// Returns a JSON object `{ "final_text", "turns_used", "usage", "plan" }`
/// on success, or null on error (with `err_out` set). Caller frees the
/// result with [`pb_string_free`].
///
/// # Safety
/// All pointers must be valid. `callback` may be null (events discarded).
#[no_mangle]
pub unsafe extern "C" fn pb_engine_chat(
    engine: *mut PbEngine,
    session_id: *const c_char,
    user_input: *const c_char,
    callback: PbEventCallback,
    user_data: *mut c_void,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let engine = match engine.as_ref() {
        Some(e) => e,
        None => {
            set_err(err_out, "engine is null".into());
            return std::ptr::null_mut();
        }
    };
    let session_id = match str_from(session_id) {
        Some(s) => s.to_string(),
        None => {
            set_err(err_out, "session_id is null or invalid".into());
            return std::ptr::null_mut();
        }
    };
    let user_input = match str_from(user_input) {
        Some(s) => s.to_string(),
        None => {
            set_err(err_out, "user_input is null or invalid".into());
            return std::ptr::null_mut();
        }
    };

    let observer: Option<Arc<dyn AgentObserver>> = if callback.is_some() {
        Some(Arc::new(FfiObserver {
            callback,
            user_data,
        }))
    } else {
        None
    };

    match engine.inner.chat(&session_id, &user_input, observer) {
        Ok(outcome) => outcome_json(&outcome),
        Err(e) => {
            set_err(err_out, e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn outcome_json(outcome: &phone_buddy::engine::ChatOutcome) -> *mut c_char {
    let plan: serde_json::Value =
        serde_json::from_str(&outcome.plan_items_json).unwrap_or(serde_json::json!([]));
    let result = serde_json::json!({
        "final_text": outcome.final_text,
        "turns_used": outcome.turns_used,
        "usage": outcome.usage,
        "plan": plan,
    });
    to_cstring(result.to_string())
}

/// Run one versioned structured user turn (`schema_version: 1`).
/// Invalid JSON, unsupported schema versions, missing attachments, or invalid
/// content parts are returned through `err_out`. There is no text fallback.
///
/// # Safety
/// All pointers must be valid. `callback` may be null (events discarded).
#[no_mangle]
pub unsafe extern "C" fn pb_engine_chat_v2(
    engine: *mut PbEngine,
    session_id: *const c_char,
    turn_json: *const c_char,
    callback: PbEventCallback,
    user_data: *mut c_void,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let engine = match engine.as_ref() {
        Some(e) => e,
        None => {
            set_err(err_out, "engine is null".into());
            return std::ptr::null_mut();
        }
    };
    let session_id = match str_from(session_id) {
        Some(s) => s.to_string(),
        None => {
            set_err(err_out, "session_id is null or invalid".into());
            return std::ptr::null_mut();
        }
    };
    let turn_json = match str_from(turn_json) {
        Some(s) => s.to_string(),
        None => {
            set_err(err_out, "turn_json is null or invalid".into());
            return std::ptr::null_mut();
        }
    };

    let observer: Option<Arc<dyn AgentObserver>> = if callback.is_some() {
        Some(Arc::new(FfiObserver {
            callback,
            user_data,
        }))
    } else {
        None
    };

    match engine.inner.chat_v2(&session_id, &turn_json, observer) {
        Ok(outcome) => outcome_json(&outcome),
        Err(e) => {
            set_err(err_out, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// List sessions as a JSON array of metadata objects. Caller frees.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_list_sessions(
    engine: *mut PbEngine,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let engine = match engine.as_ref() {
        Some(e) => e,
        None => {
            set_err(err_out, "engine is null".into());
            return std::ptr::null_mut();
        }
    };
    match engine.inner.list_sessions() {
        Ok(list) => to_cstring(serde_json::to_string(&list).unwrap_or_else(|_| "[]".into())),
        Err(e) => {
            set_err(err_out, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Get full session JSON (including all messages and reasoning) for a given session ID.
///
/// Returns a JSON string matching [`phone_buddy::session::StoredSession`] on success,
/// or null if the session does not exist (with `err_out` set to null), or null on
/// error (with `err_out` set to an error message). Caller frees the returned JSON string
/// with [`pb_string_free`].
///
/// # Safety
/// `engine` and `session_id` must be valid. `err_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_get_session(
    engine: *mut PbEngine,
    session_id: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut c_char {
    let engine = match engine.as_ref() {
        Some(e) => e,
        None => {
            set_err(err_out, "engine is null".into());
            return std::ptr::null_mut();
        }
    };
    let session_id = match str_from(session_id) {
        Some(s) => s,
        None => {
            set_err(err_out, "session_id is null or invalid".into());
            return std::ptr::null_mut();
        }
    };
    match engine.inner.get_session(session_id) {
        Ok(Some(session)) => {
            if !err_out.is_null() {
                *err_out = std::ptr::null_mut();
            }
            match serde_json::to_string(&session) {
                Ok(json) => to_cstring(json),
                Err(e) => {
                    set_err(err_out, format!("failed to serialize session: {e}"));
                    std::ptr::null_mut()
                }
            }
        }
        Ok(None) => {
            if !err_out.is_null() {
                *err_out = std::ptr::null_mut();
            }
            std::ptr::null_mut()
        }
        Err(e) => {
            set_err(err_out, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Delete a session. Returns 0 on success, non-zero on error.
///
/// # Safety
/// `engine` and `session_id` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_delete_session(
    engine: *mut PbEngine,
    session_id: *const c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        return -1;
    };
    let Some(session_id) = str_from(session_id) else {
        return -2;
    };
    match engine.inner.delete_session(session_id) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Cancel an in-flight turn for a session.
///
/// # Safety
/// `engine` and `session_id` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_cancel(engine: *mut PbEngine, session_id: *const c_char) {
    if let Some(engine) = engine.as_ref() {
        if let Some(session_id) = str_from(session_id) {
            engine.inner.cancel(session_id);
        }
    }
}

/// Register host LLM + host-tool request callbacks.
///
/// Pass null callbacks to clear. `user_data` is passed through to both
/// callbacks and must remain valid until the engine is freed or callbacks
/// are cleared.
///
/// # Safety
/// Callbacks may be invoked from engine worker threads.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_set_host_callbacks(
    engine: *mut PbEngine,
    llm_cb: PbLlmRequestCallback,
    tool_cb: PbHostToolCallback,
    user_data: *mut c_void,
) {
    let Some(engine) = engine.as_mut() else {
        return;
    };
    engine.host_user_data = user_data;

    let llm_notify = llm_cb.map(|cb| {
        let ud = user_data as usize;
        std::sync::Arc::new(move |request_id: String, request_json: String| {
            let Ok(id_c) = CString::new(request_id) else {
                return;
            };
            let Ok(json_c) = CString::new(request_json) else {
                return;
            };
            unsafe {
                cb(id_c.as_ptr(), json_c.as_ptr(), ud as *mut c_void);
            }
        }) as phone_buddy::llm::HostLlmNotify
    });

    let tool_notify = tool_cb.map(|cb| {
        let ud = user_data as usize;
        std::sync::Arc::new(move |call_id: String, name: String, args: String| {
            let Ok(id_c) = CString::new(call_id) else {
                return;
            };
            let Ok(name_c) = CString::new(name) else {
                return;
            };
            let Ok(args_c) = CString::new(args) else {
                return;
            };
            unsafe {
                cb(
                    id_c.as_ptr(),
                    name_c.as_ptr(),
                    args_c.as_ptr(),
                    ud as *mut c_void,
                );
            }
        }) as phone_buddy::tools::HostToolNotify
    });

    engine.inner.set_host_callbacks(llm_notify, tool_notify);
}

/// Push one OpenAI-compatible chat-completion chunk for a host LLM request.
///
/// Returns 0 on success, non-zero on error (`err_out` set when provided).
#[no_mangle]
pub unsafe extern "C" fn pb_engine_llm_push_chunk(
    engine: *mut PbEngine,
    request_id: *const c_char,
    chunk_json: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(request_id) = str_from(request_id) else {
        set_err(err_out, "request_id is null or invalid".into());
        return -2;
    };
    let Some(chunk_json) = str_from(chunk_json) else {
        set_err(err_out, "chunk_json is null or invalid".into());
        return -3;
    };
    let chunk: phone_buddy::llm::ChatCompletionChunk = match serde_json::from_str(chunk_json) {
        Ok(c) => c,
        Err(e) => {
            set_err(err_out, format!("invalid chunk JSON: {e}"));
            return -4;
        }
    };
    match engine.inner.host_llm().push_chunk(request_id, chunk) {
        Ok(()) => 0,
        Err(e) => {
            set_err(err_out, e);
            -5
        }
    }
}

/// Finish a host LLM stream successfully.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_llm_finish(
    engine: *mut PbEngine,
    request_id: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(request_id) = str_from(request_id) else {
        set_err(err_out, "request_id is null or invalid".into());
        return -2;
    };
    match engine.inner.host_llm().finish(request_id) {
        Ok(()) => 0,
        Err(e) => {
            set_err(err_out, e);
            -3
        }
    }
}

/// Fail a host LLM stream.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_llm_fail(
    engine: *mut PbEngine,
    request_id: *const c_char,
    error_msg: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(request_id) = str_from(request_id) else {
        set_err(err_out, "request_id is null or invalid".into());
        return -2;
    };
    let msg = str_from(error_msg).unwrap_or("host LLM failed");
    match engine.inner.host_llm().fail(request_id, msg) {
        Ok(()) => 0,
        Err(e) => {
            set_err(err_out, e);
            -3
        }
    }
}

/// Register host tools from an OpenAI tools JSON array.
///
/// Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_set_host_tools(
    engine: *mut PbEngine,
    tools_json: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(tools_json) = str_from(tools_json) else {
        set_err(err_out, "tools_json is null or invalid".into());
        return -2;
    };
    match phone_buddy::tools::host::HostToolHub::parse_tool_defs(tools_json) {
        Ok(specs) => {
            engine.inner.set_host_tools(specs);
            0
        }
        Err(e) => {
            set_err(err_out, e);
            -3
        }
    }
}

/// Complete a host tool call. `ok` is non-zero for success.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_host_tool_result(
    engine: *mut PbEngine,
    call_id: *const c_char,
    ok: i32,
    output: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(call_id) = str_from(call_id) else {
        set_err(err_out, "call_id is null or invalid".into());
        return -2;
    };
    let output = str_from(output).unwrap_or("");
    match engine.inner.host_tools().complete(call_id, ok != 0, output) {
        Ok(()) => 0,
        Err(e) => {
            set_err(err_out, e);
            -3
        }
    }
}

/// Register a host system WebView fetch callback.
///
/// iOS / Android wrappers should set this to drive a hidden WKWebView or
/// Android WebView. Pass a null callback to clear (desktop / C hosts).
/// `user_data` must remain valid until the engine is freed or the callback
/// is cleared.
///
/// # Safety
/// The callback may be invoked from an engine worker thread. The host must
/// hop to the UI thread before creating or touching a WebView.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_set_webview_callback(
    engine: *mut PbEngine,
    callback: PbWebViewFetchCallback,
    user_data: *mut c_void,
) {
    let Some(engine) = engine.as_mut() else {
        return;
    };
    engine.webview_user_data = user_data;

    let notify = callback.map(|cb| {
        let ud = user_data as usize;
        std::sync::Arc::new(move |call_id: String, request_json: String| {
            let Ok(id_c) = CString::new(call_id) else {
                return;
            };
            let Ok(json_c) = CString::new(request_json) else {
                return;
            };
            unsafe {
                cb(id_c.as_ptr(), json_c.as_ptr(), ud as *mut c_void);
            }
        }) as phone_buddy::tools::WebViewFetchNotify
    });

    engine.inner.set_webview_callback(notify);
}

/// Complete a host WebView fetch. `ok` is non-zero for success; `output` is
/// the document HTML on success or an error message on failure.
///
/// # Safety
/// `engine` and `call_id` must be valid. `output` may be null (treated as empty).
#[no_mangle]
pub unsafe extern "C" fn pb_engine_webview_result(
    engine: *mut PbEngine,
    call_id: *const c_char,
    ok: i32,
    output: *const c_char,
    err_out: *mut *mut c_char,
) -> i32 {
    let Some(engine) = engine.as_ref() else {
        set_err(err_out, "engine is null".into());
        return -1;
    };
    let Some(call_id) = str_from(call_id) else {
        set_err(err_out, "call_id is null or invalid".into());
        return -2;
    };
    let output = str_from(output).unwrap_or("");
    match engine
        .inner
        .webview_host()
        .complete(call_id, ok != 0, output)
    {
        Ok(()) => 0,
        Err(e) => {
            set_err(err_out, e);
            -3
        }
    }
}

/// Set or clear the system prompt extra text (Pal persona).
///
/// Pass null `extra` to clear.
#[no_mangle]
pub unsafe extern "C" fn pb_engine_set_system_prompt_extra(
    engine: *mut PbEngine,
    extra: *const c_char,
) {
    let Some(engine) = engine.as_ref() else {
        return;
    };
    let value = if extra.is_null() {
        None
    } else {
        str_from(extra).map(|s| s.to_string())
    };
    engine.inner.set_system_prompt_extra(value);
}

/// Set the system-prompt identity (`You are {name}…`).
///
/// Pass null or empty `name` to reset to the default (`PhoneBuddy`).
#[no_mangle]
pub unsafe extern "C" fn pb_engine_set_agent_name(engine: *mut PbEngine, name: *const c_char) {
    let Some(engine) = engine.as_ref() else {
        return;
    };
    let value = if name.is_null() {
        None
    } else {
        str_from(name).map(|s| s.to_string())
    };
    engine.inner.set_agent_name(value);
}

/// Free a string returned by this library.
///
/// # Safety
/// `ptr` must be a string returned by this library, or null.
#[no_mangle]
pub unsafe extern "C" fn pb_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Initialize native logging for debugging.
/// `min_level`: 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE, <=0=disabled.
/// On Android, automatically attaches logcat layer alongside the host callback.
/// In release builds (`release_max_level_off`), tracing macros are stripped at compile-time.
#[no_mangle]
pub unsafe extern "C" fn pb_init_logging(callback: PbLogCallback, min_level: i32) {
    if let Ok(mut guard) = GLOBAL_LOG_CALLBACK.write() {
        *guard = callback;
    }

    if min_level <= 0 {
        return;
    }

    LOG_INIT.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let filter = match min_level {
            1 => tracing_subscriber::filter::LevelFilter::ERROR,
            2 => tracing_subscriber::filter::LevelFilter::WARN,
            3 => tracing_subscriber::filter::LevelFilter::INFO,
            4 => tracing_subscriber::filter::LevelFilter::DEBUG,
            _ => tracing_subscriber::filter::LevelFilter::TRACE,
        };

        #[cfg(target_os = "android")]
        {
            if let Ok(android_layer) = tracing_android::layer("PhoneBuddy") {
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(android_layer)
                    .with(HostLogLayer)
                    .try_init();
                return;
            }
        }

        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(HostLogLayer)
            .try_init();
    });
}

#[cfg(target_os = "android")]
#[doc(hidden)]
pub mod android_jni {
    use std::ffi::c_void;

    extern "C" {
        fn pb_jni_on_load(vm: *mut c_void, reserved: *mut c_void) -> i32;
        fn pb_jni_nativeNewEngine(
            env: *mut c_void,
            clazz: *mut c_void,
            config_json: *mut c_void,
        ) -> i64;
        fn pb_jni_nativeFreeEngine(env: *mut c_void, clazz: *mut c_void, engine_ptr: i64);
        fn pb_jni_nativeChat(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            session_id: *mut c_void,
            user_input: *mut c_void,
            listener: *mut c_void,
        ) -> *mut c_void;
        fn pb_jni_nativeChatV2(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            session_id: *mut c_void,
            turn_json: *mut c_void,
            listener: *mut c_void,
        ) -> *mut c_void;
        fn pb_jni_nativeGetSession(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            session_id: *mut c_void,
        ) -> *mut c_void;
        fn pb_jni_nativeListSessions(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
        ) -> *mut c_void;
        fn pb_jni_nativeDeleteSession(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            session_id: *mut c_void,
        ) -> i32;
        fn pb_jni_nativeSetWebViewCallback(env: *mut c_void, clazz: *mut c_void, engine_ptr: i64);
        fn pb_jni_nativeClearWebViewCallback(env: *mut c_void, clazz: *mut c_void, engine_ptr: i64);
        fn pb_jni_nativeCancel(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            session_id: *mut c_void,
        );
        fn pb_jni_nativeSetHostCallbacks(env: *mut c_void, clazz: *mut c_void, engine_ptr: i64);
        fn pb_jni_nativeHostToolResult(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            call_id: *mut c_void,
            ok: i32,
            output: *mut c_void,
        ) -> i32;
        fn pb_jni_nativeWebViewResult(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            call_id: *mut c_void,
            ok: i32,
            output: *mut c_void,
        ) -> i32;
        fn pb_jni_nativeSetAgentName(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            name: *mut c_void,
        );
        fn pb_jni_nativeSetSystemPromptExtra(
            env: *mut c_void,
            clazz: *mut c_void,
            engine_ptr: i64,
            extra: *mut c_void,
        );
        fn pb_jni_nativeNewRuntime(
            env: *mut c_void,
            clazz: *mut c_void,
            routing_json: *mut c_void,
            root_dir: *mut c_void,
        ) -> i64;
        fn pb_jni_nativeFreeRuntime(env: *mut c_void, clazz: *mut c_void, runtime_ptr: i64);
        fn pb_jni_nativeUpdateRouting(
            env: *mut c_void,
            clazz: *mut c_void,
            runtime_ptr: i64,
            routing_json: *mut c_void,
        );
        fn pb_jni_nativeCreateEngine(
            env: *mut c_void,
            clazz: *mut c_void,
            runtime_ptr: i64,
            config_json: *mut c_void,
            main_pool_id: *mut c_void,
        ) -> i64;
        fn pb_jni_nativeGenerateTextAsync(
            env: *mut c_void,
            clazz: *mut c_void,
            runtime_ptr: i64,
            request_json: *mut c_void,
            listener: *mut c_void,
        ) -> *mut c_void;
        fn pb_jni_nativeCancelOperation(
            env: *mut c_void,
            clazz: *mut c_void,
            runtime_ptr: i64,
            operation_id: *mut c_void,
        );
    }

    #[no_mangle]
    pub unsafe extern "C" fn JNI_OnLoad(vm: *mut c_void, reserved: *mut c_void) -> i32 {
        pb_jni_on_load(vm, reserved)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeNewEngine(
        env: *mut c_void,
        clazz: *mut c_void,
        config_json: *mut c_void,
    ) -> i64 {
        pb_jni_nativeNewEngine(env, clazz, config_json)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeFreeEngine(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
    ) {
        pb_jni_nativeFreeEngine(env, clazz, engine_ptr);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeChat(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        session_id: *mut c_void,
        user_input: *mut c_void,
        listener: *mut c_void,
    ) -> *mut c_void {
        pb_jni_nativeChat(env, clazz, engine_ptr, session_id, user_input, listener)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeChatV2(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        session_id: *mut c_void,
        turn_json: *mut c_void,
        listener: *mut c_void,
    ) -> *mut c_void {
        pb_jni_nativeChatV2(env, clazz, engine_ptr, session_id, turn_json, listener)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeGetSession(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        session_id: *mut c_void,
    ) -> *mut c_void {
        pb_jni_nativeGetSession(env, clazz, engine_ptr, session_id)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeListSessions(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
    ) -> *mut c_void {
        pb_jni_nativeListSessions(env, clazz, engine_ptr)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeDeleteSession(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        session_id: *mut c_void,
    ) -> i32 {
        pb_jni_nativeDeleteSession(env, clazz, engine_ptr, session_id)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeSetWebViewCallback(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
    ) {
        pb_jni_nativeSetWebViewCallback(env, clazz, engine_ptr);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeClearWebViewCallback(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
    ) {
        pb_jni_nativeClearWebViewCallback(env, clazz, engine_ptr);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeCancel(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        session_id: *mut c_void,
    ) {
        pb_jni_nativeCancel(env, clazz, engine_ptr, session_id);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeSetHostCallbacks(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
    ) {
        pb_jni_nativeSetHostCallbacks(env, clazz, engine_ptr);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeHostToolResult(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        call_id: *mut c_void,
        ok: i32,
        output: *mut c_void,
    ) -> i32 {
        pb_jni_nativeHostToolResult(env, clazz, engine_ptr, call_id, ok, output)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeWebViewResult(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        call_id: *mut c_void,
        ok: i32,
        output: *mut c_void,
    ) -> i32 {
        pb_jni_nativeWebViewResult(env, clazz, engine_ptr, call_id, ok, output)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeSetAgentName(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        name: *mut c_void,
    ) {
        pb_jni_nativeSetAgentName(env, clazz, engine_ptr, name);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeAgent_nativeSetSystemPromptExtra(
        env: *mut c_void,
        clazz: *mut c_void,
        engine_ptr: i64,
        extra: *mut c_void,
    ) {
        pb_jni_nativeSetSystemPromptExtra(env, clazz, engine_ptr, extra);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeNew(
        env: *mut c_void,
        clazz: *mut c_void,
        routing_json: *mut c_void,
        root_dir: *mut c_void,
    ) -> i64 {
        pb_jni_nativeNewRuntime(env, clazz, routing_json, root_dir)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeFree(
        env: *mut c_void,
        clazz: *mut c_void,
        runtime_ptr: i64,
    ) {
        pb_jni_nativeFreeRuntime(env, clazz, runtime_ptr);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeUpdateRouting(
        env: *mut c_void,
        clazz: *mut c_void,
        runtime_ptr: i64,
        routing_json: *mut c_void,
    ) {
        pb_jni_nativeUpdateRouting(env, clazz, runtime_ptr, routing_json);
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeCreateEngine(
        env: *mut c_void,
        clazz: *mut c_void,
        runtime_ptr: i64,
        config_json: *mut c_void,
        main_pool_id: *mut c_void,
    ) -> i64 {
        pb_jni_nativeCreateEngine(env, clazz, runtime_ptr, config_json, main_pool_id)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeGenerateTextAsync(
        env: *mut c_void,
        clazz: *mut c_void,
        runtime_ptr: i64,
        request_json: *mut c_void,
        listener: *mut c_void,
    ) -> *mut c_void {
        pb_jni_nativeGenerateTextAsync(env, clazz, runtime_ptr, request_json, listener)
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_org_phonebuddy_NativeRuntime_nativeCancelOperation(
        env: *mut c_void,
        clazz: *mut c_void,
        runtime_ptr: i64,
        operation_id: *mut c_void,
    ) {
        pb_jni_nativeCancelOperation(env, clazz, runtime_ptr, operation_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::tempdir;

    #[test]
    fn test_pb_string_free() {
        unsafe {
            // Null pointer is safe
            pb_string_free(std::ptr::null_mut());

            // Valid pointer is freed safely
            let s = CString::new("test string").unwrap();
            let raw = s.into_raw();
            pb_string_free(raw);
        }
    }

    #[test]
    fn test_pb_engine_lifecycle_and_config() {
        let dir = tempdir().unwrap();

        unsafe {
            // 1. Null engine free is safe
            pb_engine_free(std::ptr::null_mut());

            // 2. Null config JSON
            let mut err: *mut c_char = std::ptr::null_mut();
            let eng_null = pb_engine_new(std::ptr::null(), &mut err);
            assert!(eng_null.is_null());
            assert!(!err.is_null());
            pb_string_free(err);

            // 3. Invalid config JSON
            let mut err: *mut c_char = std::ptr::null_mut();
            let invalid_c = CString::new("{invalid json").unwrap();
            let eng_invalid = pb_engine_new(invalid_c.as_ptr(), &mut err);
            assert!(eng_invalid.is_null());
            assert!(!err.is_null());
            pb_string_free(err);

            // 4. Valid config JSON with client_profile
            let cfg_profile_json = format!(
                r#"{{
                    "api_key": "sk-ant-test",
                    "base_url": "https://api.anthropic.com/v1",
                    "model": "claude-opus-5",
                    "client_profile": "claude_code",

                    "client_version": "2.1.238",
                    "client_session_id": "sess-ffi-123",
                    "root_dir": "{}"
                }}"#,
                dir.path().display()
            );
            let valid_c = CString::new(cfg_profile_json).unwrap();
            let engine = pb_engine_new(valid_c.as_ptr(), &mut err);
            assert!(!engine.is_null());
            assert!(err.is_null());

            // 5. Config setters
            let name_c = CString::new("MobilePal").unwrap();
            pb_engine_set_agent_name(engine, name_c.as_ptr());
            pb_engine_set_agent_name(engine, std::ptr::null());

            let extra_c = CString::new("Be extremely concise.").unwrap();
            pb_engine_set_system_prompt_extra(engine, extra_c.as_ptr());
            pb_engine_set_system_prompt_extra(engine, std::ptr::null());

            // 6. Session APIs
            let mut err: *mut c_char = std::ptr::null_mut();
            let list_ptr = pb_engine_list_sessions(engine, &mut err);
            assert!(!list_ptr.is_null());
            assert!(err.is_null());
            let list_str = CStr::from_ptr(list_ptr).to_str().unwrap();
            assert_eq!(list_str, "[]");
            pb_string_free(list_ptr);

            let session_id_c = CString::new("non_existent").unwrap();
            let sess_ptr = pb_engine_get_session(engine, session_id_c.as_ptr(), &mut err);
            assert!(sess_ptr.is_null());

            let del_res = pb_engine_delete_session(engine, session_id_c.as_ptr());
            assert_eq!(del_res, 0);

            // 7. Cancel API
            pb_engine_cancel(engine, session_id_c.as_ptr());

            // 8. Host tool & WebView result helpers
            let call_id_c = CString::new("dummy_call").unwrap();
            let output_c = CString::new("output").unwrap();
            let tool_res = pb_engine_host_tool_result(
                engine,
                call_id_c.as_ptr(),
                1,
                output_c.as_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(tool_res, -3); // Unknown call_id returns -3

            let wv_res = pb_engine_webview_result(
                engine,
                call_id_c.as_ptr(),
                1,
                output_c.as_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(wv_res, -3); // Unknown call_id returns -3

            // Free engine
            pb_engine_free(engine);
        }
    }

    struct ChatTestContext {
        engine_ptr: usize,
    }

    unsafe extern "C" fn test_llm_cb(
        request_id: *const c_char,
        _request_json: *const c_char,
        user_data: *mut c_void,
    ) {
        let ctx = &*(user_data as *const ChatTestContext);
        let req_id_str = CStr::from_ptr(request_id).to_str().unwrap();

        let engine = ctx.engine_ptr as *mut PbEngine;

        // Push text chunk
        let chunk_json = serde_json::json!({
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1000,
            "model": "mock",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "FFI chat response"
                },
                "finish_reason": null
            }]
        })
        .to_string();
        let chunk_c = CString::new(chunk_json).unwrap();
        let req_c = CString::new(req_id_str).unwrap();

        let push_res = pb_engine_llm_push_chunk(
            engine,
            req_c.as_ptr(),
            chunk_c.as_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(push_res, 0);

        let finish_res = pb_engine_llm_finish(engine, req_c.as_ptr(), std::ptr::null_mut());
        assert_eq!(finish_res, 0);
    }

    unsafe extern "C" fn test_event_cb(event_json: *const c_char, user_data: *mut c_void) {
        let events = &mut *(user_data as *mut Vec<String>);
        let s = CStr::from_ptr(event_json).to_str().unwrap();
        events.push(s.to_string());
    }

    #[test]
    fn test_pb_engine_chat_flow() {
        let dir = tempdir().unwrap();
        let cfg_json = format!(
            r#"{{
                "llm_mode": "host",
                "model": "host-model",
                "root_dir": "{}"
            }}"#,
            dir.path().display()
        );

        unsafe {
            let mut err: *mut c_char = std::ptr::null_mut();
            let valid_c = CString::new(cfg_json).unwrap();
            let engine = pb_engine_new(valid_c.as_ptr(), &mut err);
            assert!(!engine.is_null());

            let mut test_ctx = ChatTestContext {
                engine_ptr: engine as usize,
            };

            pb_engine_set_host_callbacks(
                engine,
                Some(test_llm_cb),
                None,
                &mut test_ctx as *mut ChatTestContext as *mut c_void,
            );

            let mut received_events: Vec<String> = Vec::new();
            let session_id_c = CString::new("test_ffi_session").unwrap();
            let user_msg_c = CString::new("Hello from FFI test").unwrap();

            let outcome_ptr = pb_engine_chat(
                engine,
                session_id_c.as_ptr(),
                user_msg_c.as_ptr(),
                Some(test_event_cb),
                &mut received_events as *mut Vec<String> as *mut c_void,
                &mut err,
            );

            assert!(!outcome_ptr.is_null());
            assert!(err.is_null());

            let outcome_str = CStr::from_ptr(outcome_ptr).to_str().unwrap();
            let outcome_json: serde_json::Value = serde_json::from_str(outcome_str).unwrap();
            assert_eq!(outcome_json["final_text"], "FFI chat response");
            assert!(!received_events.is_empty());

            pb_string_free(outcome_ptr);
            pb_engine_free(engine);
        }
    }

    static TEST_LOGS: std::sync::Mutex<Vec<(i32, String, String)>> =
        std::sync::Mutex::new(Vec::new());

    unsafe extern "C" fn test_log_cb(level: i32, target: *const c_char, message: *const c_char) {
        let t = if target.is_null() {
            ""
        } else {
            CStr::from_ptr(target).to_str().unwrap_or("")
        };
        let m = if message.is_null() {
            ""
        } else {
            CStr::from_ptr(message).to_str().unwrap_or("")
        };
        TEST_LOGS
            .lock()
            .unwrap()
            .push((level, t.to_string(), m.to_string()));
    }

    #[test]
    fn test_pb_init_logging_captures_traces() {
        unsafe {
            pb_init_logging(Some(test_log_cb), 3); // min_level = 3 (INFO)
        }

        tracing::warn!("test warning message from ffi test");
        tracing::info!("test info message from ffi test");

        let logs = TEST_LOGS.lock().unwrap();
        assert!(
            logs.iter()
                .any(|(lvl, _target, msg)| *lvl == 2 && msg.contains("test warning message")),
            "expected WARN log, got: {logs:?}"
        );
        assert!(
            logs.iter()
                .any(|(lvl, _target, msg)| *lvl == 3 && msg.contains("test info message")),
            "expected INFO log, got: {logs:?}"
        );
    }

    fn title_routing_json(base_url: &str) -> String {
        format!(
            r#"{{
                "providers": [{{
                    "provider_id": "light-title",
                    "base_url": "{base_url}",
                    "api_key": "secret-must-not-appear",
                    "model": "light-model",
                    "api_backend": "chat_completions"
                }}],
                "pools": {{
                    "session_title": {{
                        "members": [{{
                            "provider_id": "light-title",
                            "routing_group": "cheap",
                            "base_score": 10,
                            "order": 0,
                            "enabled": true
                        }}],
                        "when_exhausted": "fail_fast"
                    }}
                }}
            }}"#
        )
    }

    unsafe extern "C" fn generate_done_cb(envelope_json: *const c_char, user_data: *mut c_void) {
        let tx = &*(user_data as *const std::sync::mpsc::Sender<String>);
        let s = if envelope_json.is_null() {
            String::new()
        } else {
            CStr::from_ptr(envelope_json)
                .to_str()
                .unwrap_or("")
                .to_string()
        };
        let _ = tx.send(s);
    }

    #[test]
    fn test_pb_runtime_new_free_and_null_args() {
        let dir = tempdir().unwrap();
        unsafe {
            pb_runtime_free(std::ptr::null_mut());
            pb_runtime_cancel_operation(std::ptr::null_mut(), std::ptr::null());
            pb_runtime_update_routing(std::ptr::null_mut(), std::ptr::null(), std::ptr::null_mut());

            let mut err: *mut c_char = std::ptr::null_mut();
            let rt = pb_runtime_new(std::ptr::null(), std::ptr::null(), &mut err);
            assert!(rt.is_null());
            assert!(!err.is_null());
            pb_string_free(err);

            let routing = title_routing_json("http://127.0.0.1:1");
            let routing_c = CString::new(routing).unwrap();
            let root_c = CString::new(dir.path().to_string_lossy().as_ref()).unwrap();
            let mut err: *mut c_char = std::ptr::null_mut();
            let runtime = pb_runtime_new(routing_c.as_ptr(), root_c.as_ptr(), &mut err);
            assert!(err.is_null(), "runtime new err");
            assert!(!runtime.is_null());

            let op = pb_runtime_generate_text_async(
                runtime,
                std::ptr::null(),
                None,
                std::ptr::null_mut(),
                &mut err,
            );
            assert!(op.is_null());
            assert!(!err.is_null());
            pb_string_free(err);

            pb_runtime_free(runtime);
        }
    }

    #[test]
    fn test_pb_runtime_generate_text_missing_pool_completes() {
        let dir = tempdir().unwrap();
        let routing = r#"{"providers":[],"pools":{}}"#;
        unsafe {
            let routing_c = CString::new(routing).unwrap();
            let root_c = CString::new(dir.path().to_string_lossy().as_ref()).unwrap();
            let mut err: *mut c_char = std::ptr::null_mut();
            let runtime = pb_runtime_new(routing_c.as_ptr(), root_c.as_ptr(), &mut err);
            assert!(!runtime.is_null());

            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let req = CString::new(r#"{"pool_id":"session_title","input":"hello"}"#).unwrap();
            let op = pb_runtime_generate_text_async(
                runtime,
                req.as_ptr(),
                Some(generate_done_cb),
                &tx as *const _ as *mut c_void,
                &mut err,
            );
            assert!(!op.is_null(), "expected operation id");
            assert!(err.is_null());
            let op_str = CStr::from_ptr(op).to_str().unwrap().to_string();
            pb_string_free(op);

            let envelope = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            assert!(
                !envelope.contains("secret"),
                "envelope must not include API keys: {envelope}"
            );
            let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
            assert_eq!(v["version"], 1);
            assert_eq!(v["ok"], false);
            assert_eq!(v["operation_id"], op_str);
            assert_eq!(v["error"]["kind"], "RouteNotConfigured");
            assert_eq!(v["error"]["pool_id"], "session_title");

            pb_runtime_free(runtime);
        }
    }

    #[test]
    fn test_pb_runtime_cancel_does_not_hang() {
        let dir = tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                std::thread::sleep(std::time::Duration::from_secs(30));
                drop(stream);
            }
        });

        let routing = title_routing_json(&format!("http://{addr}/v1"));
        unsafe {
            let routing_c = CString::new(routing).unwrap();
            let root_c = CString::new(dir.path().to_string_lossy().as_ref()).unwrap();
            let mut err: *mut c_char = std::ptr::null_mut();
            let runtime = pb_runtime_new(routing_c.as_ptr(), root_c.as_ptr(), &mut err);
            assert!(!runtime.is_null());

            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let req =
                CString::new(r#"{"pool_id":"session_title","input":"title me","timeout_ms":8000}"#)
                    .unwrap();
            let op = pb_runtime_generate_text_async(
                runtime,
                req.as_ptr(),
                Some(generate_done_cb),
                &tx as *const _ as *mut c_void,
                &mut err,
            );
            assert!(!op.is_null());
            pb_runtime_cancel_operation(runtime, op);
            let envelope = rx
                .recv_timeout(std::time::Duration::from_secs(3))
                .expect("cancel must complete the callback");
            let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
            assert_eq!(v["ok"], false);
            let kind = v["error"]["kind"].as_str().unwrap_or("");
            assert!(
                kind == "OperationCancelled" || kind == "OperationTimedOut" || kind == "Llm",
                "unexpected kind {kind} envelope={envelope}"
            );
            pb_string_free(op);
            pb_runtime_free(runtime);
        }
    }

    #[test]
    fn test_pb_runtime_free_cancels_in_flight_generate() {
        let dir = tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                std::thread::sleep(std::time::Duration::from_secs(30));
                drop(stream);
            }
        });

        let routing = title_routing_json(&format!("http://{addr}/v1"));
        unsafe {
            let routing_c = CString::new(routing).unwrap();
            let root_c = CString::new(dir.path().to_string_lossy().as_ref()).unwrap();
            let mut err: *mut c_char = std::ptr::null_mut();
            let runtime = pb_runtime_new(routing_c.as_ptr(), root_c.as_ptr(), &mut err);
            assert!(!runtime.is_null());

            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let req = CString::new(r#"{"pool_id":"session_title","input":"title me"}"#).unwrap();
            let op = pb_runtime_generate_text_async(
                runtime,
                req.as_ptr(),
                Some(generate_done_cb),
                &tx as *const _ as *mut c_void,
                &mut err,
            );
            assert!(!op.is_null());
            pb_string_free(op);
            std::thread::sleep(std::time::Duration::from_millis(30));
            pb_runtime_free(runtime);
            let envelope = rx
                .recv_timeout(std::time::Duration::from_secs(3))
                .expect("free must complete the callback");
            let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
            assert_eq!(v["ok"], false);
            assert_eq!(
                v["error"]["kind"].as_str().unwrap_or(""),
                "OperationCancelled",
                "envelope={envelope}"
            );
        }
    }
}
