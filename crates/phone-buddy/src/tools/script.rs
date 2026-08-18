//! `run_script` tool — execute model-generated JavaScript.
//!
//! Mobile platforms cannot spawn interpreters, so the engine embeds a pure
//! Rust JavaScript engine (boa) with a small sandboxed host API:
//!
//! - `console.log/info/warn/error` — captured and returned to the model;
//! - `readFile(path)`, `writeFile(path, text)`, `listDir(path)` — sandboxed.
//!
//! Runtime limits (loop iteration budget) bound runaway scripts.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use boa_engine::gc::{empty_trace, Finalize, Trace};
use boa_engine::object::ObjectInitializer;
use boa_engine::property::Attribute;
use boa_engine::{js_string, Context, JsError, JsNativeError, JsResult, JsValue, NativeFunction, Source};
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::fs::Sandbox;
use crate::tools::{arg_str, schema_object, s_string, Tool, ToolCtx, ToolOutput, ToolSpec};

/// Bound for `for`/`while` iterations in user scripts.
const LOOP_LIMIT: u64 = 20_000_000;
/// Max chars of console output returned to the model.
const MAX_OUTPUT_CHARS: usize = 30_000;

pub struct RunScriptTool;

#[async_trait]
impl Tool for RunScriptTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_script".into(),
            description: concat!(
                "Run a JavaScript program (embedded engine, no network). ",
                "Helpers: console.log, readFile(path), writeFile(path, text), listDir(path). ",
                "Inspect input files first, compute in the script, print key results, ",
                "write artifacts with writeFile when useful, and sanity-check the numbers."
            )
            .into(),
            parameters: schema_object(
                vec![
                    ("code", s_string(), "JavaScript source code."),
                ],
                &["code"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let code = arg_str(&args, "code")?;
        let sandbox = ctx.sandbox.clone();

        // Run JS on a blocking thread to avoid stalling the async runtime.
        tokio::task::spawn_blocking(move || {
            run_js(&code, &sandbox)
        })
        .await
        .unwrap_or_else(|e| Err(EngineError::Script(format!("script task panicked: {e}"))))
    }
}

// ── Capture handles for JS closures ──────────────────────────────────────

#[derive(Clone)]
struct LogHandle(Arc<Mutex<Vec<String>>>);

impl Finalize for LogHandle {}
unsafe impl Trace for LogHandle {
    empty_trace!();
}

#[derive(Clone)]
struct LogCapture {
    log: LogHandle,
    level: String,
}

impl Finalize for LogCapture {}
unsafe impl Trace for LogCapture {
    empty_trace!();
}

#[derive(Clone)]
struct SandboxHandle(Arc<Sandbox>);

impl Finalize for SandboxHandle {}
unsafe impl Trace for SandboxHandle {
    empty_trace!();
}

impl Sandbox {
    fn handle(self: &Arc<Self>) -> SandboxHandle {
        SandboxHandle(self.clone())
    }
}

// ── Runner ───────────────────────────────────────────────────────────────

fn run_js(
    code: &str,
    sandbox: &Arc<Sandbox>,
) -> EngineResult<ToolOutput> {
    let mut context = Context::default();
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(LOOP_LIMIT);

    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sb = sandbox.handle();

    // console.{log,info,warn,error} → captured buffer
    let mut console_builder = ObjectInitializer::new(&mut context);
    for level in ["log", "info", "warn", "error"] {
        let f = capture_fn(log.clone(), level);
        console_builder.function(f, js_string!(level), 1);
    }
    let console = console_builder.build();
    context
        .register_global_property(js_string!("console"), console, Attribute::all())
        .map_err(js_err)?;

    // readFile(path) -> string
    let f = NativeFunction::from_copy_closure_with_captures(
        |_this, args, sb: &SandboxHandle, ctx| {
            let Some(path) = first_str_arg(args, ctx, "readFile(path)")? else {
                return Err(js_error_msg("readFile(path)".into()));
            };
            let abs = sb.0.resolve(&path).map_err(|e| js_error_msg(e.to_string()))?;
            let bytes = std::fs::read(&abs).map_err(|e| js_error_msg(format!("{path}: {e}")))?;
            let ext = abs
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if crate::tools::binary::is_binary(&ext, &bytes) {
                return Err(js_error_msg(format!("{path}: binary file")));
            }
            let (cow, _, _) = encoding_rs::UTF_8.decode(&bytes);
            Ok(JsValue::from(js_string!(cow.as_ref())))
        },
        sb.clone(),
    );
    context
        .register_global_property(
            js_string!("readFile"),
            f.to_js_function(context.realm()),
            Attribute::all(),
        )
        .map_err(js_err)?;

    // writeFile(path, text) -> bytes written
    let f = NativeFunction::from_copy_closure_with_captures(
        |_this, args, sb: &SandboxHandle, ctx| {
            let Some(path) = first_str_arg(args, ctx, "writeFile(path, text)")? else {
                return Err(js_error_msg("writeFile(path, text)".into()));
            };
            let Some(content) = args.get(1).map(|v| v.to_string(ctx)).transpose()? else {
                return Err(js_error_msg("writeFile(path, text)".into()));
            };
            let content = content.to_std_string_escaped();
            let abs = sb.0.resolve(&path).map_err(|e| js_error_msg(e.to_string()))?;
            if let Some(parent) = abs.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&abs, &content).map_err(|e| js_error_msg(format!("{path}: {e}")))?;
            Ok(JsValue::from(content.len() as f64))
        },
        sb.clone(),
    );
    context
        .register_global_property(
            js_string!("writeFile"),
            f.to_js_function(context.realm()),
            Attribute::all(),
        )
        .map_err(js_err)?;

    // listDir(path) -> string[]
    let f = NativeFunction::from_copy_closure_with_captures(
        |_this, args, sb: &SandboxHandle, ctx| {
            let Some(path) = first_str_arg(args, ctx, "listDir(path)")? else {
                return Err(js_error_msg("listDir(path)".into()));
            };
            let abs = sb.0.resolve(&path).map_err(|e| js_error_msg(e.to_string()))?;
            let mut names = Vec::new();
            if abs.is_dir() {
                let mut entries: Vec<_> = std::fs::read_dir(&abs)
                    .map_err(|e| js_error_msg(format!("{path}: {e}")))?
                    .filter_map(|e| e.ok())
                    .collect();
                entries.sort_by_key(|e| e.file_name());
                for e in entries {
                    let name = e.file_name().to_string_lossy().into_owned();
                    names.push(if e.path().is_dir() {
                        format!("{name}/")
                    } else {
                        name
                    });
                }
            }
            let json = serde_json::Value::Array(
                names.into_iter().map(serde_json::Value::String).collect(),
            );
            JsValue::from_json(&json, ctx)
        },
        sb.clone(),
    );
    context
        .register_global_property(
            js_string!("listDir"),
            f.to_js_function(context.realm()),
            Attribute::all(),
        )
        .map_err(js_err)?;

    let result = context.eval(Source::from_bytes(code));

    let mut output = String::new();
    {
        let logs = log.lock().unwrap();
        for line in logs.iter() {
            output.push_str(line);
            output.push('\n');
        }
    }

    match result {
        Ok(value) => {
            if !value.is_undefined() {
                let repr = value
                    .to_string(&mut context)
                    .map(|s| s.to_std_string_escaped())
                    .unwrap_or_else(|_| "<value>".into());
                output.push_str(&format!("[result] {repr}\n"));
            }
            Ok(ToolOutput::new(crate::tools::truncate_chars(
                &output,
                MAX_OUTPUT_CHARS,
            )))
        }
        Err(e) => {
            let mut msg = output;
            msg.push_str(&format!("[script error] {e}"));
            Err(EngineError::Script(crate::tools::truncate_chars(
                &msg,
                MAX_OUTPUT_CHARS,
            )))
        }
    }
}

fn first_str_arg(
    args: &[JsValue],
    ctx: &mut Context,
    usage: &str,
) -> JsResult<Option<String>> {
    let Some(v) = args.first() else {
        return Err(js_error_msg(usage.into()));
    };
    if v.is_undefined() || v.is_null() {
        return Err(js_error_msg(usage.into()));
    }
    Ok(Some(v.to_string(ctx)?.to_std_string_escaped()))
}

fn capture_fn(log: Arc<Mutex<Vec<String>>>, level: &str) -> NativeFunction {
    NativeFunction::from_copy_closure_with_captures(
        |_this, args, cap: &LogCapture, ctx| {
            let mut parts = Vec::new();
            for arg in args {
                let s = arg
                    .to_string(ctx)
                    .map(|s| s.to_std_string_escaped())
                    .unwrap_or_else(|_| "<value>".into());
                parts.push(s);
            }
            let line = parts.join(" ");
            let mut buf = cap.log.0.lock().unwrap();
            if cap.level == "log" || cap.level == "info" {
                buf.push(line);
            } else {
                buf.push(format!("[{}] {line}", cap.level));
            }
            Ok(JsValue::undefined())
        },
        LogCapture {
            log: LogHandle(log),
            level: level.to_string(),
        },
    )
}

fn js_err(e: JsError) -> EngineError {
    EngineError::Script(format!("{e}"))
}

fn js_error_msg(msg: String) -> JsError {
    JsError::from_native(JsNativeError::typ().with_message(msg))
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(RunScriptTool)
}
