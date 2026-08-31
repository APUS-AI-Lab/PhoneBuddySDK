//! Host diagnostics that survive release builds.
//!
//! Workspace `tracing` is compiled with `release_max_level_off`, so
//! `tracing::info!` in the Android/iOS `.so` is a no-op. Router ranking
//! still has to reach logcat / Metro before every LLM HTTP request, so
//! this module talks to the FFI log callback directly.

use std::sync::RwLock;

static SINK: RwLock<Option<fn(i32, &str, &str)>> = RwLock::new(None);

#[cfg(test)]
static TEST_LOG: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Install (or clear) the host log callback. FFI `pb_init_logging` owns this.
pub fn set_sink(cb: Option<fn(i32, &str, &str)>) {
    if let Ok(mut guard) = SINK.write() {
        *guard = cb;
    }
}

/// INFO-level host log. Never includes secrets; callers must not pass API keys.
pub fn info(target: &str, message: &str) {
    emit(3, target, message);
}

fn emit(level: i32, target: &str, message: &str) {
    #[cfg(test)]
    if let Ok(mut log) = TEST_LOG.lock() {
        log.push(format!("{target}|{message}"));
    }
    let cb = match SINK.read() {
        Ok(guard) => *guard,
        Err(_) => None,
    };
    if let Some(cb) = cb {
        cb(level, target, message);
    }
}

#[cfg(test)]
pub fn take_test_messages() -> Vec<String> {
    TEST_LOG.lock().unwrap().drain(..).collect()
}
