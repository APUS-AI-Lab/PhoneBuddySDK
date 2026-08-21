# PhoneBuddy SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20Android%20%7C%20C%20ABI-green.svg)]()
[![Rust](https://img.shields.io/badge/Rust-1.94%2B-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

**English** | [简体中文](README_CN.md)

> A lightweight, embeddable LLM Agent runtime engine in pure Rust, designed specifically for mobile operating systems (iOS and Android).

**PhoneBuddy SDK** empowers mobile applications with autonomous Agent capabilities: dynamic task planning, pure-Rust virtual shell applets, embedded JavaScript execution & data analytics (`boa_engine`), in-memory subagents, scheduled tasks, human-in-the-loop interactions, and real-time streaming callbacks—all strictly adhering to Apple App Store and Google Play sandbox restrictions with **zero child processes spawned**.

PhoneBuddy SDK's core agent runtime is derived and adapted from xAI's open-source agent engine, **[xai-org/grok-build](https://github.com/xai-org/grok-build)**. Leading desktop AI agent harnesses—such as **Claude Code**, OpenAI's **[Codex](https://github.com/openai/codex)**, and xAI's **[grok-build](https://github.com/xai-org/grok-build)**—have established state-of-the-art standards for autonomous agent execution loops on desktop and server environments. **PhoneBuddy SDK brings this exact tier of desktop-class harness maturity down to mobile platforms (iOS & Android).**

---

## 🌟 Core Features & Highlights

- 🔒 **100% Mobile Sandbox Compliant**
  - Completely eliminates `std::process::Command`, `tokio::process`, and `fork`/`exec` syscalls.
  - Passes Apple App Store and Google Play app sandbox security reviews with zero permission escalations.
- 🎯 **Autonomous Planning & Dynamic ReAct Loop**
  - Dynamically decomposes complex user requests into structured, step-by-step execution plans (`plan`).
  - Sequentially dispatches tools, inspects outputs, and automatically refines execution paths until task completion.
- 🧰 **Pure-Rust In-Memory Virtual Shell (`busybox`)**
  - Built-in pure-Rust POSIX command applets executing entirely within memory: `cat`, `head`, `tail`, `ls`, `wc`, `sort`, `uniq`, `find`, `echo`, `touch`, `mkdir`, `rm`, `cp`, `mv`, `du`, `pwd`, `basename`, `dirname` (with fast regex & glob text search provided by the dedicated `grep` tool).
- ⚡ **Embedded JavaScript Sandbox (`boa_engine`)**
  - Integrated pure-Rust ECMAScript engine (`run_script` tool) for safe in-app mathematical computations, algorithmic filtering, data manipulation, and JSON transformations.
- 📊 **Embedded Data Analytics & Scripting**
  - Fast in-memory data processing, aggregation, mathematical statistics, and structured JSON/CSV manipulation via the embedded JavaScript engine (`run_script`) and sandboxed file tools without external runtimes.
- 📁 **Jailed File Sandbox & Deep Search**
  - Jails all file operations within a designated `root_dir` (e.g. app's Documents directory) with strict path traversal prevention.
  - Comprehensive toolset: `read_file` (with line slicing, offset navigation, confusable unicode mapping), `write_file`, `edit_file` (smart chunk replacement), `list_dir` (recursive directory scanning), and `grep` (fast regex and glob search).
- 🤖 **In-Memory Subagents & Asynchronous Task Coordination**
  - Spawns concurrent background subagents (`task`) managed by an in-memory Tokio async task manager.
  - Complete task lifecycle control: query task execution logs (`task_output`, `get_task_output`), synchronize on multiple tasks (`wait_tasks`), cancel tasks (`kill_task`), and monitor running jobs (`monitor`).
- ⏰ **Scheduling, Notifications & Human-in-the-Loop**
  - In-memory cron and timer task scheduling (`scheduler`).
  - System push notifications and UI alerts via host callbacks (`notification`).
  - Interactive user prompts (`ask_user_question`) supporting multi-choice options and freeform write-ins.
- 🌐 **Web Search & Fetch with SSRF Protection**
  - Live web search (`web_search`) with multi-engine routing (headless DuckDuckGo Lite via host WebView, LLM search API fallback, or Grok hosted `{type: web_search}` on Responses when `enable_web_search` is on).
  - Clean HTML-to-Markdown page conversion via pure-Rust `htmd` (`web_fetch`) with built-in SSRF protection blocking private network / loopback ranges.
- 🛡️ **Production-Grade Harness Robustness**
  - **Doom-Loop Detection**: Identifies action stationarity (`IdenticalToolCallRun`, 8-turn nudge / 16-turn break) plus server-side check support (`x-grok-doom-loop-check`).
  - **Context Compaction**: Heuristic token estimation with automatic history compaction when surpassing `24,000` tokens.
  - **Fault Resilience**: Exponential backoff retry logic with jitter handling rate limits (HTTP 429) and server errors (HTTP 5xx).
- 🔌 **Flexible Protocol Support & Extensible Host Bridge**
  - Supports multiple LLM backends: `responses` (OpenAI Response API + SSE streaming), `chat_completions`, and `messages`.
  - Pure-Rust TLS transport via `rustls-ring` with HTTP/2 streaming.
  - Host LLM mode (`LlmMode::Host`) allowing inference delegation to on-device models (e.g. `llama.cpp` / `llama.rn`) or custom host gateways via `PbLlmRequestCallback`.
  - Dynamic host tool injection (`PbHostToolCallback`) from Swift, Kotlin, or C.
- 📱 **Cross-Platform C ABI & Idiomatic Native Wrappers**
  - Clean C-ABI interface (`phone_buddy.h`) with guaranteed unwind safety (`catch_unwind`).
  - First-class Swift SDK (`PhoneBuddy.swift`) with `async`/`await` and SwiftUI support.
  - First-class Android Kotlin SDK (`NativeAgent.kt` + JNI bridge) with Jetpack Compose support.
  - Local CLI development tool (`phone-buddy-cli`) for testing, debugging, and self-testing tools offline.

---

## 🏗 Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            Host Application Layer                                │
│        iOS App (Swift / SwiftUI)          Android App (Kotlin / Compose)         │
│        C / C++ Native Applications        Local Developer CLI (phone-buddy-cli)  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┴────────────────────────────────────────┐
│                     Platform Integration & C FFI Layer                           │
│  • phone_buddy.h (C ABI)                 • PhoneBuddy.swift (Swift Async/Await)  │
│  • phonebuddy_jni.c / NativeAgent.kt     • Host Callbacks (LLM, Tool, WebView)   │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┴────────────────────────────────────────┐
│                        Core Agent Engine (phone-buddy)                           │
│                                                                                  │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐  │
│  │   Agent Turn Loop    │ │   Subagent System    │ │    LLM Transport Layer   │  │
│  │ • Task Planner       │ │ • In-Memory Manager  │ │ • Responses (SSE stream) │  │
│  │ • Doom-Loop Detector │ │ • Task Concurrency   │ │ • ChatCompletions        │  │
│  │ • History Compactor  │ │ • Scheduler (cron)   │ │ • Messages               │  │
│  │ • Session Store      │ │ • Task Monitor       │ │ • Host LLM / rustls-ring │  │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────────┘  │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Pure-Rust Toolset & Sandboxes                       │  │
│  │ • File Sandbox: read_file, write_file, edit_file, list_dir, grep           │  │
│  │ • Virtual POSIX Shell: Pure-Rust BusyBox Applets (cat, head, sort, uniq, …)│  │
│  │ • Embedded JS Runtime: boa_engine ECMAScript sandbox (run_script)          │  │
│  │ • Data Analytics: In-memory JavaScript computations & CSV/JSON processing  │  │
│  │ • Web & Network: web_search (WebView / DDG), web_fetch (htmd), SSRF guard │  │
│  │ • Interaction: ask_user_question (Human-in-the-loop), notification, Host   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. iOS Integration (Swift)

Import `libphone_buddy_ffi.a` and `phone_buddy.h` into your Xcode project, or directly include [`PhoneBuddy.swift`](examples/ios/PhoneBuddy.swift):

```swift
import Foundation

// 1. Initialize configuration
let config = PhoneBuddyConfig(
    apiKey: "your-api-key-or-jwt-token",
    baseUrl: "https://api.x.ai/v1",
    model: "grok-4.6",
    apiBackend: "responses", // OpenAI Response API + SSE streaming
    rootDir: PhoneBuddyConfig.sandboxRoot(workspaceName: "workspace"),
    maxTurns: 24,
    enableWebSearch: true,
    agentName: "Acme", // system-prompt identity; omit to keep "PhoneBuddy"
    extraHeaders: [
        "X-App-Version": "1.0.0",
        "X-Client-Platform": "iOS"
    ]
)

// 2. Instantiate the engine
let engine = try PhoneBuddyEngine(config: config)
// engine.setAgentName("Acme") // optional runtime rename; nil/empty resets to PhoneBuddy

// 3. Run agent turn with real-time streaming events
let outcome = try await engine.chat(
    sessionId: "session-001",
    userInput: "Analyze the quarterly sales data in data/sales.csv and compute total profit",
    onEvent: { eventJson in
        print("Real-time event: \(eventJson)")
    }
)

print("Final response: \(outcome.finalText)")
print("Turns used: \(outcome.turnsUsed)")
```

*See full iOS demo at [examples/ios/](examples/ios/)*

---

### 2. Android Integration (Kotlin)

Place `libphone_buddy_ffi.so` into `app/src/main/jniLibs/<abi>/` and include [`NativeAgent.kt`](examples/android/NativeAgent.kt):

```kotlin
import org.phonebuddy.NativeAgent
import org.phonebuddy.EventListener
import org.json.JSONObject

// 1. Prepare configuration JSON
val config = JSONObject().apply {
    put("api_key", "your-api-key-or-jwt-token")
    put("base_url", "https://api.x.ai/v1")
    put("model", "grok-4.6")
    put("api_backend", "responses")
    put("root_dir", context.filesDir.resolve("workspace").absolutePath)
    put("max_turns", 24)
    put("enable_web_search", false)
    put("agent_name", "Acme")
    put("extra_headers", JSONObject().apply {
        put("X-App-Version", "1.0.0")
        put("X-Client-Platform", "Android")
    })
}

// 2. Create agent instance
val agent = NativeAgent(config.toString(), context)
// agent.setAgentName("Acme") // optional runtime rename; null/blank resets to PhoneBuddy

// 3. Execute chat turn
val resultJson = agent.chat(
    sessionId = "session-001",
    userInput = "Analyze the quarterly sales data in data/sales.csv and compute total profit",
    listener = object : EventListener {
        override fun onEvent(eventJson: String) {
            println("Real-time event: $eventJson")
        }
    }
)

val outcome = JSONObject(resultJson ?: "{}")
println("Final response: ${outcome.optString("final_text")}")
```

*See full Android demo at [examples/android/](examples/android/)*

---

### 3. Standard C API (`phone_buddy.h`)

```c
#include <stdio.h>
#include "phone_buddy.h"

void on_event(const char *event_json, void *user_data) {
    printf("[Event] %s\n", event_json);
}

int main(void) {
    const char *config_json = "{"
        "\"api_key\":\"your-api-key\","
        "\"base_url\":\"https://api.x.ai/v1\","
        "\"model\":\"grok-4.6\","
        "\"root_dir\":\"/tmp/phone-buddy-workspace\""
    "}";

    char *err = NULL;
    PbEngine *engine = pb_engine_new(config_json, &err);
    if (!engine) {
        fprintf(stderr, "Engine init failed: %s\n", err);
        pb_string_free(err);
        return 1;
    }

    char *result = pb_engine_chat(engine, "session-001", "Hello Agent", on_event, NULL, &err);
    if (result) {
        printf("Result: %s\n", result);
        pb_string_free(result);
    } else {
        fprintf(stderr, "Chat error: %s\n", err);
        pb_string_free(err);
    }

    pb_engine_free(engine);
    return 0;
}
```

*See C demo at [examples/c_demo/](examples/c_demo/)*

---

### 4. Local CLI Testing Tool (`phone-buddy-cli`)

Use the built-in CLI for local debugging, automated self-tests, and live chat:

```bash
# 1. Run offline scripted demo (analyzes sample data without API key)
cargo run -p phone-buddy-cli -- mock

# 2. Run built-in tool self-test (exercises BusyBox, JS engine, and grep directly)
cargo run -p phone-buddy-cli -- self-test

# 3. Run real LLM interactive turn
PHONEBUDDY_API_KEY="your-api-key" cargo run -p phone-buddy-cli -- chat "Analyze the sales numbers"
```

---

## 📡 Real-Time Streaming Events

During execution, `PhoneBuddyEngine` emits structured JSON event objects to the registered callback handler:

| Event Type | Field / Payload | Description |
| :--- | :--- | :--- |
| `TextDelta` | `{"text": "..."}` | Streaming delta chunk of assistant text response |
| `ReasoningDelta` | `{"text": "..."}` | Streaming delta chunk of LLM reasoning / thinking |
| `ToolCallStart` | `{"call_id": "...", "name": "...", "arguments_json": "..."}` | Emitted immediately before a tool begins executing |
| `ToolCallResult` | `{"call_id": "...", "name": "...", "ok": true, "output": "..."}` | Emitted when a tool finishes execution with its output |
| `PlanUpdated` | `{"items_json": "..."}` | Emitted whenever the planning tool updates the task step list |
| `Completed` | `{"final_text": "...", "usage": { ... }}` | Emitted when the turn finishes successfully |
| `Failed` | `{"message": "..."}` | Emitted when an unrecoverable failure occurs during the turn |

---

## ⚙️ Configuration Reference

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `api_key` | String | Yes (in HTTP mode) | `""` | User API Key or Bearer Token |
| `base_url` | String | Yes (in HTTP mode) | `https://api.x.ai/v1` | OpenAI / xAI compatible endpoint URL |
| `model` | String | Yes | `grok-4` | Model identifier (e.g. `grok-4.6`, `grok-4`, `gpt-4o`) |
| `root_dir` | String | Yes | `/tmp/phone-buddy` | Root path for jailed file sandbox and session storage |
| `api_backend` | String | No | `chat_completions` | API protocol: `responses` (SSE stream), `chat_completions`, `messages` |
| `llm_mode` | String | No | `http` | Transport mode: `http` (direct networking) or `host` (FFI callback bridge) |
| `locale` | String | No | `zh` | UI locale instructing agent response language |
| `max_turns` | Integer | No | `24` | Maximum tool loop turns per user turn |
| `temperature` | Float | No | `0.2` | Model sampling temperature |
| `max_output_tokens` | Integer | No | `8192` | Maximum output token generation limit |
| `enable_web_search` | Boolean | No | `false` | Attach Grok hosted `{type: web_search}` on Responses (Grok Build `supports_backend_search`). Client DuckDuckGo/`web_search` function tool stays registered independently. Leave off for gateways that do not implement hosted search (e.g. PackyAPI). |
| `agent_name` | String | No | `PhoneBuddy` | Identity used in the system prompt (`You are {agent_name}…`). Empty falls back to `PhoneBuddy`. |
| `system_prompt_extra` | String | No | `null` | Custom persona or product instructions appended to system prompt |
| `stream_idle_timeout_secs`| Integer | No | `120` | Streaming connection idle timeout in seconds |
| `max_retries` | Integer | No | `5` | Maximum exponential backoff retry attempts for HTTP requests |
| `extra_headers` | Object | No | `{}` | Custom HTTP headers sent with LLM requests (`X-App-Version`, etc.) |
| `extra_body` | Object | No | `{}` | Custom JSON fields merged into top level of LLM request payload |
| `enable_doom_loop_check`| Boolean | No | Auto | Server-side doom loop header (`x-grok-doom-loop-check`) |
| `web_fetch_allow_local` | Boolean | No | `false` | Allow `web_fetch` to access local loopback addresses (for testing) |
| `http_dump` | Object | No | `{"mode":"off"}`| Raw HTTP request/response dumper for debugging (e.g. `{"mode":"on_error","max_files":30}`) |

---

## 🩺 HTTP Traffic Diagnostics & Raw Dump (Troubleshooting Network & API Errors)

When troubleshooting network connectivity issues, authentication errors, API gateway failures, or unexpected HTTP response codes (e.g. 4xx / 5xx) on mobile clients, configure `http_dump` to record complete raw HTTP request and response exchanges to disk:

```json
{
  "api_key": "your-access-token",
  "base_url": "https://api.example.com/v1",
  "model": "grok-4",
  "http_dump": {
    "mode": "on_error",
    "mask_sensitive": true,
    "max_files": 30
  }
}
```

### `http_dump` Configuration Options:
* **`mode`**:
  * `"off"` *(default)*: Traffic dumping disabled.
  * `"on_error"`: Dumps raw HTTP exchange only when a request fails (non-2xx HTTP status or network timeout/connection error).
  * `"all"`: Dumps all HTTP exchanges (both 200 OK handshakes and failures).
* **`dump_dir`**: Custom output directory. Defaults to `<root_dir>/.phonebuddy/http_dumps/`.
* **`mask_sensitive`**: Mask sensitive headers (`Authorization`, `x-api-key`, `cookie`, etc.). Defaults to `true`.
* **`max_files`**: Maximum number of dump JSON files retained before automatically rotating out the oldest files (FIFO rotation). Defaults to `30`.

### Error Message and Dump JSON Format:
When an error occurs, the SDK error message includes the absolute dump path:
```text
Completion error: Error: LLM request failed: status=500 { "error": ... } [HTTP dump: /data/user/0/com.example.app/files/sandbox/.phonebuddy/http_dumps/dump_20260821_211530_500_req_a1b2c3.json]
```

Each dump file captures:
```json
{
  "schema_version": "1.0",
  "request_id": "req_a1b2c3d4",
  "timestamp": "2026-08-21T21:15:30.123+08:00",
  "duration_ms": 352,
  "request": {
    "method": "POST",
    "url": "https://api.example.com/v1/chat/completions",
    "headers": {
      "accept": "text/event-stream",
      "authorization": "Bearer sk-p***3456",
      "content-type": "application/json"
    },
    "body": { "model": "grok-4", "messages": [...], "stream": true }
  },
  "response": {
    "status": 502,
    "status_text": "Bad Gateway",
    "headers": {
      "content-type": "text/html; charset=UTF-8",
      "server": "nginx/1.22.1",
      "x-gateway-trace-id": "gw-987654321"
    },
    "body_text": "<html><head><title>502 Bad Gateway</title></head><body>...</body></html>"
  },
  "error": "status=502 <html><head><title>502 Bad Gateway..."
}
```

---

## 📁 Repository Structure

```text
PhoneBuddySDK/
├── Cargo.toml                  # Workspace configuration
├── crates/
│   ├── phone-buddy/            # Core Rust Agent engine (planning, tools, sandbox, subagents)
│   ├── phone-buddy-ffi/        # C ABI export layer (exports phone_buddy.h)
│   └── phone-buddy-cli/        # Local CLI developer & testing tool
├── dist/                       # Compiled SDK release packages
│   ├── ios/                    # iOS static libraries (.a) & header
│   └── android/                # Android shared libraries (.so) & header
├── examples/                   # Native mobile app and C demos
│   ├── ios/                    # Swift wrapper (PhoneBuddy.swift) & SwiftUI demo app
│   ├── android/                # Kotlin wrapper (NativeAgent.kt), JNI bridge & Compose demo app
│   └── c_demo/                 # Standard C API usage sample (main.c)
└── scripts/                    # Cross-compilation and distribution packaging scripts
    ├── build-ios-sdk.sh        # iOS cross-compilation (aarch64 / simulator universal)
    ├── build-android-sdk.sh    # Android cross-compilation (arm64-v8a / x86_64)
    └── package-sdk.sh          # One-shot distribution ZIP bundler
```

---

## 🔨 Building from Source

### Prerequisites

- Rust `1.94+` with mobile target toolchains:
  ```bash
  rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
  rustup target add aarch64-linux-android x86_64-linux-android
  ```
- Xcode 14+ (for iOS builds)
- Android NDK r25+ (for Android builds, export `ANDROID_NDK_HOME`)

### Build Commands

```bash
# Check Rust compilation
cargo check

# Run unit, integration, and E2E tests
cargo test

# Build iOS static libraries (device + simulator)
./scripts/build-ios-sdk.sh

# Build Android dynamic shared libraries
./scripts/build-android-sdk.sh

# One-click build and install demo on iOS device / simulator:
./examples/ios/build-and-install.sh [--simulator | --device]

# One-click build and install demo on Android phone:
./examples/android/build-and-install.sh [--logs]

# Package release distribution artifacts into dist/
./scripts/package-sdk.sh --build
```

---

## 🔒 Mobile Security & Sandbox Constraints

1. **Zero Subprocess Guarantee**: All operations (BusyBox shell commands, JavaScript execution, file edits) run entirely in-process in pure Rust. No `fork`, `exec`, or shell processes are ever invoked.
2. **Strict File Jailing**: The file sandbox prevents directory traversal attacks (`../`) and enforces boundary checks against `root_dir`.
3. **SSRF Guard**: The `web_fetch` tool blocks internal loopback and RFC 1918 private network address ranges by default.
4. **Panic Safety**: All Rust panics at the C FFI boundary are caught via `std::panic::catch_unwind` to prevent undefined behavior across foreign language boundaries.

---

## 🙏 Acknowledgments

PhoneBuddy SDK is built upon the open-source foundations established by the AI agent community. We would like to express our gratitude to:

- **[xAI](https://github.com/xai-org)** for open-sourcing **[grok-build](https://github.com/xai-org/grok-build)**, whose core agent harness architecture, task planning paradigms, and doom-loop detection algorithms serve as the upstream foundation for PhoneBuddy SDK.
- The **[Boa](https://github.com/boa-dev/boa)** team for providing a robust pure-Rust ECMAScript engine that powers our safe mobile code execution sandbox.
- The **[Rustls](https://github.com/rustls/rustls)** project for pure-Rust memory-safe TLS transport.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

