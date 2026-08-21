# Client Profile Emulation Guide (1:1 Official Client Replication)

PhoneBuddy SDK provides built-in, out-of-the-box **Client Profiles** (`ClientProfile`) to strictly replicate the network behavior of official AI coding agent clients (such as **xAI Grok Build**, **OpenAI Codex**, and **Anthropic Claude Code**) on a **1:1 basis**.

This ensures complete fidelity with upstream API gateways, authentication flows, backend-hosted tools, and server-side features.

---

## 1. Overview of Supported Profiles

| Profile | Wire Backend | User-Agent Specification | Distinctive Headers & Body |
| :--- | :--- | :--- | :--- |
| `grok_build` | `responses` (`/v1/responses`) | `grok-cli/{version} ({os}; {arch})` | <ul><li>`x-grok-client-identifier: grok-cli`</li><li>`x-grok-doom-loop-check: 1`</li><li>`Authorization: Bearer <key>`</li><li>Responses API with `reasoning` / `reasoning_text` blocks and native `{ "type": "web_search" }` tools</li></ul> |
| `codex` | `responses` (`/v1/responses`) | `codex-cli/{version} ({os}; {arch}) codex_cli` | <ul><li>`openai-beta: responses=true`</li><li>`session-id`, `thread-id`, `x-client-request-id`</li><li>`Authorization: Bearer <key>`</li><li>Standard OpenAI Responses payload</li></ul> |
| `claude_code` | `messages` (`/v1/messages`) | `claude-cli/{version} (external, cli)` | <ul><li>`anthropic-version: 2023-06-01`</li><li>`anthropic-beta: ccr-byoc-2025-07-29,prompt-caching-2024-07-31`</li><li>`x-app: cli`</li><li>`x-claude-code-session-id: <uuid>`</li><li>`x-api-key: <key>`</li><li>Thinking blocks (`type: "thinking"`, `signature`) and Anthropic tool schema (`input_schema`)</li></ul> |
| `default` | `chat_completions` (`/v1/chat/completions`) | `PhoneBuddy/{version} (Mobile SDK; ...)` | Standard OpenAI-compatible Chat Completions payload |

---

## 2. Multi-Language Usage Examples

### 2.1 Rust Core API (`phone-buddy`)

#### Using `EngineConfigBuilder` (Recommended)

```rust
use phone_buddy::prelude::*;

// 1. Claude Code 1:1 Preset
let claude_cfg = EngineConfig::for_claude_code("sk-ant-...", "claude-opus-5")
    .url("https://api.anthropic.com/v1") // custom endpoint or proxy
    .client_session_id("custom-session-uuid")
    .build()?;

// 2. Grok Build 1:1 Preset
let grok_cfg = EngineConfig::for_grok("xai-...", "grok-4.6")
    .url("https://api.x.ai/v1")
    .enable_web_search(true)
    .build()?;

// 3. Codex 1:1 Preset
let codex_cfg = EngineConfig::for_codex("sk-...", "o3-mini")
    .url("https://api.openai.com/v1")
    .build()?;

// 4. Fluent Custom Builder
let custom_cfg = EngineConfig::builder()
    .client_profile(ClientProfile::ClaudeCode)
    .url("https://my-custom-proxy.internal/v1")
    .api_key("sk-...")
    .model("claude-opus-5")
    .extra_header("X-Custom-Tenant", "mobile-app")
    .build()?;
```

---

### 2.2 Swift (iOS)

```swift
import PhoneBuddy

// Configure via JSON / Swift struct
let configJson = """
{
    "api_key": "sk-ant-...",
    "base_url": "https://api.anthropic.com/v1",
    "model": "claude-opus-5",
    "client_profile": "claude_code",
    "client_session_id": "\(UUID().uuidString)",
    "root_dir": "\(documentsPath)"
}
"""

let engine = try PhoneBuddyEngine(configJson: configJson)
```

---

### 2.3 Kotlin (Android)

```kotlin
val configJson = JSONObject().apply {
    put("api_key", "sk-ant-...")
    put("base_url", "https://api.anthropic.com/v1")
    put("model", "claude-opus-5")
    put("client_profile", "claude_code")
    put("client_session_id", UUID.randomUUID().toString())
    put("root_dir", context.filesDir.absolutePath)
}.toString()

val agent = NativeAgent(configJson)
```

---

### 2.4 C API / FFI

```c
#include "phone_buddy.h"

const char* config_json = "{"
    "\"api_key\": \"sk-ant-...\","
    "\"base_url\": \"https://api.anthropic.com/v1\","
    "\"model\": \"claude-opus-5\","
    "\"client_profile\": \"claude_code\","
    "\"root_dir\": \"/tmp/phone-buddy\""
"}";

char* err = NULL;
PbEngine* engine = pb_engine_new(config_json, &err);
if (!engine) {
    printf("Failed to init: %s\n", err);
    pb_string_free(err);
}
```

---

### 2.5 CLI Debugging

You can test emulation directly from the command line:

```bash
# Claude Code Emulation
PHONEBUDDY_CLIENT_PROFILE=claude-code \
PHONEBUDDY_API_KEY="sk-ant-..." \
PHONEBUDDY_MODEL="claude-opus-5" \
cargo run -p phone-buddy-cli -- chat "List files and summarize sales"

# Grok Build Emulation
PHONEBUDDY_CLIENT_PROFILE=grok-build \
PHONEBUDDY_API_KEY="xai-..." \
PHONEBUDDY_MODEL="grok-4.6" \
cargo run -p phone-buddy-cli -- chat "Search latest news on Rust"
```


---

---

## 3. Session ID Generation & Upstream Claude Code (`cc-src`) Architecture

According to the upstream Claude Code codebase (`cc-src/bootstrap/state.ts`), `client_session_id` is generated as follows:

1. **Generation Mechanism**:
   - Initialized at CLI bootstrap using Node's standard `crypto.randomUUID()` (UUID v4 format: `xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`).
   - Returned via `getSessionId()` in `bootstrap/state.ts`.
2. **HTTP Header Placement**:
   - Attached to all outgoing Messages requests via header: `'X-Claude-Code-Session-Id': getSessionId()`.
3. **Session Lifecycle & Subagents**:
   - When `/compact` or `clearConversation` occurs, `regenerateSessionId()` generates a fresh `randomUUID()`, recording `parentSessionId = STATE.sessionId`.
   - When subagents are spawned (`spawnMultiAgent.ts`), subagents receive a new UUID and pass the parent session ID via `--parent-session-id <parentSessionId>`.
4. **PhoneBuddy SDK Alignment**:
   - PhoneBuddy SDK automatically generates a UUID v4 for `client_session_id` if not explicitly specified by the host application.

---

## 4. Verification with `HttpDumper`

To verify 100% byte-level fidelity against official clients:

1. Enable HTTP dump in your config:
   ```json
   {
       "client_profile": "claude_code",
       "http_dump": {
           "mode": "all",
           "mask_sensitive": false
       }
   }
   ```
2. Run a turn and inspect `.phonebuddy/http_dumps/req_*.json`.
3. Compare the saved request headers (`user-agent`, `anthropic-version`, `anthropic-beta`, `x-app`, `x-claude-code-session-id`) and JSON body against the official client request logs.

