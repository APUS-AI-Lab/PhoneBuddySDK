# PhoneBuddy SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20Android%20%7C%20C%20ABI-green.svg)]()
[![Rust](https://img.shields.io/badge/Rust-1.94%2B-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

[English](README.md) | **简体中文**

> 专为移动操作系统（iOS 和 Android）打造的纯 Rust 轻量级嵌入式 LLM Agent 运行引擎。

**PhoneBuddy SDK** 为移动端应用提供全生命周期的自主 Agent 能力：任务动态拆解规划、纯 Rust 虚拟 Shell 命令、内置 JavaScript 执行沙盒与数据计算（`boa_engine`）、内存级异步子任务与调度管理、人机交互协同（Human-in-the-loop）以及实时流式事件回调——所有功能均严格遵循 iOS App Store 与 Google Play 的沙盒安全规范，**全程零外部子进程衍生（Zero Child Processes）**。

PhoneBuddy SDK 的核心 Agent 运行引擎抽离并移植自 xAI 的开源项目 **[xai-org/grok-build](https://github.com/xai-org/grok-build)**。在桌面端与云端领域，业界领先的 Agent Harness（如 **Claude Code**、OpenAI **[Codex](https://github.com/openai/codex)** 以及 xAI 的 **[grok-build](https://github.com/xai-org/grok-build)**）确立了自动化智能体 Harness 的最高水准。**PhoneBuddy SDK 将这套具备工业级成熟度的 Harness 机制完整带到了 iOS 与 Android 移动平台上。**

---

## 🌟 核心特性与工程亮点

- 🔒 **100% 移动沙盒合规**
  - 完全剔除 `std::process::Command`、`tokio::process` 及 `fork`/`exec` 等外部系统进程调用。
  - 零提权风险，完美契合 Apple App Store 和 Google Play 严格的应用上架审查规范。
- 🎯 **自主规划与动态 ReAct 循环**
  - 自动将用户复杂指令动态分解为清晰、分步的执行计划（`plan` 工具）。
  - 按序调度工具执行，实时感知中间结果并自主纠错与调整后续步骤，直至最终任务达成。
- 🧰 **纯 Rust 内存级虚拟 Shell（`busybox`）**
  - 内置纯 Rust 实现的 POSIX 常用命令集，全部在内存中执行：`cat`、`head`、`tail`、`ls`、`wc`、`sort`、`uniq`、`find`、`echo`、`touch`、`mkdir`、`rm`、`cp`、`mv`、`du`、`pwd`、`basename`、`dirname`（高级正则与文本检索由专属 `grep` 工具提供）。
- ⚡ **内置轻量 JavaScript 沙盒（`boa_engine`）**
  - 集成纯 Rust ECMAScript 引擎（`run_script` 工具），可在移动端 App 内安全执行数学运算、复杂算法、数据提取与 JSON 结构转换。
- 📊 **轻量数据分析与脚本运算**
  - 支持通过内置 JavaScript 沙盒（`run_script`）和文件工具链快速执行内存级数据过滤、统计汇总、数学计算与 CSV/JSON 结构化处理，无需外部运行环境。
- 📁 **严格越狱隔离的文件沙盒与检索**
  - 将所有文件读写严格限制在指定的沙盒根目录 `root_dir`（如应用 Documents 目录），严防路径穿越逃逸。
  - 提供完整的文件工具集：`read_file`（支持切片分段读取、字符集与混淆 Unicode 纠偏）、`write_file`、`edit_file`（智能代码/文本块替换）、`list_dir`（目录树递归扫描）与 `grep`（高效正则与 Glob 模式检索）。
- 🤖 **内存级异步子任务与并发协同**
  - 基于 Tokio 内存运行时并发拉起后台子任务（`task`）。
  - 支持完整的子任务生命周期管控：获取输出日志（`task_output`、`get_task_output`）、多任务同步等待（`wait_tasks`）、终止任务（`kill_task`）以及状态监控（`monitor`）。
- ⏰ **定时调度、系统通知与人机交互**
  - 内置内存级 Cron 定时器与延迟调度器（`scheduler`）。
  - 支持通过宿主回调触发系统 Push 通知与弹窗（`notification`）。
  - 支持交互式人机交互提问（`ask_user_question`），支持单选/多选及自由补充输入。
- 🌐 **联网搜索与防 SSRF 网页解析**
  - 支持多源网页搜索（`web_search`）：移动端静默调起系统 WebView 抓取 DuckDuckGo Lite，失败则走 LLM 搜索 API；`enable_web_search` 为真且协议为 Responses 时，对齐 Grok Build 在主请求 `tools` 中附加 hosted `{ "type": "web_search" }`。
  - 基于纯 Rust `htmd` 引擎将网页 HTML 高保真转换为 Clean Markdown（`web_fetch`），内置 SSRF 防护机制，默认拦截内网与私有 IP。
- 🛡️ **生产级 Harness 韧性机制**
  - **死循环检测（Doom-Loop Detection）**：工具调用静止性检测（`IdenticalToolCallRun`，8 轮微调提示 / 16 轮强制熔断）及服务端检查支持（`x-grok-doom-loop-check`）。
  - **上下文智能压缩（History Compaction）**：轻量 Token 估算，超出 `24,000` Tokens 时自动触发历史对话压缩。
  - **高可用重试机制**：带随机抖动（Jitter）的指数退避重试，自动应对 429 限流与 5xx 服务端波动。
- 🔌 **多协议支持与宿主扩展能力**
  - 支持多种 API 协议：`responses`（OpenAI Response API + SSE 流式传输）、`chat_completions` 与 `messages`。
  - 纯 Rust TLS 传输（`rustls-ring`）与 HTTP/2 流式通信。
  - 宿主 LLM 模式（`LlmMode::Host`）：支持将推理转发至宿主端本地模型（如 `llama.cpp` / `llama.rn`）或自定义企业代理。
  - 支持 Swift / Kotlin / C 宿主动态注入自定义原生工具（`PbHostToolCallback`）。
- 📱 **跨平台 C ABI 与原生开发语言封装**
  - 标准 C-ABI 接口（[phone_buddy.h](crates/phone-buddy-ffi/include/phone_buddy.h)），内部全面包裹 `catch_unwind` 保证跨语言内存安全。
  - 原生 Swift SDK（[PhoneBuddy.swift](examples/ios/PhoneBuddy.swift)）支持 `async`/`await` 与 SwiftUI。
  - 原生 Android Kotlin SDK（[NativeAgent.kt](examples/android/NativeAgent.kt) + JNI 桥接）支持 Jetpack Compose。
  - 提供本地命令行工具（`phone-buddy-cli`），便于开发、调试与离线自测。

---

## 🏗 架构总览

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 宿主应用层                                       │
│        iOS App (Swift / SwiftUI)          Android App (Kotlin / Compose)         │
│        C / C++ 原生应用程序                本地开发调试工具 (phone-buddy-cli)     │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┴────────────────────────────────────────┐
│                          平台集成与 C FFI 接口层                                 │
│  • phone_buddy.h (C ABI)                 • PhoneBuddy.swift (Swift Async/Await)  │
│  • phonebuddy_jni.c / NativeAgent.kt     • 宿主回调接口 (LLM / 自定义工具 / WebView)│
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┴────────────────────────────────────────┐
│                        核心 Agent 引擎 (phone-buddy)                             │
│                                                                                  │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐  │
│  │   Agent 决策循环     │ │    异步子任务系统    │ │      LLM 传输层        │  │
│  │ • 任务规划器 (plan)  │ │ • 内存级 TaskManager │ │ • Responses (SSE 流)   │  │
│  │ • 死循环熔断检测     │ │ • 并发子 Agent 协同  │ │ • ChatCompletions      │  │
│  │ • 历史上下文压缩     │ │ • Cron 调度器        │ │ • Messages             │  │
│  │ • 会话持久化管理     │ │ • 任务监控与回收     │ │ • Host LLM / rustls-ring│ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────────┘  │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        纯 Rust 工具库与安全沙盒                            │  │
│  │ • 文件沙盒：read_file, write_file, edit_file, list_dir, grep               │  │
│  │ • 内存级虚拟 Shell：纯 Rust BusyBox Applets (cat, head, sort, uniq, …)      │  │
│  │ • 内置 JS 引擎：boa_engine ECMAScript 沙盒 (run_script)                    │  │
│  │ • 数据分析：内置 JS 计算沙盒 (run_script)、CSV 与结构化数据处理            │  │
│  │ • 联网与检索：web_search (WebView / DDG), web_fetch (htmd), SSRF 安全防护  │  │
│  │ • 交互与扩展：ask_user_question (人机协同), notification, 宿主自定义工具   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. iOS 原生集成 (Swift)

将编译好的静态库 `libphone_buddy_ffi.a` 与头文件 `phone_buddy.h` 引入 Xcode 工程，或直接引入 [`PhoneBuddy.swift`](examples/ios/PhoneBuddy.swift)：

```swift
import Foundation

// 1. 初始化配置
let config = PhoneBuddyConfig(
    apiKey: "your-api-key-or-jwt-token",
    baseUrl: "https://api.x.ai/v1",
    model: "grok-4.6",
    apiBackend: "responses", // 使用 OpenAI Response API + SSE 协议
    rootDir: PhoneBuddyConfig.sandboxRoot(workspaceName: "workspace"),
    maxTurns: 24,
    enableWebSearch: true,
    agentName: "小智", // 系统提示词中的身份名；省略则保持 "PhoneBuddy"
    extraHeaders: [
        "X-App-Version": "1.0.0",
        "X-Client-Platform": "iOS"
    ]
)

// 2. 创建引擎实例
let engine = try PhoneBuddyEngine(config: config)
// engine.setAgentName("小智") // 运行时改名；nil/空字符串回退为 PhoneBuddy

// 3. 发起对话并监听实时流式事件
let outcome = try await engine.chat(
    sessionId: "session-001",
    userInput: "分析 data/sales.csv 中的季度销售数据并计算各区域总利润",
    onEvent: { eventJson in
        print("实时事件: \(eventJson)")
    }
)

print("最终报告: \(outcome.finalText)")
print("消耗轮数: \(outcome.turnsUsed)")
```

*完整 iOS 示例项目请参见 [examples/ios/](examples/ios/)*

---

### 2. Android 原生集成 (Kotlin)

将 `libphone_buddy_ffi.so` 放入 `app/src/main/jniLibs/<abi>/` 目录，并引入 [`NativeAgent.kt`](examples/android/NativeAgent.kt)：

```kotlin
import org.phonebuddy.NativeAgent
import org.phonebuddy.EventListener
import org.json.JSONObject

// 1. 构建配置 JSON
val config = JSONObject().apply {
    put("api_key", "your-api-key-or-jwt-token")
    put("base_url", "https://api.x.ai/v1")
    put("model", "grok-4.6")
    put("api_backend", "responses")
    put("root_dir", context.filesDir.resolve("workspace").absolutePath)
    put("max_turns", 24)
    put("enable_web_search", false)
    put("agent_name", "小智")
    put("extra_headers", JSONObject().apply {
        put("X-App-Version", "1.0.0")
        put("X-Client-Platform", "Android")
    })
}

// 2. 创建 Native Agent 实例
val agent = NativeAgent(config.toString(), context)
// agent.setAgentName("小智") // 运行时改名；null/空白回退为 PhoneBuddy

// 3. 执行任务对话
val resultJson = agent.chat(
    sessionId = "session-001",
    userInput = "分析 data/sales.csv 中的季度销售数据并计算各区域总利润",
    listener = object : EventListener {
        override fun onEvent(eventJson: String) {
            println("实时事件: $eventJson")
        }
    }
)

val outcome = JSONObject(resultJson ?: "{}")
println("最终报告: ${outcome.optString("final_text")}")
```

*完整 Android 示例项目请参见 [examples/android/](examples/android/)*

---

### 3. 标准 C API 接口 (`phone_buddy.h`)

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
        fprintf(stderr, "引擎初始化失败: %s\n", err);
        pb_string_free(err);
        return 1;
    }

    char *result = pb_engine_chat(engine, "session-001", "你好 Agent", on_event, NULL, &err);
    if (result) {
        printf("执行结果: %s\n", result);
        pb_string_free(result);
    } else {
        fprintf(stderr, "对话出错: %s\n", err);
        pb_string_free(err);
    }

    pb_engine_free(engine);
    return 0;
}
```

*完整 C 示例请参见 [examples/c_demo/](examples/c_demo/)*

---

### 4. 长生命周期 Runtime 与一次性文本生成 `generate_text`

`PhoneBuddyRuntime` 持有路由池、Provider 健康度以及重试/故障转移策略，其生命周期长于单个 `PhoneBuddyEngine`。基于同一个 Runtime 重建 Engine 时，冷却状态会被完整保留。

`generate_text` 是一个无工具、无会话的一次性调用。它复用 SDK 的路由器、协议适配器、HTTP Dump 与用量解析，但**不会**运行 Agent 循环、上下文压缩或任何工具。调用方需自行指定池 ID（例如 `session_title`）；池不存在时返回 `RouteNotConfigured`，不会隐式回落到 `main`。

```rust
use phone_buddy::prelude::*;
use tokio_util::sync::CancellationToken;

let runtime = PhoneBuddyRuntime::new(routing_config, root_dir)?;
let engine = runtime.create_engine(agent_config, "main")?;
let title = runtime.generate_text_blocking(
    GenerateTextRequest {
        pool_id: "session_title".into(),
        instructions: Some("Return a short conversation title.".into()),
        input: transcript.into(),
        max_output_tokens: Some(32),
        temperature: Some(0.2),
        reasoning_effort: None,
        // 可选的结构化输出。Anthropic `Messages` 协议无法表达该约束，
        // 会直接返回 `ResponseFormatUnsupported`，而不是回一段还需自行
        // 当作 JSON 解析的散文。
        response_format: Some(ResponseFormat::JsonObject),
        timeout_ms: Some(8_000),
    },
    CancellationToken::new(),
)?;
println!("{} via {} / {}", title.text, title.provider_id, title.model);
```

每次经过路由的请求都会上报所属工作负载（`main`、`subagent` 或 `one_shot`）：既出现在 `GenerateTextResult` 上，也出现在 `Retrying` / `ProviderSwitched` 事件中。因此即使多个池共用同一个 `provider_id`，仍然可以定位到究竟是哪类工作把它打挂的。

C FFI（异步完成回调 + 取消；与聊天会话回调相互独立）：

```c
PbRuntime *rt = pb_runtime_new(routing_json, root_dir, &err);
PbEngine *engine = pb_engine_new_with_runtime(rt, agent_config_json, "main", &err);
char *op = pb_runtime_generate_text_async(rt, request_json, on_done, user, &err);
/* on_done 收到 {"version":1,"ok":true,"operation_id":"op_...","result":{...}} */
pb_runtime_cancel_operation(rt, op);
pb_runtime_free(rt);
```

`pb_engine_new` 仍作为兼容路径保留：它会依据 primary + `fallback_providers` 合成一个私有 Runtime。

---

### 5. 本地开发与命令行测试 (`phone-buddy-cli`)

```bash
# 1. 运行离线 Script 演示（无需 API Key 即可体验销售数据分析）
cargo run -p phone-buddy-cli -- mock

# 2. 运行内置工具自测（直接测试 BusyBox、JS 引擎及文件检索）
cargo run -p phone-buddy-cli -- self-test

# 3. 运行在线真实 LLM 对话
PHONEBUDDY_API_KEY="your-api-key" cargo run -p phone-buddy-cli -- chat "分析销售数据"

# 4. 无工具的一次性文本生成（未指定 --pool 时使用合成出来的 `main` 池）
PHONEBUDDY_API_KEY="your-api-key" cargo run -p phone-buddy-cli -- generate "给这段对话起个标题"

# 5. 同上，但约束模型只返回 JSON 对象
PHONEBUDDY_API_KEY="your-api-key" cargo run -p phone-buddy-cli -- generate --json "给这段对话起个标题"
```

---

## 🎯 1:1 官方客户端仿真 (`ClientProfile`)

PhoneBuddy SDK 支持对主流桌面 AI 编程 Agent 客户端（**xAI Grok Build**、**OpenAI Codex**、**Anthropic Claude Code**）进行 **1:1 深度仿真**，严格对齐其 HTTP Headers（`User-Agent`、`anthropic-version`、`anthropic-beta`、`x-grok-*`、`session-id`）、Thinking 思考链签名及 JSON Wire 协议体。

```rust
use phone_buddy::prelude::*;

// 链式构建 1:1 Claude Code 预设：
let config = EngineConfig::for_claude_code("sk-ant-...", "claude-opus-5")
    .url("https://api.anthropic.com/v1")
    .build()?;
```

*完整协议字段矩阵与各语言使用示例请参考：[**docs/client_profiles.md**](docs/client_profiles.md)。*

---

## 📡 实时流式事件回调

在 Agent 执行过程中，引擎会通过回调接口向宿主实时推送结构化 JSON 事件：

| 事件类型 | 字段载荷 | 说明 |
| :--- | :--- | :--- |
| `TextDelta` | `{"text": "..."}` | 助手文本 Token 流式增量输出 |
| `ReasoningDelta` | `{"text": "..."}` | 思考/推理过程流式增量（支持思考模型） |
| `ToolCallStart` | `{"call_id": "...", "name": "...", "arguments_json": "..."}` | 工具即将开始执行时触发 |
| `ToolCallResult` | `{"call_id": "...", "name": "...", "ok": true, "output": "..."}` | 工具执行完毕并返回结果时触发 |
| `PlanUpdated` | `{"items_json": "..."}` | 规划工具生成或更新任务计划列表时触发 |
| `Completed` | `{"final_text": "...", "usage": { ... }}` | 单轮对话及工具循环正常结束时触发 |
| `Failed` | `{"message": "..."}` | 执行过程中发生不可恢复异常时触发 |

---

## ⚙️ 配置参数参考

| 参数名 | 类型 | 是否必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `api_key` | String | HTTP 模式必填 | `""` | 用户 API Key 或 Bearer 鉴权 Token |
| `base_url` | String | HTTP 模式必填 | `https://api.x.ai/v1` | OpenAI / xAI / Anthropic 兼容的 Endpoint URL |
| `model` | String | 是 | `grok-4.6` | 目标模型标识（如 `grok-4.6`, `claude-opus-5`, `gpt-4o`） |
| `client_profile` | String | 否 | `default` | 1:1 客户端仿真预设：`grok_build`, `codex`, `claude_code`, `default`。详见 [docs/client_profiles.md](docs/client_profiles.md) |
| `client_version` | String | 否 | `null` | 自定义 `User-Agent` 报告的客户端版本号 |
| `client_session_id` | String | 否 | `null` | 自定义厂商会话 UUID（`x-claude-code-session-id`, `session-id`） |
| `root_dir` | String | 是 | `/tmp/phone-buddy` | 文件沙盒隔离根目录及会话持久化存储路径 |
| `api_backend` | String | 否 | `chat_completions` | API 协议：`responses` (SSE 流), `chat_completions`, `messages` |
| `llm_mode` | String | 否 | `http` | 通信模式：`http` (直接网络请求) 或 `host` (宿主 FFI 回调桥接) |
| `locale` | String | 否 | `zh` | UI 语言环境，提示模型回答所用的语言 |
| `max_turns` | Integer | 否 | `24` | 单轮用户任务的最大工具调用循环轮数 |
| `temperature` | Float | 否 | `0.2` | 模型采样温度 |
| `reasoning_effort` | String | 否 | `null` | 思考/推理强度级别（`low`, `medium`, `high`, `xhigh`, `max`, `minimal`, `none`）。自动跨 Responses、ChatCompletions 及 Messages 协议适配 |
| `max_output_tokens` | Integer | 否 | `8192` | 单次输出的最大 Token 限制 |
| `enable_web_search` | Boolean | 否 | `false` | Responses 主请求是否附加 Grok hosted `{type: web_search}`（对齐 Grok Build `supports_backend_search`）。客户端 DuckDuckGo / `web_search` function tool 独立注册。PackyAPI 等不支持 hosted search 的网关请保持关闭。 |
| `enable_x_search` | Boolean | 否 | `false` | Responses 主请求是否附加 Grok hosted `{type: x_search}`，支持原生 X/Twitter 推文与 Thread 检索（`x_thread_fetch`、`x_keyword_search` 等）。 |
| `x_search_options` | Object | 否 | `null` | `x_search` 的可选配置（例如 `{"from_date": "2026-01-01", "to_date": "2026-08-28"}` 指定时间范围过滤）。 |
| `agent_name` | String | 否 | `PhoneBuddy` | 系统提示词中的身份名（`You are {agent_name}…`）。空值回退为 `PhoneBuddy` |
| `system_prompt_extra` | String | 否 | `null` | 追加到 System Prompt 尾部的自定义人设或业务指令 |
| `stream_idle_timeout_secs`| Integer | 否 | `300` | 流式连接空闲超时时间（秒，对齐 grok-build） |
| `max_retries` | Integer | 否 | `5` | HTTP 请求发生错误时的指数退避最大重试次数 |
| `extra_headers` | Object | 否 | `{}` | 附加到 LLM 请求的自定义 HTTP Header 字典 |
| `extra_body` | Object | 否 | `{}` | 透传合并到 LLM 请求 JSON 体顶层的自定义字段 |
| `enable_doom_loop_check`| Boolean | 否 | 自动判定 | 是否开启服务端死循环检测 Header（`x-grok-doom-loop-check`） |
| `web_fetch_allow_local` | Boolean | 否 | `false` | 是否允许 `web_fetch` 请求本地回环地址（仅供测试） |
| `http_dump` | Object | 否 | `{"mode":"off"}`| 原始 HTTP 请求/响应报文落盘诊断配置（如 `{"mode":"on_error","max_files":30}`） |


---

## 🩺 HTTP 流量转储与网络排错（诊断 HTTP 异常与网络故障）

当接入客户端遇到网络连通性异常、鉴权失败、网关代理报错或非预期的 HTTP 状态码（如 4xx / 5xx）时，可通过配置 `http_dump` 将完整的底层 HTTP 请求及响应报文保存到本地沙盒，便于快速排查与定位问题：

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

### `http_dump` 参数说明：
* **`mode`**：
  * `"off"` *(默认)*：关闭报文转储。
  * `"on_error"`：仅当 HTTP 请求失败（返回非 2xx 状态码或发生连接超时/网络故障）时转储。
  * `"all"`：转储所有 HTTP 往返交互（包含 200 成功握手与失败报文）。
* **`dump_dir`**：自定义转储目录。默认路径为 `<root_dir>/.phonebuddy/http_dumps/`。
* **`mask_sensitive`**：是否对 `Authorization`、`x-api-key`、`cookie` 等敏感 Header 自动脱敏掩码。默认为 `true`。
* **`max_files`**：保留的最大 Dump JSON 文件数量，超出自动按修改时间删除最旧文件（FIFO 轮转），防止占用手机存储。默认为 `30`。

### 报错提示与 Dump JSON 格式：
开启后，当触发网络或接口报错时，抛出的错误信息尾部会自动携带对应的 Dump 文件绝对路径：
```text
Completion error: Error: LLM request failed: status=500 { "error": ... } [HTTP dump: /data/user/0/com.example.app/files/sandbox/.phonebuddy/http_dumps/dump_20260821_211530_500_req_a1b2c3.json]
```

生成的 JSON 包含完整的上下文：
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

## 📁 代码仓库目录结构

```text
PhoneBuddySDK/
├── Cargo.toml                  # Cargo Workspace 配置
├── crates/
│   ├── phone-buddy/            # 核心 Rust Agent 引擎（规划、工具集、沙盒、子任务）
│   ├── phone-buddy-ffi/        # C ABI 导出层（生成导出 phone_buddy.h）
│   └── phone-buddy-cli/        # 本地命令行开发与测试工具
├── dist/                       # 编译构建输出产物
│   ├── ios/                    # iOS 静态库 (.a) 及头文件
│   └── android/                # Android 动态库 (.so) 及头文件
├── examples/                   # 移动端原生 Demo 与示例代码
│   ├── ios/                    # Swift 封装层 (PhoneBuddy.swift) 及 SwiftUI Demo
│   ├── android/                # Kotlin 封装层 (NativeAgent.kt)、JNI 桥接及 Compose Demo
│   └── c_demo/                 # 标准 C API 使用范例 (main.c)
└── scripts/                    # 交叉编译与自动化打包脚本
    ├── build-ios-sdk.sh        # iOS 静态库编译脚本 (aarch64 / simulator universal)
    ├── build-android-sdk.sh    # Android 动态库编译脚本 (arm64-v8a / x86_64)
    └── package-sdk.sh          # 一键 SDK ZIP 分发打包脚本
```

---

## 🔨 源码编译与构建

### 环境准备

- Rust `1.94+` 并安装对应移动端 Target：
  ```bash
  rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
  rustup target add aarch64-linux-android x86_64-linux-android
  ```
- Xcode 14+（用于 iOS 构建）
- Android NDK r25+（用于 Android 构建，需配置 `ANDROID_NDK_HOME`）

### 编译指令

```bash
# 检查 Rust 代码编译
cargo check

# 运行全量单元测试与集成测试
cargo test

# 交叉编译 iOS 静态库（真机 + 模拟器）
./scripts/build-ios-sdk.sh

# 交叉编译 Android 动态链接库 (.so)
./scripts/build-android-sdk.sh

# 一键编译并在 iOS 模拟器/真机上安装运行 Demo：
./examples/ios/build-and-install.sh [--simulator | --device]

# 一键编译并在 Android 手机上安装运行 Demo：
./examples/android/build-and-install.sh [--logs]

# 一键打包 SDK 最终分发交付件至 dist/：
./scripts/package-sdk.sh --build
```

---

## 🔒 移动端安全与沙盒规范

1. **零外部子进程保证**：所有操作（BusyBox 命令、JavaScript 执行、文件修改）全部在当前进程内由纯 Rust 驱动，绝不发起 `fork`、`exec` 或系统 Shell 进程。
2. **严格文件隔离**：文件沙盒层严格拦截所有 `../` 路径穿越攻击，确保读写范围被严格限定在 `root_dir` 内。
3. **SSRF 安全防御**：`web_fetch` 默认拦截内网与私有 IP 地址范围（RFC 1918），防范内网探测。
4. **跨语言 Panic 防护**：C FFI 边界全面封装 `std::panic::catch_unwind`，严防 Rust Panic 穿透至 C/Swift/Kotlin 导致 Undefined Behavior。

---

## 🙏 致谢与鸣谢

PhoneBuddy SDK 得益于开源 AI 智能体社区的技术积累与先驱探索。我们由衷感谢：

- **[xAI](https://github.com/xai-org)** 开源的 **[grok-build](https://github.com/xai-org/grok-build)** 项目，其优秀的 Agent Harness 架构、长程任务规划范式与死循环检测算法为 PhoneBuddy SDK 提供了核心设计基石与上游渊源。
- **[Boa](https://github.com/boa-dev/boa)** 团队打造的高性能纯 Rust ECMAScript 引擎，支撑了移动端轻量安全的无进程 JS 沙盒计算。
- **[Rustls](https://github.com/rustls/rustls)** 项目提供的纯 Rust 内存安全 TLS 传输能力。

---

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可证开源。

