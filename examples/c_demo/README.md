# C Agent CLI Demo

This example demonstrates a complete, interactive AI Agent application written in C that uses the PhoneBuddy SDK native C API.

## Features

1. **Configuration File Loading**: Reads configuration from `config.json` (or a custom path passed as `argv[1]`). No hardcoded fallback values are used; if the file is missing or invalid, an explicit error is reported and the program exits.
2. **Interactive Console REPL**: Continuous conversation loop reading user input from `stdin`, executing agent turns, streaming reasoning/text output, displaying tool executions in real-time, and continuing to the next prompt until typing `exit` or pressing `Ctrl+C`.
3. **Rich TUI Progress Output**: Displays live intermediate execution progress similar to the `grok-build` / `phone-buddy-cli` TUI:
   - 💭 **Thinking / Reasoning**: Streaming model reasoning deltas in dim/italic text.
   - ⚙️ **Tool Calls**: Real-time tool invocation with parameters preview (`ToolCallStart`).
   - ✓ / ✗ **Tool Results**: Execution outcome and formatted output preview (`ToolCallResult`).
   - 📋 **Plan Updates**: Dynamic task planning / todo list status (`PlanUpdated`).
   - ❓ **Host Interactions**: Support for `ask_user_question` clarifying prompts from the agent.
4. **Graceful Signal Handling**: Intercepts `SIGINT` (Ctrl+C). If pressed during an in-flight agent turn, cancels the current turn via `pb_engine_cancel`; if pressed at the prompt, exits cleanly and frees engine resources.

---

## Files

- **[main.c](file:///Users/hyattjackson/Projects/PhoneBuddySDK/examples/c_demo/main.c)** - Complete interactive C Agent implementation with built-in lightweight JSON parser and streaming TUI.
- **[config.json.example](file:///Users/hyattjackson/Projects/PhoneBuddySDK/examples/c_demo/config.json.example)** - Example configuration file template.
- **[build.sh](file:///Users/hyattjackson/Projects/PhoneBuddySDK/examples/c_demo/build.sh)** - Automated build script with Rust FFI compilation and run support.
- **[Makefile](file:///Users/hyattjackson/Projects/PhoneBuddySDK/examples/c_demo/Makefile)** - Standard Makefile for macOS and Linux.

---

## Building and Running

### 1. Configure `config.json`

Copy `config.json.example` to `config.json` and fill in your API key, model, and base URL:

```bash
cp config.json.example config.json
```

Example `config.json`:
```json
{
  "api_key": "xai-your-api-key-here",
  "base_url": "https://api.x.ai/v1",
  "model": "grok-3",
  "root_dir": "./workspace",
  "api_backend": "chat_completions",
  "max_turns": 24,
  "enable_web_search": false
}
```

C hosts have no system WebView. If `enable_web_search` is true, `web_search` skips DuckDuckGo scraping and uses the configured LLM search API.

### 2. Build the Static Library and C Demo

You can build using either the automated shell script or `make`:

```bash
# Option A: Build with the build script (automatically builds phone-buddy-ffi)
./build.sh

# Or build in debug mode:
./build.sh --debug

# Or build and run immediately:
./build.sh --run

# Option B: Build with make
make demo
```

### 3. Run the Agent

```bash
# Run with default config.json and dynamically generate a fresh session UUID
./demo

# Resume a previous session by UUID
./demo -r <session_uuid>
./demo --resume <session_uuid>

# Or specify a custom configuration file path
./demo /path/to/my_config.json
./demo --config /path/to/my_config.json --resume <session_uuid>
```

---

## Interactive Commands

Inside the REPL:
- `/resume <uuid>`: Switch to or resume an existing session history.
- `/new`: Start a fresh session with a newly generated UUID.
- `/sessions`: List all persisted sessions and message counts.
- `clear` or `/clear`: Clear the terminal screen.
- `help` or `/help`: Display available REPL commands.
- `exit` or `quit`: Terminate the agent session.
- `Ctrl+C`: Cancel currently running turn or exit program.
