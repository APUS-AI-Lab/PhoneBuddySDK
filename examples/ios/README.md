# PhoneBuddy SDK iOS Integration Example

This directory contains example code for integrating PhoneBuddy SDK on iOS.

## Files

- **PhoneBuddy.swift** - Swift wrapper layer with a type-safe API
- **PhoneBuddy-Bridging-Header.h** - Objective-C bridging header
- **DemoApp.swift** - complete SwiftUI demo app

## One-Click Build & Run (Simulator / iPhone)

To build the iOS app and install/launch it directly onto the iOS Simulator or a connected physical iPhone:

```bash
# Run from workspace root (Simulator)
./examples/ios/build-and-install.sh

# Target connected physical iPhone
./examples/ios/build-and-install.sh --device

# Or from this directory
./build-and-install.sh
```

---

## Manual Integration

### 1. Create an Xcode project

```bash
# Create a new iOS App project (SwiftUI)
# File -> New -> Project -> iOS -> App
# Name it PhoneBuddyDemo
```

### 2. Add the static libraries

Copy the static libraries from the SDK package's `ios/libs/` directory into your project:

- `libphone_buddy_ffi-device.a` (iOS device)
- `libphone_buddy_ffi-sim.a` (iOS simulator)

In Xcode:
1. Select the project -> Build Phases -> Link Binary With Libraries
2. Add `libphone_buddy_ffi-device.a` and `libphone_buddy_ffi-sim.a`

### 3. Configure the bridging header

1. Copy the C header from `ios/include/phone_buddy.h` into your project directory
2. Copy the bridging header from `ios/wrapper/PhoneBuddy-Bridging-Header.h` into your project directory
3. In Xcode's Build Settings, set `Objective-C Bridging Header` to that bridging header path

### 4. Add the Swift wrapper

Copy the native Swift wrapper from `ios/wrapper/PhoneBuddy.swift` into your project and start using it directly.

### 5. Write your app code

See the complete example in `DemoApp.swift`, or use this minimal example:

```swift
import Foundation

// 1. Create the config (pass the user login token and the proxy gateway)
let config = PhoneBuddyConfig(
    apiKey: "bearer-user-jwt-token-here", // client login credential (User Access Token)
    baseUrl: "https://ai-gateway.yourcompany.com/v1", // enterprise AI proxy gateway
    model: "grok-4.6",
    apiBackend: "responses", // uses the OpenAI Response API + SSE protocol
    rootDir: "/path/to/work/dir",
    maxTurns: 10,
    enableWebSearch: true,
    extraHeaders: [
        "X-App-Version": "2.4.0",
        "X-Client-Platform": "iOS"
    ]
)

// 2. Initialize the engine (registers a hidden WKWebView for web_search & web_fetch)
let engine = try PhoneBuddyEngine(config: config)

// 3. Run a conversation
let outcome = try await engine.chat(
    sessionId: "session-001",
    userInput: "Help me plan a weekend trip",
    onEvent: { eventJson in
        print("Event: \(eventJson)")
    }
)

print("Result: \(outcome.finalText)")
print("Turns: \(outcome.turnsUsed)")
```

## Running the Demo

1. Copy the contents of `DemoApp.swift` into your `ContentView.swift`
2. Fill in a logged-in user's `User Access Token`
3. Build and Run

## Architecture Support

- **Device**: `aarch64-apple-ios` (iPhone/iPad ARM64)
- **Simulator**: `aarch64-apple-ios-sim` + `x86_64-apple-ios` (M1/M2 Mac + Intel Mac)

The static libraries are processed with `strip -S`, keeping the symbol table for debugging.

## Configuration Options

```swift
public struct PhoneBuddyConfig {
    var apiKey: String              // user Access Token (Bearer Auth)
    var baseUrl: String             // enterprise AI proxy gateway URL
    var model: String               // model name ("grok-4.6")
    var apiBackend: String          // transport protocol ("responses": OpenAI Response API + SSE)
    var rootDir: String             // working directory (stores sessions, cache)
    var maxTurns: Int               // maximum number of conversation turns
    var enableWebSearch: Bool       // whether to enable web search
    var extraHeaders: [String: String]? // custom HTTP headers (e.g. X-App-Version)
    var extraBody: [String: String]?    // optional custom JSON body fields
    var httpDump: HttpDumpConfig?       // optional raw HTTP request/response dumper for diagnostics
}

public struct HttpDumpConfig {
    var mode: String                    // "off", "on_error", "all"
    var dumpDir: String?                // custom dump directory
    var maskSensitive: Bool             // whether to mask Authorization headers (default: true)
    var maxFiles: Int                   // max retained dump files (default: 30)
}
```

## API Reference

### PhoneBuddyEngine

```swift
// Initialization
init(config: PhoneBuddyConfig) throws

// Run a conversation (async)
func chat(
    sessionId: String,
    userInput: String,
    onEvent: ((String) -> Void)?
) async throws -> ChatOutcome

// Cancel execution
func cancel(sessionId: String)
```

### ChatOutcome

```swift
struct ChatOutcome {
    let finalText: String    // final response text
    let turnsUsed: Int       // number of conversation turns used
}
```

### Event callbacks

The `onEvent` callback receives a JSON event stream:

```json
{
  "event": "PlanGenerated",
  "data": { "plan": "..." }
}
```

Common event types:
- `PlanGenerated` - an execution plan was generated
- `ToolCall` - a tool was invoked
- `SearchStarted` - a search started
- `Thinking` - model is reasoning

## Dependencies

All dependencies are statically linked into the engine library; no extra configuration is required. Includes:
- Tokio async runtime
- reqwest + rustls (HTTPS)
- Boa JS engine (script execution)
- calamine (Excel parsing)

## Notes

1. **Auth security**: never hardcode any vendor API key client-side; obtain a User Access Token dynamically from your app's user login flow
2. **Gateway requirement**: all client requests must go through the enterprise AI gateway proxy and routing via `baseUrl`
3. **Background execution**: `chat()` runs on a background thread and event callbacks fire there too; switch to the main thread to update the UI
4. **File permissions**: `rootDir` needs read/write access; prefer the app's Documents or Caches directory
5. **Cancellation**: long-running tasks can be interrupted with `cancel()`

## Troubleshooting

### Link errors

If you hit `Undefined symbols`, check:
- Whether the static library was added correctly
- Whether you used `-sim.a` for the simulator and `-device.a` for the device
- Whether `Other Linker Flags` in Build Settings includes `-lc++` (if needed)

### Runtime errors

If engine initialization fails, check:
- Whether the `rootDir` path exists and is writable
- Whether the User Access Token credentials or the AI gateway `baseUrl` are valid
- Whether the network connection is working

### Simulator crashes

If it crashes on the simulator, make sure you linked `libphone_buddy_ffi-sim.a` instead of the device version.
