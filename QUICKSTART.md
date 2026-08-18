# PhoneBuddy SDK Quick Start

A 5-minute quick-start guide to integrate and use PhoneBuddy SDK.

## Prerequisites

- **iOS**: Xcode 14+, Swift 5.5+
- **Android**: Android Studio, Kotlin 1.8+, NDK r25+
- **Credentials**: a logged-in user's User Access Token (Bearer Token) and an enterprise AI gateway endpoint (`base_url`)

## iOS Integration (3 steps)

### 1. Add the library files

Copy the files into your Xcode project directory:
```bash
cp ios/libs/libphone_buddy_ffi-device.a YourProject/
cp ios/libs/libphone_buddy_ffi-sim.a YourProject/
cp ios/include/phone_buddy.h YourProject/
```

In Xcode:
- Target -> Build Phases -> Link Binary With Libraries
- Add both `.a` files

### 2. Configure the bridging header

Create `YourProject-Bridging-Header.h`:
```objc
#import "phone_buddy.h"
```

Build Settings -> Objective-C Bridging Header -> set it to that file path

### 3. Use the engine

Copy `examples/ios/PhoneBuddy.swift` into your project, then:

```swift
import Foundation

// 1. Obtain the logged-in client user's Access Token (never use a raw LLM API key client-side)
let userAccessToken = "bearer-user-jwt-token-here"

// 2. Initialize the configuration (base_url points to the enterprise AI proxy gateway, not the vendor directly)
let config = PhoneBuddyConfig(
    apiKey: userAccessToken, // pass the user login token
    baseUrl: "https://ai-gateway.yourcompany.com/v1", // enterprise AI proxy gateway
    model: "grok-4.6",
    apiBackend: "responses",
    rootDir: FileManager.default.temporaryDirectory.path,
    extraHeaders: [
        "X-App-Version": "2.4.0",
        "X-Client-Platform": "iOS"
        // Note: for user privacy protection, never transmit device IDs (IDFA/IMEI/Android ID)
    ]
)
let engine = try PhoneBuddyEngine(config: config)

// Run a task
let outcome = try await engine.chat(
    sessionId: "session-001",
    userInput: "Help me plan a weekend trip"
)

print(outcome.finalText)
```

Full example: `examples/ios/DemoApp.swift`

## Android Integration (3 steps)

### 1. Add the library files

Copy the JNI library into your project's `app/src/main/` directory:
```bash
cp -r android/jniLibs app/src/main/
```

### 2. Add the Kotlin wrapper

Copy `examples/android/NativeAgent.kt` into your package directory and update the package name.

### 3. Use the engine

```kotlin
import org.phonebuddy.NativeAgent
import org.json.JSONObject
import kotlinx.coroutines.*

// 1. Obtain the logged-in client user's Access Token
val userAccessToken = "bearer-user-jwt-token-here"

// 2. Initialize the configuration
val config = JSONObject().apply {
    put("api_key", userAccessToken) // pass the user login token
    put("base_url", "https://ai-gateway.yourcompany.com/v1") // enterprise AI proxy gateway
    put("model", "grok-4.6")
    put("api_backend", "responses")
    put("root_dir", context.cacheDir.resolve("phone-buddy").absolutePath)
    put("extra_headers", JSONObject().apply {
        put("X-App-Version", "2.4.0")
        put("X-Client-Platform", "Android")
    })
}
val agent = NativeAgent(config.toString(), context)

// Run a task (on the IO thread)
val resultJson = withContext(Dispatchers.IO) {
    agent.chat(
        sessionId = "session-001",
        userInput = "Help me plan a weekend trip"
    )
}

val outcome = JSONObject(resultJson)
println(outcome.getString("final_text"))

// Clean up
agent.close()
```

Full example: `examples/android/MainActivity.kt`

## Configuration Options

```json
{
  "api_key": "<User_Access_Token>",
  "base_url": "https://ai-gateway.yourcompany.com/v1",
  "model": "grok-4.6",
  "api_backend": "responses",
  "root_dir": "/path/to/work/dir",
  "max_turns": 10,
  "enable_web_search": false,
  "extra_headers": {
    "X-App-Version": "2.4.0",
    "X-Client-Platform": "iOS"
  }
}
```

## Examples

### Task planning
```swift
let outcome = try await engine.chat(
    sessionId: "s1",
    userInput: "Plan my work schedule for next week"
)
```

### Excel analysis
```swift
let outcome = try await engine.chat(
    sessionId: "s2",
    userInput: "Analyze this sales data spreadsheet and give recommendations"
)
```

### Streaming events
```swift
let outcome = try await engine.chat(
    sessionId: "s3",
    userInput: "...",
    onEvent: { eventJson in
        print("Progress: \(eventJson)")
    }
)
```

## FAQ

### iOS link errors
Make sure you use the correct library:
- Device: `libphone_buddy_ffi-device.a`
- Simulator: `libphone_buddy_ffi-sim.a`

### Android UnsatisfiedLinkError
Check that the `.so` files are in `app/src/main/jniLibs/<abi>/`

### Initialization failures
Check:
- Whether the User Access Token credentials and the AI gateway `base_url` are valid
- Whether the `root_dir` path is writable
- Whether the network connection is working

## Next Steps

- Full documentation: `README.md`
- iOS detailed guide: `examples/ios/README.md`
- Android detailed guide: `examples/android/README.md`
- API reference: `dist/ios/include/phone_buddy.h`
