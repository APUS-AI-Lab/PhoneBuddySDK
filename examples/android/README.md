# PhoneBuddy SDK Android Integration Example

This directory contains example code for integrating PhoneBuddy SDK on Android.

## Files

- **NativeAgent.kt** - Kotlin JNI wrapper class
- **phonebuddy_jni.c** - JNI bridge C code
- **MainActivity.kt** - complete Jetpack Compose demo app

## One-Click Build APK & Install to Phone

To build the APK and install it directly onto your connected Android phone (via ADB) with a single command:

```bash
# Run from workspace root
./examples/android/build-and-install.sh

# Or from this directory
./build-and-install.sh

# To stream logcat logs after launch:
./examples/android/build-and-install.sh --logs
```

---

## Manual Integration

### 1. Create an Android project

```bash
# Create a new project with Android Studio
# File -> New -> New Project -> Empty Compose Activity
# Name it PhoneBuddyDemo, choose Kotlin + Jetpack Compose
```

### 2. Add the native libraries (.so)

Copy the library files from the SDK package's `android/jniLibs/` directory into your project's `app/src/main/` directory:

```bash
cp -r android/jniLibs app/src/main/
```

The directory layout should be:
```
app/src/main/
├── jniLibs/
│   ├── arm64-v8a/
│   │   └── libphone_buddy_ffi.so
│   └── x86_64/
│       └── libphone_buddy_ffi.so
```

### 3. Add the Kotlin wrapper

Copy `android/wrapper/NativeAgent.kt` from the SDK package into your Android project's package directory:

```bash
cp android/wrapper/NativeAgent.kt app/src/main/java/org/phonebuddy/
```

**Important**: change the `package` declaration at the top of `NativeAgent.kt` to your project's actual package name.

### 4. (Optional) Using the JNI bridge

If you need custom JNI bindings, use `phonebuddy_jni.c`:

Add to `app/build.gradle.kts`:

```kotlin
android {
    ...
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}
```

Create `app/src/main/cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project("phonebuddy-jni")

add_library(phonebuddy-jni SHARED phonebuddy_jni.c)

# Link the prebuilt .so
add_library(phone_buddy_ffi SHARED IMPORTED)
set_target_properties(phone_buddy_ffi PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libphone_buddy_ffi.so
)

target_link_libraries(phonebuddy-jni phone_buddy_ffi log)
```

### 5. Write your app code

See the complete example in `MainActivity.kt`, or use this minimal example:

```kotlin
import org.phonebuddy.NativeAgent
import org.phonebuddy.EventListener
import org.json.JSONObject
import kotlinx.coroutines.*

// 1. Create the config JSON (pass the user login token and the proxy gateway)
val config = JSONObject().apply {
    put("api_key", "bearer-user-jwt-token-here") // client login credential (User Access Token)
    put("base_url", "https://ai-gateway.yourcompany.com/v1") // enterprise AI proxy gateway
    put("model", "grok-4.6")
    put("api_backend", "responses") // uses the OpenAI Response API + SSE protocol
    put("root_dir", context.cacheDir.resolve("phone-buddy").absolutePath)
    put("max_turns", 10)
    put("enable_web_search", true)
    put("extra_headers", JSONObject().apply {
        put("X-App-Version", "2.4.0")
        put("X-Client-Platform", "Android")
    })
}

// 2. Initialize the engine (pass Context to enable system WebView search)
val agent = NativeAgent(config.toString(), context)

// 3. Run a conversation (must be on the IO thread)
val resultJson = withContext(Dispatchers.IO) {
    agent.chat(
        sessionId = "session-001",
        userInput = "Help me plan a weekend trip",
        listener = object : EventListener {
            override fun onEvent(eventJson: String) {
                println("Event: $eventJson")
            }
        }
    )
}

// 4. Parse the result
val outcome = JSONObject(resultJson)
val finalText = outcome.getString("final_text")
val turnsUsed = outcome.getInt("turns_used")

println("Result: $finalText")
println("Turns: $turnsUsed")

// 5. Clean up
agent.close()
```

## Running the Demo

1. Copy the contents of `MainActivity.kt` into your project
2. Add the Material Icons dependency to `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("androidx.compose.material:material-icons-extended:1.5.4")
    // ... other dependencies
}
```

3. Fill in a logged-in user's `User Access Token`
4. Build and Run

## Architecture Support

The current build includes the following ABIs:
- **arm64-v8a**: 64-bit ARM (modern Android devices)
- **x86_64**: 64-bit x86 (emulators)

To support more architectures, modify the `ABIS` array in `scripts/build-android.sh`.

## Configuration Options

The config is passed as a JSON string:

```json
{
  "api_key": "<User_Access_Token>",
  "base_url": "https://ai-gateway.yourcompany.com/v1",
  "model": "grok-4.6",
  "api_backend": "responses",
  "root_dir": "/data/data/com.yourapp/cache/phone-buddy",
  "max_turns": 10,
  "enable_web_search": false,
  "extra_headers": {
    "X-App-Version": "2.4.0",
    "X-Client-Platform": "Android"
  }
}
```

## API Reference

### NativeAgent

```kotlin
class NativeAgent(configJson: String, context: Context? = null) : AutoCloseable {
    // Run a conversation (blocking; must be called on the IO thread)
    fun chat(
        sessionId: String,
        userInput: String,
        listener: EventListener? = null
    ): String?

    // Register a hidden system WebView for web_search (also done when context is passed)
    fun enableSystemWebView(context: Context)

    // Shut down the engine
    override fun close()
}
```

### EventListener

```kotlin
interface EventListener {
    fun onEvent(eventJson: String)
}
```

### Return value format

`chat()` returns a JSON string:

```json
{
  "final_text": "this is the final response...",
  "turns_used": 3,
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 200
  }
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

## Permissions

Add the network permissions to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

## Dependencies

All dependencies are statically linked into the engine library; no extra configuration is required. Includes:
- Tokio async runtime
- reqwest + rustls (HTTPS)
- Boa JS engine (script execution)
- calamine (Excel parsing)

## Threading Model

⚠️ **Important**: `chat()` is a blocking call and **must** run on a background thread:

```kotlin
// ✅ Correct
lifecycleScope.launch(Dispatchers.IO) {
    val result = agent.chat(sessionId, userInput)
    withContext(Dispatchers.Main) {
        // Update the UI
    }
}

// ❌ Wrong (blocks the main thread)
val result = agent.chat(sessionId, userInput)
```

Event callbacks fire on a native thread; switch to the main thread to update the UI:

```kotlin
listener = object : EventListener {
    override fun onEvent(eventJson: String) {
        runOnUiThread {
            // Update the UI
        }
    }
}
```

## ProGuard Configuration

If code obfuscation is enabled, add to `proguard-rules.pro`:

```proguard
# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep PhoneBuddy classes
-keep class org.phonebuddy.** { *; }
```

## Notes

1. **Auth security**: never hardcode any vendor API key client-side; obtain a User Access Token dynamically from your app's user login flow
2. **Gateway proxy**: all LLM interaction requests must go through the enterprise AI authenticated routing gateway (`base_url`)
3. **File permissions**: prefer `context.cacheDir` or `context.filesDir` for `root_dir`
4. **Memory management**: call `close()` when done to release native resources
5. **Network status**: check network connectivity before calling the engine

## Troubleshooting

### UnsatisfiedLinkError

If the library cannot be found:

```
java.lang.UnsatisfiedLinkError: dlopen failed: library "libphone_buddy_ffi.so" not found
```

Check:
- Whether the `.so` files are in the correct `jniLibs/<abi>/` directory
- Whether the APK contains libraries for the target architecture (Build -> Analyze APK)
- Whether `ndk.abiFilters` in `build.gradle.kts` is configured correctly

### Crashes or native crashes

If a native crash occurs, check logcat:

```bash
adb logcat | grep -E "(FATAL|DEBUG)"
```

Common causes:
- Calling `chat()` on the main thread
- Malformed config JSON
- `root_dir` does not exist or is not writable

### Emulator issues

If the app fails to run on an emulator, make sure:
- The emulator ABI is `x86_64` or `arm64-v8a`
- The matching `.so` has been built
- On M1/M2 Macs, ARM64 emulators perform better

## Build Optimizations

### Reducing APK size

If the APK is too large, you can:

1. Include only the ABIs you need:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a")  // arm64 only
        }
    }
}
```

2. Ship with App Bundle distribution:

```bash
./gradlew bundleRelease
```

3. Enable code obfuscation:

```kotlin
buildTypes {
    release {
        isMinifyEnabled = true
        proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
}
```

Current library sizes:
- arm64-v8a: ~8MB
- x86_64: ~9MB
