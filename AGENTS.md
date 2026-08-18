# PhoneBuddy SDK - AI Agent Guide

## 1. Project Origin & Background

**PhoneBuddy SDK** is a lightweight LLM Agent engine library designed for mobile platforms (iOS and Android).

* **Project origin**: The core Agent engine is extracted and ported from the upstream open-source project `../grok-build` (the core Agent runtime library of xAI grok-build).
* **Mobile adaptation & porting notes**:
  * **Decoupling and removal of native dependencies**: Dependencies unsuitable for iOS/Android cross-compilation and mobile sandbox environments were removed, such as `jemalloc`, `git2`, `tree-sitter`, `tokio::process`, `gcloud-storage`, and `tonic`.
  * **Pure Rust replacements**: The pure-Rust `boa_engine` serves as the built-in JavaScript code sandbox engine, and `rustls-ring` provides secure TLS transport. Subagent task execution (`task`, `task_output`, `kill_task`, `wait_tasks`) is handled via an in-memory Tokio task manager.
  * **Process & sandbox constraints**: `tokio::process` and all child-process fork/exec calls were completely removed, satisfying iOS's strict app sandbox requirements and Apple/Google review policies.

---

## 2. Repository Structure

```text
PhoneBuddySDK/
├── Cargo.toml                  # Cargo workspace root configuration
├── crates/
│   ├── phone-buddy/            # Core Rust Agent engine library (task planning, tool calling, LLM streaming, JS sandbox, subagent tasks)
│   ├── phone-buddy-ffi/        # C-ABI interface layer exported via C FFI (exports phone_buddy.h)
│   └── phone-buddy-cli/        # Local CLI development, validation, and testing tool
├── dist/                       # Compiled build output directory
│   ├── ios/                    # iOS static library (.a) and phone_buddy.h
│   └── android/                # Android shared library (.so) and phone_buddy.h
├── examples/                   # Platform wrappers and demo applications
│   ├── ios/                    # Swift native wrapper (PhoneBuddy.swift) & SwiftUI demo
│   ├── android/                # Kotlin native wrapper (NativeAgent.kt) & JNI bridge (phonebuddy_jni.c) & Compose demo
│   └── c_demo/                 # C API usage example (main.c)
└── scripts/                    # Cross-compilation and packaging scripts
    ├── build-ios-sdk.sh        # iOS static library (aarch64 / simulator universal) build script
    ├── build-android-sdk.sh    # Android shared library (arm64-v8a / x86_64) build script
    └── package-sdk.sh          # One-shot SDK distribution ZIP packaging script
```

---

## 3. Development & Validation Commands

When developing, refactoring, or fixing bugs in this project, an AI agent must validate with the following commands:

* **Check compilation**:
  ```bash
  cargo check
  ```
* **Run unit and integration tests**:
  ```bash
  cargo test
  ```
* **Local CLI debugging and running**:
  ```bash
  cargo run -p phone-buddy-cli -- --help
  ```
* **iOS cross-compilation build**:
  ```bash
  ./scripts/build-ios-sdk.sh
  ```
* **iOS build & run on simulator/device**:
  ```bash
  ./examples/ios/build-and-install.sh [--simulator | --device]
  ```
* **Android cross-compilation build**:
  ```bash
  ./scripts/build-android-sdk.sh
  ```
* **Android build APK & install onto phone**:
  ```bash
  ./examples/android/build-and-install.sh [--logs]
  ```
* **One-shot SDK distribution packaging**:
  ```bash
  ./scripts/package-sdk.sh [--build]
  ```

---

## 4. Agent Development Rules & Constraints

1. **Stay aligned with the upstream `../grok-build` engine paradigm**:
   * When modifying core Agent mechanisms (such as Task Planner, Tool Call Loop, Doom Detection, Retry logic), maintain design consistency with the upstream `../grok-build` philosophy.
   * **Never** reintroduce the external process dependencies (`std::process::Command` / `tokio::process`) or C dynamic library bindings used by `../grok-build`.
2. **Strictly follow mobile sandbox and cross-compilation constraints**:
   * Any new dependency must support targets such as `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios`, `arm64-v8a` (Android), and `x86_64` (Android).
   * Prefer dependencies implemented in pure Rust.
3. **C FFI interface conventions and memory safety**:
   * The C API exported by `crates/phone-buddy-ffi` must stay standard C-ABI compatible (`#[no_mangle]`, `pub unsafe extern "C"`, `repr(C)`).
   * All Rust panics crossing the FFI boundary must be caught internally with `catch_unwind` to avoid undefined behavior (UB) across the C boundary.
4. **Validation requirements**:
   * After every change to the core Rust library, run at least `cargo check` and `cargo test` to ensure no compilation errors or regressions.
5. **Language & Documentation Conventions**:
   * As an open-source project targeting a global mobile developer community, all generated code, inline comments, docstrings, public APIs (Rust/C/Swift/Kotlin), git commits, and primary documentation (e.g., `README.md`) must default to English.
   * Supplementary localized documentation (e.g., `README_CN.md`) can be provided alongside the primary English docs.
