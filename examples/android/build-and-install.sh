#!/usr/bin/env bash
# ==============================================================================
# PhoneBuddy Android Demo: One-click Build APK & Install onto Phone
#
# 1. Builds PhoneBuddy Rust core library (.so for arm64-v8a / x86_64)
# 2. Compiles Android demo app with Jetpack Compose into APK
# 3. Detects connected phone / emulator and installs the APK via ADB
# 4. Automatically launches the PhoneBuddy Agent demo app
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# Colored output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

REBUILD_RUST=false
FOLLOW_LOGS=false
DEVICE_SERIAL=""
BUILD_TYPE="debug"

show_help() {
  cat <<HELP
PhoneBuddy Android Demo: One-click Build & Install Script

Usage:
  ./examples/android/build-and-install.sh [options]

Options:
  -r, --rebuild-rust      Force re-compile the Rust phone-buddy-ffi native library
  -d, --device <serial>   Specify target ADB device serial number
  -l, --logs              Stream logcat logs after launching the app
  --release               Build Release APK instead of Debug APK
  -h, --help              Show this help message

Examples:
  ./examples/android/build-and-install.sh
  ./examples/android/build-and-install.sh --logs
  ./examples/android/build-and-install.sh --rebuild-rust
HELP
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--rebuild-rust)
      REBUILD_RUST=true
      shift
      ;;
    -d|--device)
      DEVICE_SERIAL="$2"
      shift 2
      ;;
    -l|--logs)
      FOLLOW_LOGS=true
      shift
      ;;
    --release)
      BUILD_TYPE="release"
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}" >&2
      show_help
      ;;
  esac
done

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PhoneBuddy Android Demo - Build APK & Install Script          ${NC}"
echo -e "${BLUE}================================================================${NC}"

# ── 1. Locate Android SDK & ADB ───────────────────────────────────────────────
echo -e "\n${BLUE}==> [1/5] Checking Android SDK and ADB...${NC}"

SDK_DIR="${ANDROID_HOME:-}"
if [[ -z "$SDK_DIR" || ! -d "$SDK_DIR" ]]; then
  for cand in "$HOME/Library/Android/sdk" "$HOME/Android/Sdk" "/usr/local/share/android-sdk"; do
    if [[ -d "$cand" ]]; then
      SDK_DIR="$cand"
      break
    fi
  done
fi

if [[ -z "$SDK_DIR" || ! -d "$SDK_DIR" ]]; then
  echo -e "${RED}Error: Android SDK not found! Please set ANDROID_HOME environment variable.${NC}" >&2
  exit 1
fi
export ANDROID_HOME="$SDK_DIR"
echo -e "  Found Android SDK: ${GREEN}$SDK_DIR${NC}"

# Find ADB
ADB_BIN=""
if command -v adb >/dev/null 2>&1; then
  ADB_BIN="$(command -v adb)"
elif [[ -f "$SDK_DIR/platform-tools/adb" ]]; then
  ADB_BIN="$SDK_DIR/platform-tools/adb"
fi

if [[ -z "$ADB_BIN" || ! -x "$ADB_BIN" ]]; then
  echo -e "${YELLOW}Warning: adb tool not found in PATH or platform-tools. Cannot auto-install APK.${NC}"
fi

# Ensure local.properties exists in examples/android
mkdir -p "$ROOT/examples/android"
echo "sdk.dir=$SDK_DIR" > "$ROOT/examples/android/local.properties"

# ── 2. Build or verify Rust .so native libraries ──────────────────────────────
echo -e "\n${BLUE}==> [2/5] Checking PhoneBuddy native library (.so)...${NC}"

NEED_RUST_BUILD=false
if [[ "$REBUILD_RUST" == true ]]; then
  NEED_RUST_BUILD=true
elif [[ ! -f "$ROOT/dist/android/jniLibs/arm64-v8a/libphone_buddy_ffi.so" && ! -f "$ROOT/dist/android/jniLibs/x86_64/libphone_buddy_ffi.so" ]]; then
  NEED_RUST_BUILD=true
fi

if [[ "$NEED_RUST_BUILD" == true ]]; then
  echo "  Compiling Rust native libraries for Android (arm64-v8a & x86_64)..."
  ./scripts/build-android-sdk.sh arm64-v8a x86_64
else
  echo -e "  Using existing prebuilt native libraries in ${GREEN}dist/android/jniLibs/${NC}"
fi

# Copy into app jniLibs
mkdir -p "$ROOT/examples/android/app/src/main/jniLibs"
cp -r "$ROOT/dist/android/jniLibs"/* "$ROOT/examples/android/app/src/main/jniLibs/"

# ── 3. Compile Android Application into APK ───────────────────────────────────
echo -e "\n${BLUE}==> [3/5] Building Android APK (${BUILD_TYPE})...${NC}"
cd "$ROOT/examples/android"

GRADLE_CMD="./gradlew"
if [[ ! -x "$GRADLE_CMD" ]]; then
  chmod +x "$GRADLE_CMD" || true
fi

if [[ "$BUILD_TYPE" == "release" ]]; then
  $GRADLE_CMD assembleRelease
  APK_PATH="$ROOT/examples/android/app/build/outputs/apk/release/app-release.apk"
  if [[ ! -f "$APK_PATH" ]]; then
    APK_PATH="$ROOT/examples/android/app/build/outputs/apk/release/app-release-unsigned.apk"
  fi
else
  $GRADLE_CMD assembleDebug
  APK_PATH="$ROOT/examples/android/app/build/outputs/apk/debug/app-debug.apk"
fi

cd "$ROOT"

if [[ ! -f "$APK_PATH" ]]; then
  echo -e "${RED}Error: APK build failed or output file not found!${NC}" >&2
  exit 1
fi

APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
echo -e "  ${GREEN}✓ APK built successfully!${NC}"
echo -e "  Location: ${YELLOW}$APK_PATH${NC} (${APK_SIZE})"

# ── 4. Detect Connected Android Device ────────────────────────────────────────
echo -e "\n${BLUE}==> [4/5] Checking connected Android devices via ADB...${NC}"

if [[ -z "$ADB_BIN" ]]; then
  echo -e "${YELLOW}ADB not found. Please manually install the APK:${NC}"
  echo -e "  adb install -r $APK_PATH"
  exit 0
fi

DEVICES_OUTPUT=$("$ADB_BIN" devices | grep -v "List of devices attached" | grep -v "^$" || true)
CONNECTED_COUNT=$(echo "$DEVICES_OUTPUT" | grep -c "device$" || true)

if [[ "$CONNECTED_COUNT" -eq 0 ]]; then
  echo -e "${YELLOW}No Android phone or emulator found connected with USB debugging enabled.${NC}"
  echo -e "\nTo install onto your phone:"
  echo -e "  1. Connect your Android phone with a USB cable"
  echo -e "  2. Enable 'Developer options' -> 'USB debugging' on the phone"
  echo -e "  3. Re-run this script: ${GREEN}./examples/android/build-and-install.sh${NC}"
  echo -e "  Or manually run: ${GREEN}$ADB_BIN install -r $APK_PATH${NC}"
  exit 0
fi

DEVICE_INFO=$(echo "$DEVICES_OUTPUT" | grep "device$" | head -n 1 | awk '{print $1}')
echo -e "  Found active Android device: ${GREEN}$DEVICE_INFO${NC}"

# ── 5. Install APK & Launch Demo App ───────────────────────────────────────────
echo -e "\n${BLUE}==> [5/5] Installing APK and launching PhoneBuddy Demo...${NC}"

if [[ -n "$DEVICE_SERIAL" ]]; then
  "$ADB_BIN" -s "$DEVICE_SERIAL" install -r "$APK_PATH"
  echo -e "  ${GREEN}✓ APK installed successfully!${NC}"
  echo "  Launching org.phonebuddy.demo/.MainActivity on device..."
  "$ADB_BIN" -s "$DEVICE_SERIAL" shell am start -n "org.phonebuddy.demo/.MainActivity"
else
  "$ADB_BIN" install -r "$APK_PATH"
  echo -e "  ${GREEN}✓ APK installed successfully!${NC}"
  echo "  Launching org.phonebuddy.demo/.MainActivity on device..."
  "$ADB_BIN" shell am start -n "org.phonebuddy.demo/.MainActivity"
fi

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  PhoneBuddy Agent is now running on your phone! 🎉             ${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "  - Session persistence: Active"
echo -e "  - Headless WebView: Active (web_search & web_fetch)"
echo -e "  - Interactive clarifications (ask_user_question): Active"

if [[ "$FOLLOW_LOGS" == true ]]; then
  echo -e "\nStreaming logcat (Ctrl+C to stop)..."
  if [[ -n "$DEVICE_SERIAL" ]]; then
    "$ADB_BIN" -s "$DEVICE_SERIAL" logcat -s "PhoneBuddy:*" "NativeAgent:*" "AndroidRuntime:E"
  else
    "$ADB_BIN" logcat -s "PhoneBuddy:*" "NativeAgent:*" "AndroidRuntime:E"
  fi
fi
