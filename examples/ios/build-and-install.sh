#!/usr/bin/env bash
# ==============================================================================
# PhoneBuddy iOS Demo: One-click Build & Install Script (Simulator / Device)
#
# 1. Builds PhoneBuddy Rust core static library (.a for device & simulator)
# 2. Compiles SwiftUI iOS demo app using xcodebuild
# 3. Automatically installs and launches the app on Simulator or connected iPhone
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

TARGET_MODE="auto"
SPECIFIED_TARGET=""
REBUILD_RUST=false

show_help() {
  cat <<HELP
PhoneBuddy iOS Demo: One-click Build & Install Script

Usage:
  ./examples/ios/build-and-install.sh [options]

Options:
  -s, --simulator [id/name]   Build, install, and run on iOS Simulator (default: auto-select/boot)
  -d, --device [id/name]      Build, install, and run on connected physical iPhone
  -r, --rebuild-rust          Force re-compile the Rust phone-buddy-ffi static libraries
  -h, --help                  Show this help message

Examples:
  ./examples/ios/build-and-install.sh
  ./examples/ios/build-and-install.sh --simulator "iPhone 17"
  ./examples/ios/build-and-install.sh --device
  ./examples/ios/build-and-install.sh --rebuild-rust
HELP
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--simulator)
      TARGET_MODE="simulator"
      if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
        SPECIFIED_TARGET="$2"
        shift 2
      else
        shift
      fi
      ;;
    -d|--device)
      TARGET_MODE="device"
      if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
        SPECIFIED_TARGET="$2"
        shift 2
      else
        shift
      fi
      ;;
    -r|--rebuild-rust)
      REBUILD_RUST=true
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
echo -e "${BLUE}  PhoneBuddy iOS Demo - Build & Install Script                  ${NC}"
echo -e "${BLUE}================================================================${NC}"

# ── 1. Check Xcode Tools ──────────────────────────────────────────────────────
echo -e "\n${BLUE}==> [1/4] Checking Xcode build toolchain...${NC}"

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo -e "${RED}Error: xcodebuild not found. Please install Xcode and Command Line Tools.${NC}" >&2
  exit 1
fi
echo -e "  Found xcodebuild: ${GREEN}$(xcodebuild -version | tr '\n' ' ')${NC}"

# ── 2. Build or verify Rust static libraries ──────────────────────────────────
echo -e "\n${BLUE}==> [2/4] Checking PhoneBuddy native static libraries (.a)...${NC}"

NEED_RUST_BUILD=false
if [[ "$REBUILD_RUST" == true ]]; then
  NEED_RUST_BUILD=true
elif [[ ! -f "$ROOT/dist/ios/libphone_buddy_ffi-device.a" || ! -f "$ROOT/dist/ios/libphone_buddy_ffi-sim.a" ]]; then
  NEED_RUST_BUILD=true
fi

if [[ "$NEED_RUST_BUILD" == true ]]; then
  echo "  Compiling Rust static libraries for iOS (device & simulator)..."
  ./scripts/build-ios-sdk.sh
else
  echo -e "  Using existing prebuilt static libraries in ${GREEN}dist/ios/${NC}"
fi

# ── 3. Determine Target (Simulator vs Physical Device) ─────────────────────────
echo -e "\n${BLUE}==> [3/4] Determining target platform...${NC}"

DERIVED_DATA="$ROOT/target/DerivedData/iOS"
mkdir -p "$DERIVED_DATA"

if [[ "$TARGET_MODE" == "device" ]]; then
  # Physical device build
  echo -e "  Target: ${GREEN}Physical iOS Device${NC}"

  DEVICE_ID="$SPECIFIED_TARGET"
  if [[ -z "$DEVICE_ID" ]]; then
    # Auto-detect available physical device via devicectl
    DEVICE_LIST_RAW=$(xcrun devicectl list devices 2>&1 || true)
    DEVICE_ID=$(echo "$DEVICE_LIST_RAW" | grep "available" | head -n 1 | awk '{print $3}' || true)
    if [[ -z "$DEVICE_ID" ]]; then
      DEVICE_ID=$(echo "$DEVICE_LIST_RAW" | grep -v "Identifier" | grep -v "\-\-\-" | grep -v "^$" | head -n 1 | awk '{print $3}' || true)
    fi
  fi

  echo "  Building PhoneBuddyDemo.app for iOS Device..."
  xcodebuild -project "$ROOT/examples/ios/PhoneBuddyDemo.xcodeproj" \
    -scheme PhoneBuddyDemo \
    -destination "generic/platform=iOS" \
    -derivedDataPath "$DERIVED_DATA" \
    CODE_SIGNING_ALLOWED=YES \
    CODE_SIGNING_REQUIRED=NO \
    build

  APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphoneos/PhoneBuddyDemo.app"
  echo -e "  ${GREEN}✓ App built at: $APP_PATH${NC}"

  echo -e "\n${BLUE}==> [4/4] Installing and launching on iPhone...${NC}"
  if [[ -n "$DEVICE_ID" ]]; then
    echo "  Installing onto device ($DEVICE_ID)..."
    if xcrun devicectl device install app --device "$DEVICE_ID" "$APP_PATH" 2>/dev/null; then
      echo "  Launching app..."
      xcrun devicectl device process launch --device "$DEVICE_ID" org.phonebuddy.demo || true
      echo -e "\n${GREEN}✓ PhoneBuddy Demo launched on your iPhone! 🎉${NC}"
    else
      echo -e "${YELLOW}Could not auto-install via devicectl. Open Xcode to install or run:${NC}"
      echo "  open $ROOT/examples/ios/PhoneBuddyDemo.xcodeproj"
    fi
  else
    echo -e "${YELLOW}No connected iPhone found via devicectl.${NC}"
    echo -e "  You can open the project in Xcode and hit Run: ${GREEN}open $ROOT/examples/ios/PhoneBuddyDemo.xcodeproj${NC}"
  fi

else
  # Simulator build (default)
  echo -e "  Target: ${GREEN}iOS Simulator${NC}"

  # Find a suitable simulator
  SIM_ID="$SPECIFIED_TARGET"
  if [[ -z "$SIM_ID" ]]; then
    # Check if a simulator is already booted
    BOOTED_SIM=$(xcrun simctl list devices | grep "(Booted)" | head -n 1 || true)
    if [[ -n "$BOOTED_SIM" ]]; then
      SIM_ID=$(echo "$BOOTED_SIM" | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}')
      SIM_NAME=$(echo "$BOOTED_SIM" | sed -E 's/^[[:space:]]+//' | cut -d'(' -f1)
      echo -e "  Using currently booted simulator: ${GREEN}$SIM_NAME ($SIM_ID)${NC}"
    else
      # Pick an available iPhone simulator
      AVAIL_SIM=$(xcrun simctl list devices available | grep "iPhone" | head -n 1 || true)
      if [[ -n "$AVAIL_SIM" ]]; then
        SIM_ID=$(echo "$AVAIL_SIM" | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}')
        SIM_NAME=$(echo "$AVAIL_SIM" | sed -E 's/^[[:space:]]+//' | cut -d'(' -f1)
        echo -e "  Booting simulator: ${GREEN}$SIM_NAME ($SIM_ID)${NC}"
        xcrun simctl boot "$SIM_ID" || true
      else
        echo -e "${RED}Error: No available iOS simulator found!${NC}" >&2
        exit 1
      fi
    fi
  fi

  echo "  Building PhoneBuddyDemo.app for iOS Simulator..."
  xcodebuild -project "$ROOT/examples/ios/PhoneBuddyDemo.xcodeproj" \
    -scheme PhoneBuddyDemo \
    -destination "platform=iOS Simulator,id=$SIM_ID" \
    -derivedDataPath "$DERIVED_DATA" \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGNING_REQUIRED=NO \
    build

  APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphonesimulator/PhoneBuddyDemo.app"
  echo -e "  ${GREEN}✓ App built at: $APP_PATH${NC}"

  echo -e "\n${BLUE}==> [4/4] Installing and launching on iOS Simulator...${NC}"
  xcrun simctl boot "$SIM_ID" 2>/dev/null || true
  xcrun simctl install "$SIM_ID" "$APP_PATH"

  # Copy config.json to App Documents container if available
  CONTAINER_PATH=$(xcrun simctl get_app_container "$SIM_ID" org.phonebuddy.demo data 2>/dev/null || true)
  if [[ -n "$CONTAINER_PATH" && -d "$CONTAINER_PATH" ]]; then
    mkdir -p "$CONTAINER_PATH/Documents" "$CONTAINER_PATH/Documents/PhoneBuddy"
    if [[ -f "$ROOT/examples/ios/config.json" ]]; then
      cp -f "$ROOT/examples/ios/config.json" "$CONTAINER_PATH/Documents/config.json"
      cp -f "$ROOT/examples/ios/config.json" "$CONTAINER_PATH/Documents/PhoneBuddy/config.json"
      echo -e "  ${GREEN}✓ Synced config.json to Simulator App Documents!${NC}"
    elif [[ -f "$HOME/Downloads/config.json" ]]; then
      cp -f "$HOME/Downloads/config.json" "$CONTAINER_PATH/Documents/config.json"
      cp -f "$HOME/Downloads/config.json" "$CONTAINER_PATH/Documents/PhoneBuddy/config.json"
      echo -e "  ${GREEN}✓ Synced config.json to Simulator App Documents!${NC}"
    fi
  fi

  xcrun simctl launch "$SIM_ID" org.phonebuddy.demo
  open -a Simulator >/dev/null 2>&1 &
  disown 2>/dev/null || true

  echo -e "\n${GREEN}================================================================${NC}"
  echo -e "${GREEN}  PhoneBuddy Agent is now running in iOS Simulator! 🎉          ${NC}"
  echo -e "${GREEN}================================================================${NC}"
  echo -e "  - Session persistence: Active"
  echo -e "  - Headless WKWebView: Active (web_search & web_fetch)"
  echo -e "  - Interactive clarifications (ask_user_question): Active"
fi
