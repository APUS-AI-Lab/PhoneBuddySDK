#!/usr/bin/env bash
# ==============================================================================
# PhoneBuddy SDK one-shot packaging script
# Bundles the iOS / Android / C artifacts, wrapper code, headers and docs into
# a standard SDK ZIP package.
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Colored output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

VERSION=""
AUTO_BUILD=false
CUSTOM_OUTPUT=""

show_help() {
  cat <<HELP
PhoneBuddy SDK one-shot packaging script

Usage:
  ./scripts/package-sdk.sh [options]

Options:
  -b, --build           Run the iOS and Android build scripts before packaging
  -v, --version <ver>   SDK version to embed (default: version from Cargo.toml)
  -o, --output <path>   Output zip file path (default: dist/phone-buddy-sdk-<version>.zip)
  -h, --help            Show this help message

Examples:
  ./scripts/package-sdk.sh
  ./scripts/package-sdk.sh --build
  ./scripts/package-sdk.sh --version 0.1.0 -o ./phone-buddy-sdk.zip
HELP
  exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build)
      AUTO_BUILD=true
      shift
      ;;
    -v|--version)
      VERSION="$2"
      shift 2
      ;;
    -o|--output)
      CUSTOM_OUTPUT="$2"
      shift 2
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

# Read the version from Cargo.toml unless one was given
if [[ -z "$VERSION" ]]; then
  if [[ -f "Cargo.toml" ]]; then
    VERSION=$(grep -m1 '^version =' Cargo.toml | awk -F '"' '{print $2}')
  fi
  if [[ -z "$VERSION" ]]; then
    VERSION="v0.1.0"
  else
    VERSION="v${VERSION}"
  fi
fi

# Make sure the zip command is available
if ! command -v zip >/dev/null 2>&1; then
  echo -e "${RED}Error: zip is not installed; please install it (e.g. brew install zip or apt install zip)${NC}" >&2
  exit 1
fi

# If --build was given, trigger the iOS and Android builds first
if [[ "$AUTO_BUILD" == true ]]; then
  echo -e "${BLUE}==> [1/4] Running platform builds...${NC}"
  echo "--> Building iOS libraries..."
  ./scripts/build-ios-sdk.sh
  echo "--> Building Android libraries..."
  ./scripts/build-android-sdk.sh
else
  echo -e "${BLUE}==> [1/4] Checking prebuilt artifacts...${NC}"
fi

# Check that the required artifacts exist
MISSING_FILES=()

if [[ ! -f "dist/ios/libphone_buddy_ffi-device.a" || ! -f "dist/ios/libphone_buddy_ffi-sim.a" ]]; then
  MISSING_FILES+=("dist/ios (iOS static libraries not found; run ./scripts/build-ios-sdk.sh first)")
fi

if [[ ! -d "dist/android/jniLibs" ]] || [[ -z "$(find dist/android/jniLibs -name "libphone_buddy_ffi.so" 2>/dev/null)" ]]; then
  MISSING_FILES+=("dist/android/jniLibs (Android .so libraries not found; run ./scripts/build-android-sdk.sh first)")
fi

if [[ ! -f "crates/phone-buddy-ffi/include/phone_buddy.h" && ! -f "dist/ios/include/phone_buddy.h" && ! -f "dist/android/include/phone_buddy.h" ]]; then
  MISSING_FILES+=("phone_buddy.h (C header not found)")
fi

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
  echo -e "${RED}Error: required build artifacts are missing!${NC}"
  for item in "${MISSING_FILES[@]}"; do
    echo -e "  - ${YELLOW}$item${NC}"
  done
  echo -e "\nTip: add the ${GREEN}--build${NC} flag to build and package automatically, e.g. ${GREEN}./scripts/package-sdk.sh --build${NC}"
  exit 1
fi

# Set up the staging directory for packaging
STAGE_DIR="$ROOT/target/sdk_package"
SDK_DIR_NAME="phone-buddy-sdk-${VERSION}"
SDK_DIR="$STAGE_DIR/$SDK_DIR_NAME"

rm -rf "$STAGE_DIR"
mkdir -p "$SDK_DIR"

echo -e "${BLUE}==> [2/4] Assembling SDK file layout...${NC}"

# 1. Shared docs and metadata
mkdir -p "$SDK_DIR/docs"
[ -f README.md ] && cp README.md "$SDK_DIR/docs/README.md"
[ -f README_CN.md ] && cp README_CN.md "$SDK_DIR/docs/README_CN.md"
[ -f QUICKSTART.md ] && cp QUICKSTART.md "$SDK_DIR/docs/QUICKSTART.md"
[ -f LICENSE ] && cp LICENSE "$SDK_DIR/LICENSE"
[ -f NOTICE ] && cp NOTICE "$SDK_DIR/NOTICE"

cat <<EOF > "$SDK_DIR/VERSION"
PhoneBuddy SDK
Version: ${VERSION}
Build Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

cat <<EOF > "$SDK_DIR/README.md"
# PhoneBuddy SDK Integration Guide (${VERSION})

Welcome! This SDK package ships prebuilt multi-architecture binaries plus the Swift / Kotlin wrappers, so you can **drop it straight into your app project and use it out of the box, without installing Rust locally or running any build scripts**.

## 📦 SDK Package Layout

- **\`ios/\`**: iOS static libraries (\`.a\`), C header (\`phone_buddy.h\`), high-level Swift wrapper (\`PhoneBuddy.swift\`) and a SwiftUI example
- **\`android/\`**: Android shared libraries (\`jniLibs/arm64-v8a\` & \`x86_64\`), Kotlin wrapper (\`NativeAgent.kt\`) and a Compose example
- **\`c_api/\`**: Generic C-ABI interface and C/C++ integration examples
- **\`docs/\`**: Detailed API manual and quick start guide (\`QUICKSTART.md\`)

## 🚀 Quick Integration (drag and drop)

### 1. iOS integration (see [ios/README.md](ios/README.md))
1. Import the static libraries under \`ios/libs/\` (\`libphone_buddy_ffi-device.a\` / \`libphone_buddy_ffi-sim.a\`) and \`ios/include/phone_buddy.h\` into your Xcode project.
2. Add \`ios/wrapper/PhoneBuddy.swift\` and call the Agent engine through its \`async/await\` API.

### 2. Android integration (see [android/README.md](android/README.md))
1. Copy the \`android/jniLibs/\` folder into your module's \`src/main/jniLibs/\` directory.
2. Copy \`android/wrapper/NativeAgent.kt\` into your project package and call the Agent engine through coroutines.

### 3. C / C++ integration (see [c_api/README.md](c_api/README.md))
* Include the \`phone_buddy.h\` header and link the shared/static library for your platform.
EOF

# 2. Assemble the iOS SDK
echo "  -> Assembling iOS SDK files..."
mkdir -p "$SDK_DIR/ios/libs"
mkdir -p "$SDK_DIR/ios/include"
mkdir -p "$SDK_DIR/ios/wrapper"
mkdir -p "$SDK_DIR/ios/examples"

cp dist/ios/libphone_buddy_ffi-device.a "$SDK_DIR/ios/libs/"
cp dist/ios/libphone_buddy_ffi-sim.a "$SDK_DIR/ios/libs/"
if [ -f "dist/ios/include/phone_buddy.h" ]; then
  cp dist/ios/include/phone_buddy.h "$SDK_DIR/ios/include/"
else
  cp crates/phone-buddy-ffi/include/phone_buddy.h "$SDK_DIR/ios/include/"
fi
cp examples/ios/PhoneBuddy.swift "$SDK_DIR/ios/wrapper/"
cp examples/ios/PhoneBuddy-Bridging-Header.h "$SDK_DIR/ios/wrapper/"
cp examples/ios/README.md "$SDK_DIR/ios/README.md"
cp examples/ios/DemoApp.swift "$SDK_DIR/ios/examples/"

# 3. Assemble the Android SDK
echo "  -> Assembling Android SDK files..."
mkdir -p "$SDK_DIR/android/include"
mkdir -p "$SDK_DIR/android/wrapper"
mkdir -p "$SDK_DIR/android/examples"
mkdir -p "$SDK_DIR/android/jniLibs"

for abi_dir in dist/android/jniLibs/*; do
  if [ -d "$abi_dir" ]; then
    abi="$(basename "$abi_dir")"
    if [ -f "$abi_dir/libphone_buddy_ffi.so" ]; then
      mkdir -p "$SDK_DIR/android/jniLibs/$abi"
      cp "$abi_dir/libphone_buddy_ffi.so" "$SDK_DIR/android/jniLibs/$abi/"
    fi
  fi
done

if [ -f "dist/android/include/phone_buddy.h" ]; then
  cp dist/android/include/phone_buddy.h "$SDK_DIR/android/include/"
else
  cp crates/phone-buddy-ffi/include/phone_buddy.h "$SDK_DIR/android/include/"
fi
cp examples/android/NativeAgent.kt "$SDK_DIR/android/wrapper/"
cp examples/android/phonebuddy_jni.c "$SDK_DIR/android/wrapper/"
cp examples/android/README.md "$SDK_DIR/android/README.md"
cp examples/android/MainActivity.kt "$SDK_DIR/android/examples/"

# 4. Assemble the C API
echo "  -> Assembling C API files..."
mkdir -p "$SDK_DIR/c_api/include"
mkdir -p "$SDK_DIR/c_api/examples"

cp crates/phone-buddy-ffi/include/phone_buddy.h "$SDK_DIR/c_api/include/"
cp examples/c_demo/main.c "$SDK_DIR/c_api/examples/"
cp examples/c_demo/README.md "$SDK_DIR/c_api/README.md"
[ -f examples/c_demo/config.json.example ] && cp examples/c_demo/config.json.example "$SDK_DIR/c_api/examples/"
[ -f examples/c_demo/Makefile ] && cp examples/c_demo/Makefile "$SDK_DIR/c_api/examples/"
[ -f examples/c_demo/build.sh ] && cp examples/c_demo/build.sh "$SDK_DIR/c_api/examples/"

# Determine the output ZIP absolute path
mkdir -p "$ROOT/dist"
if [[ -n "$CUSTOM_OUTPUT" ]]; then
  ZIP_OUTPUT="$CUSTOM_OUTPUT"
else
  ZIP_OUTPUT="$ROOT/dist/${SDK_DIR_NAME}.zip"
fi

# Make sure the output directory exists
mkdir -p "$(dirname "$ZIP_OUTPUT")"

echo -e "${BLUE}==> [3/4] Compressing into ZIP...${NC}"
cd "$STAGE_DIR"
rm -f "$ZIP_OUTPUT"
zip -r -q "$ZIP_OUTPUT" "$SDK_DIR_NAME"

cd "$ROOT"
echo -e "${BLUE}==> [4/4] Cleaning up temp directory...${NC}"
rm -rf "$STAGE_DIR"

FILE_SIZE=$(du -h "$ZIP_OUTPUT" | cut -f1)

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  PhoneBuddy SDK packaged successfully!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "  - Version:    ${YELLOW}${VERSION}${NC}"
echo -e "  - Output path:${YELLOW}${ZIP_OUTPUT}${NC}"
echo -e "  - Size:       ${YELLOW}${FILE_SIZE}${NC}"
echo -e "\nPackage layout overview:"
echo -e "----------------------------------------------------------------"
unzip -l "$ZIP_OUTPUT" | head -n 35
echo -e "... (some files omitted)"
echo -e "----------------------------------------------------------------\n"
