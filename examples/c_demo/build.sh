#!/usr/bin/env bash
# ==============================================================================
# Build script for PhoneBuddy SDK C Agent CLI Demo (examples/c_demo/main.c)
#
# Supports macOS (Darwin) and Linux. Automatically compiles the underlying
# phone-buddy-ffi Rust static library and generates the C header if needed.
#
# Usage:
#   ./build.sh [options] [-- run_args...]
#
# Options:
#   -r, --release       Build in release mode (default, optimized)
#   -d, --debug         Build in debug mode (faster compile, debug symbols)
#       --clean         Clean build artifacts
#   -e, --run           Run the demo immediately after a successful build
#   -o, --output <path> Specify output binary path (default: examples/c_demo/demo)
#   -h, --help          Show this help message
# ==============================================================================
set -euo pipefail

# Colored output helpers
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROFILE="release"
CARGO_FLAG="--release"
RUN_AFTER_BUILD=false
CLEAN_ONLY=false
OUTPUT_BIN="$SCRIPT_DIR/demo"
RUN_ARGS=()

show_help() {
  cat <<HELP
PhoneBuddy SDK - C Demo Build Script

Usage:
  $0 [options] [-- <args_to_demo>]

Options:
  -r, --release       Build with release profile (default, optimized)
  -d, --debug         Build with debug profile (faster compilation, debug symbols)
  -e, --run           Execute the demo binary after compilation
  -o, --output <path> Custom output binary path (default: $OUTPUT_BIN)
      --clean         Remove build artifacts and exit
  -h, --help          Display this help message

Examples:
  $0                  # Compile release binary
  $0 --debug          # Compile debug binary
  $0 --run            # Compile and run immediately
  $0 --run -- config.json  # Compile and run with specific config
HELP
  exit 0
}

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--release)
      PROFILE="release"
      CARGO_FLAG="--release"
      shift
      ;;
    -d|--debug)
      PROFILE="debug"
      CARGO_FLAG=""
      shift
      ;;
    -e|--run)
      RUN_AFTER_BUILD=true
      shift
      ;;
    --clean)
      CLEAN_ONLY=true
      shift
      ;;
    -o|--output)
      OUTPUT_BIN="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      ;;
    --)
      shift
      RUN_ARGS=("$@")
      break
      ;;
    *)
      if [[ "$RUN_AFTER_BUILD" == true ]]; then
        RUN_ARGS+=("$1")
        shift
      else
        echo -e "${RED}Error: Unknown option '$1'${NC}" >&2
        echo "Run '$0 --help' for usage." >&2
        exit 1
      fi
      ;;
  esac
done

# Handle --clean
if [[ "$CLEAN_ONLY" == true ]]; then
  echo -e "${YELLOW}==> Cleaning build artifacts...${NC}"
  rm -f "$SCRIPT_DIR/demo" "$SCRIPT_DIR/demo_dylib"
  echo -e "${GREEN}Clean completed.${NC}"
  exit 0
fi

# Locate cargo / rust toolchain
if ! command -v cargo >/dev/null 2>&1; then
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo -e "${RED}Error: cargo not found. Please install Rust (https://rustup.rs).${NC}" >&2
  exit 1
fi

# Locate C compiler (CC env var -> clang -> gcc -> cc)
CC="${CC:-}"
if [[ -z "$CC" ]]; then
  if command -v clang >/dev/null 2>&1; then
    CC="clang"
  elif command -v gcc >/dev/null 2>&1; then
    CC="gcc"
  elif command -v cc >/dev/null 2>&1; then
    CC="cc"
  else
    echo -e "${RED}Error: No C compiler found (clang/gcc/cc). Please install one.${NC}" >&2
    exit 1
  fi
fi

HEADER_FILE="$ROOT/crates/phone-buddy-ffi/include/phone_buddy.h"
STATIC_LIB="$ROOT/target/$PROFILE/libphone_buddy_ffi.a"
SRC_FILE="$SCRIPT_DIR/main.c"
INCLUDE_DIR="$ROOT/crates/phone-buddy-ffi/include"

# 1. Build Rust crate (and generate C header if not present)
echo -e "${BLUE}==> [1/2] Building phone-buddy-ffi ($PROFILE)...${NC}"
if [[ ! -f "$HEADER_FILE" ]]; then
  echo -e "    Generating C header (phone_buddy.h)..."
  (cd "$ROOT" && PB_BUILD_HEADER=1 cargo build -p phone-buddy-ffi ${CARGO_FLAG})
else
  (cd "$ROOT" && cargo build -p phone-buddy-ffi ${CARGO_FLAG})
fi

if [[ ! -f "$STATIC_LIB" ]]; then
  echo -e "${RED}Error: Static library was not created: $STATIC_LIB${NC}" >&2
  exit 1
fi

# 2. Compile C Demo
echo -e "${BLUE}==> [2/2] Compiling C demo executable ($CC)...${NC}"

UNAME_S="$(uname -s)"
CFLAGS=("-Wall" "-Wextra" "-I$INCLUDE_DIR")
if [[ "$PROFILE" == "release" ]]; then
  CFLAGS+=("-O2")
else
  CFLAGS+=("-g" "-O0")
fi

LDFLAGS=("-lpthread" "-ldl" "-lm")
if [[ "$UNAME_S" == "Darwin" ]]; then
  LDFLAGS+=("-framework" "Security" "-framework" "CoreFoundation")
fi

mkdir -p "$(dirname "$OUTPUT_BIN")"

"$CC" "${CFLAGS[@]}" \
  -o "$OUTPUT_BIN" \
  "$SRC_FILE" \
  "$STATIC_LIB" \
  "${LDFLAGS[@]}"

echo -e "${GREEN}✓ Successfully built:${NC} $OUTPUT_BIN"
echo -e "  Binary size: $(ls -lh "$OUTPUT_BIN" | awk '{print $5}')"

# 3. Optional Run
if [[ "$RUN_AFTER_BUILD" == true ]]; then
  echo ""
  echo -e "${BLUE}==> Launching demo...${NC}"
  echo "------------------------------------------------------------"
  cd "$SCRIPT_DIR"
  if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
    exec "$OUTPUT_BIN" "${RUN_ARGS[@]}"
  else
    exec "$OUTPUT_BIN"
  fi
fi
