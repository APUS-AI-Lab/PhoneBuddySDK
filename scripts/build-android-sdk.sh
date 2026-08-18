#!/usr/bin/env bash
# Build the Android shared libraries (JNI-loadable .so), output per ABI.
#
# Outputs to dist/android/jniLibs/<abi>/libphone_buddy_ffi.so
#
# Requires: Android NDK (default lookup in $ANDROID_NDK_HOME,
# $ANDROID_HOME/ndk/*, ~/Library/Android/sdk/ndk/*).
#
# Usage: ./scripts/build-android-sdk.sh [abi...]   # default: all ABIs
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Locate cargo/rustup ─────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v cargo >/dev/null 2>&1 || { echo "cargo not found; please install Rust first" >&2; exit 1; }

# ── Locate NDK ──────────────────────────────────────────────────────────
NDK="${ANDROID_NDK_HOME:-}"
if [[ -z "$NDK" ]]; then
  for cand in \
    "${ANDROID_HOME:-$HOME/Library/Android/sdk}/ndk"/* \
    "$HOME/Library/Android/sdk/ndk"/*; do
    if [[ -d "$cand" ]]; then NDK="$cand"; break; fi
  done
fi
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
  echo "Android NDK not found; please set ANDROID_NDK_HOME" >&2
  exit 1
fi
HOST_TAG="darwin-x86_64"
[[ "$(uname -s)" == "Linux" ]] && HOST_TAG="linux-x86_64"
NDK_BIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
if [[ ! -d "$NDK_BIN" ]]; then
  echo "NDK toolchain directory does not exist: $NDK_BIN" >&2
  exit 1
fi
echo "==> Using NDK: $NDK"

declare -a ABIS
if [[ $# -gt 0 ]]; then
  ABIS=("$@")
else
  ABIS=(arm64-v8a x86_64 armeabi-v7a x86)
fi

OUT="$ROOT/dist/android/jniLibs"
mkdir -p "$OUT"

echo "==> Generating C header"
PB_BUILD_HEADER=1 cargo build -p phone-buddy-ffi --quiet
mkdir -p "$ROOT/dist/android/include"
cp crates/phone-buddy-ffi/include/phone_buddy.h "$ROOT/dist/android/include/"

triple_for() {
  case "$1" in
    arm64-v8a) echo "aarch64-linux-android" ;;
    x86_64) echo "x86_64-linux-android" ;;
    armeabi-v7a) echo "armv7-linux-androideabi" ;;
    x86) echo "i686-linux-android" ;;
    *) echo "" ;;
  esac
}
cc_for() {
  case "$1" in
    aarch64-linux-android) echo "aarch64-linux-android24-clang" ;;
    x86_64-linux-android) echo "x86_64-linux-android24-clang" ;;
    armv7-linux-androideabi) echo "armv7a-linux-androideabi24-clang" ;;
    i686-linux-android) echo "i686-linux-android24-clang" ;;
    *) echo "" ;;
  esac
}

for abi in "${ABIS[@]}"; do
  triple="$(triple_for "$abi")"
  if [[ -z "$triple" ]]; then echo "Unknown ABI: $abi" >&2; exit 2; fi
  cc="$NDK_BIN/$(cc_for "$triple")"
  envname="$(echo "$triple" | tr '-' '_')"
  envupper="$(echo "$envname" | tr '[:lower:]' '[:upper:]')"

  echo "==> Building $abi ($triple)"
  env "CC_${envname}=$cc" \
      "AR_${envname}=$NDK_BIN/llvm-ar" \
      "CARGO_TARGET_${envupper}_LINKER=$cc" \
      "PATH=$NDK_BIN:$PATH" \
    cargo build -p phone-buddy-ffi --target "$triple" --release

  mkdir -p "$OUT/$abi"
  cp "target/$triple/release/libphone_buddy_ffi.so" "$OUT/$abi/"
  "$NDK_BIN/llvm-strip" "$OUT/$abi/libphone_buddy_ffi.so" 2>/dev/null || true
done

echo ""
echo "Done: $OUT"
find "$OUT" -name "*.so" -exec ls -lh {} \;
cat <<'HELP'

Android integration steps:
  1. Copy dist/android/jniLibs into app/src/main/jniLibs (or configure
     abiFilters).
  2. Use the JNI bridge in examples/android/phonebuddy_jni.c (update the
     package name, compile it into your CMake/ndk-build), or declare your
     own native methods following the System.loadLibrary("phone_buddy_ffi")
     pattern in examples/android/NativeAgent.kt.
HELP
