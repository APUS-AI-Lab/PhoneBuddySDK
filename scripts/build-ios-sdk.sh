#!/usr/bin/env bash
# Build the iOS static libraries and the artifacts needed for Swift integration.
#
# Outputs into dist/ios/:
#   - libphone_buddy_ffi-device.a            (aarch64-apple-ios)
#   - libphone_buddy_ffi-sim.a               (arm64 + x86_64 universal sim)
#   - include/phone_buddy.h                  (C header)
#
# Usage: ./scripts/build-ios-sdk.sh [--profile release|ios-dist]
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

PROFILE="${1:-release}"
case "$PROFILE" in
  --profile) PROFILE="${2:-release}" ;;
  release|ios-dist) ;;
  *) echo "usage: $0 [release|ios-dist]"; exit 2 ;;
esac

OUT="$ROOT/dist/ios"
mkdir -p "$OUT/include"

echo "==> Generating C header"
PB_BUILD_HEADER=1 cargo build -p phone-buddy-ffi --quiet
cp crates/phone-buddy-ffi/include/phone_buddy.h "$OUT/include/"

echo "==> Building device target (aarch64-apple-ios)"
cargo build -p phone-buddy-ffi --target aarch64-apple-ios --profile "$PROFILE"

echo "==> Building simulator targets (aarch64-apple-ios-sim + x86_64-apple-ios)"
cargo build -p phone-buddy-ffi --target aarch64-apple-ios-sim --profile "$PROFILE"
cargo build -p phone-buddy-ffi --target x86_64-apple-ios --profile "$PROFILE"

strip -S "target/aarch64-apple-ios/$PROFILE/libphone_buddy_ffi.a" -o "$OUT/libphone_buddy_ffi-device.a"
lipo -create \
  "target/aarch64-apple-ios-sim/$PROFILE/libphone_buddy_ffi.a" \
  "target/x86_64-apple-ios/$PROFILE/libphone_buddy_ffi.a" \
  -output "$OUT/libphone_buddy_ffi-sim.a"
strip -S "$OUT/libphone_buddy_ffi-sim.a"

echo ""
echo "Done: $OUT"
ls -lh "$OUT"
cat <<'HELP'

Xcode integration steps:
  1. Add libphone_buddy_ffi-device.a / libphone_buddy_ffi-sim.a to the
     project (or wrap them in an .xcframework / switch per architecture).
  2. Add include/phone_buddy.h to the project and import it from Swift via
     a Bridging Header (see examples/ios/PhoneBuddy-Bridging-Header.h).
  3. Follow the wrapper and usage in examples/ios/PhoneBuddy.swift.
HELP
