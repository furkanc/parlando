#!/bin/sh
# Build the Mach-O stub used as Parlando.app's main executable.
# Output: src/parlando/assets/parlando-launcher (arm64, ad-hoc signed by the
# linker). Requires the Xcode Command Line Tools. Run after editing
# scripts/launcher.c; the result is committed so users need no compiler.
set -eu
cd "$(dirname "$0")/.."
out=src/parlando/assets/parlando-launcher
cc -Os -Wall -Wextra -arch arm64 -mmacosx-version-min=13.0 -o "$out" scripts/launcher.c
strip "$out"
codesign --force --sign - --identifier com.parlando.launcher "$out"
ls -l "$out"
file "$out"
