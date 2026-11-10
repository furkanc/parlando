#!/bin/sh
# Build a self-contained, distributable Parlando.app (+ .dmg).
#
# The bundle embeds a relocatable CPython (python-build-standalone, the
# same builds uv uses) with parlando and all dependencies installed into
# it, under Contents/Resources/runtime. The main executable is the Mach-O
# stub from scripts/launcher.c; launch.sh starts the embedded interpreter.
# The ASR model is NOT embedded: it downloads on first run (~2 GB).
#
# Output: build/Parlando.app and build/Parlando-<version>.dmg
# Requires: uv, macOS with codesign/hdiutil (stock).
#
# Signing: CODESIGN_IDENTITY (e.g. "Developer ID Application: ..." for real
# distribution; notarization is a separate step: xcrun notarytool). Without
# it, the "Parlando Signing" self-signed identity is used if it exists
# (scripts/make_signing_cert.sh creates it; it keeps permissions stable
# across rebuilds), otherwise the build is ad-hoc signed.
set -eu
cd "$(dirname "$0")/.."
ROOT=$PWD
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/parlando/__init__.py)
PYVER=${PYTHON_VERSION:-3.12}
SIGNING_CERT_NAME=${SIGNING_CERT_NAME:-Parlando Signing}
BUILD=$ROOT/build
APP=$BUILD/Parlando.app
RUNTIME=$APP/Contents/Resources/runtime

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }

rm -rf "$BUILD/pyinstall" "$APP" "$BUILD/dmg"
mkdir -p "$BUILD"

# Resolve the signing identity.
if [ -n "${CODESIGN_IDENTITY:-}" ]; then
    IDENTITY=$CODESIGN_IDENTITY
elif security find-identity -v -p codesigning | grep -q "\"$SIGNING_CERT_NAME\""; then
    IDENTITY=$SIGNING_CERT_NAME
else
    say "No signing identity: ad-hoc signature (run scripts/make_signing_cert.sh for a stable one)"
    IDENTITY=-
fi

say "Installing relocatable CPython $PYVER"
UV_PYTHON_INSTALL_DIR="$BUILD/pyinstall" uv python install "$PYVER" --no-bin >/dev/null 2>&1 || \
UV_PYTHON_INSTALL_DIR="$BUILD/pyinstall" uv python install "$PYVER"
# uv also writes an unversioned alias directory next to the real one.
PYHOME=$(ls -d "$BUILD"/pyinstall/cpython-"$PYVER".[0-9]*-macos-aarch64-none | head -1)
PY=$PYHOME/bin/python3
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

say "Installing parlando $VERSION and dependencies into it"
# --target: install straight into the interpreter's site-packages (no venv).
uv pip install --python "$PY" --target "$SP" --compile-bytecode --quiet "$ROOT"

say "Trimming the runtime"
LIB=$PYHOME/lib/python$PYVER
rm -rf "$LIB/test" "$LIB/idlelib" "$LIB/tkinter" "$LIB/turtledemo" "$LIB/ensurepip" \
       "$LIB/config-$PYVER-darwin" "$PYHOME/include" "$PYHOME/share" \
       "$PYHOME"/lib/libtcl* "$PYHOME"/lib/libtk* "$PYHOME"/lib/tcl* "$PYHOME"/lib/tk* \
       "$PYHOME"/lib/itcl* "$PYHOME"/lib/Tix* "$PYHOME"/lib/thread* \
       "$SP/PyObjCTest" "$SP/pip" "$SP"/pip-*.dist-info
find "$SP" -type d \( -name tests -o -name test -o -name testing \) -prune -exec rm -rf {} +
find "$PYHOME" -type f \( -name '*.pyi' -o -name '*.h' -o -name '*.a' \) -delete

say "Assembling the bundle"
uv run --quiet python - "$APP" "$VERSION" <<'PY'
import sys
from pathlib import Path
from parlando.menubar import build_app_bundle
build_app_bundle(Path(sys.argv[1]), python="runtime/bin/python3", sign=False, version=sys.argv[2])
PY
mv "$PYHOME" "$RUNTIME"
rm -rf "$BUILD/pyinstall"

say "Signing ($IDENTITY)"
# Every nested Mach-O first (extensions, dylibs, the interpreter), then the bundle.
# (`file` prints one line per architecture for fat binaries; normalize.)
find "$RUNTIME" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0 \
  | xargs -0 file | grep 'Mach-O' | sed -E 's/ \(for architecture [^)]*\)//; s/:.*//' | sort -u \
  | while IFS= read -r f; do codesign --force --sign "$IDENTITY" "$f" 2>/dev/null; done
codesign --force --sign "$IDENTITY" --identifier com.parlando.menubar --options runtime \
  --entitlements /dev/stdin "$APP" <<'PLIST' 2>/dev/null || codesign --force --sign "$IDENTITY" --identifier com.parlando.menubar "$APP"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
  <key>com.apple.security.device.audio-input</key><true/>
</dict></plist>
PLIST
codesign --verify --deep --strict "$APP"
codesign -dr - "$APP" 2>&1 | grep designated

say "Creating the DMG"
mkdir -p "$BUILD/dmg"
cp -R "$APP" "$BUILD/dmg/"
ln -s /Applications "$BUILD/dmg/Applications"
DMG=$BUILD/Parlando-$VERSION.dmg
rm -f "$DMG"
hdiutil create -quiet -volname "Parlando" -srcfolder "$BUILD/dmg" -ov -format UDZO "$DMG"
rm -rf "$BUILD/dmg"

say "Done"
du -sh "$APP" "$DMG"
