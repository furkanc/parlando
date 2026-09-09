#!/bin/sh
# parlando installer — macOS (Apple Silicon)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/furkanc/parlando/main/install.sh | sh
#   or from a local checkout:
#   ./install.sh
#
# Installs the parlando package as a uv tool, which puts the `parlando` and
# `parlando-menubar` commands on your PATH. Uninstall:
#   uv tool uninstall parlando

set -eu

REPO_GIT="git+https://github.com/furkanc/parlando"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
err() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || err "parlando is macOS-only."
[ "$(uname -m)" = "arm64" ] || err "parlando requires Apple Silicon (arm64)."

# 1) uv (Python runtime + dependency manager)
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv (Python runner)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || err "uv installation failed; see https://docs.astral.sh/uv/"

# 2) install the package as a uv tool (creates the commands automatically).
#    When run from a checkout, install from it; otherwise from the repo.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
if [ -n "${SCRIPT_DIR:-}" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    say "Installing parlando from local checkout: $SCRIPT_DIR"
    uv tool install --force --reinstall "$SCRIPT_DIR"
else
    say "Installing parlando from the repository..."
    uv tool install --force "$REPO_GIT"
fi

# 3) PATH hint
if ! command -v parlando >/dev/null 2>&1; then
    say "NOTE: the uv tool bin directory is not on your PATH. Run:"
    printf '    uv tool update-shell\n'
    say "then open a new terminal."
fi

say "Installed. Run:  parlando        (terminal)"
say "           or:  parlando-menubar (menu bar app)"
say "First run downloads the speech model (~1-2 GB, one-time)."
say "macOS will ask for Microphone and Accessibility permissions."
