<p align="center">
  <img src="https://raw.githubusercontent.com/furkanc/parlando/main/assets/banner.png" alt="parlando — local voice dictation for macOS" width="720">
</p>

**Local voice dictation for macOS (Apple Silicon).** Tap a key, speak, tap
again — your words are typed into whatever window has focus. 100% on-device:
audio never leaves your machine.

[![PyPI](https://img.shields.io/pypi/v/parlando)](https://pypi.org/project/parlando/)
[![Tests](https://github.com/furkanc/parlando/actions/workflows/tests.yml/badge.svg)](https://github.com/furkanc/parlando/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Features

- **Fully local & offline** — MLX-accelerated Qwen3-ASR on the Apple Neural
  stack; works on a plane, nothing is uploaded, no API keys.
- **Tap-to-dictate** — single tap of right ⌥ Option starts/stops recording
  (the same pattern superwhisper and Wispr Flow use). No always-on mic.
- **Types anywhere** — text is delivered as native keyboard events (Quartz
  CGEvent unicode), so it works in any app, editor, terminal, or chat box.
- **30 languages** — Turkish by default; switch instantly (e.g. `--language
  English`) with no model reload.
- **Voice commands** — "nokta", "virgül", "soru işareti", "yeni satır",
  "gönder" (presses Enter); the English set ("period", "comma", "question
  mark", "new line", "send") is active with `--language English`.
- **Cleanup & polish** — vocalized fillers ("um", "uh", "eee") are removed
  automatically; optional `--polish` runs a small local LLM that drops
  contextual fillers ("you know", "yani"), stutters and false starts, and
  fixes punctuation — guarded so it can only *edit* your words, never
  answer them.
- **Menu bar app** — <img src="https://raw.githubusercontent.com/furkanc/parlando/main/assets/menubar-states.png" alt="menu bar icon states: idle, recording, paused" height="26" align="top"> (idle · recording · paused) with start/stop, language switcher, and start-at-login support.
- **Self-healing** — audio watchdog survives sleep/wake and device changes;
  inference errors never kill the session.
- **Streaming mode (optional)** — words appear as you speak, stabilized with
  LocalAgreement-2 and append-only commits (typed text is never retracted).

## Requirements

- macOS on Apple Silicon
- [`uv`](https://docs.astral.sh/uv/) installed
- ~2 GB disk for the default ASR model (downloaded once from Hugging Face)

## Installation

parlando is on [PyPI](https://pypi.org/project/parlando/). With
[`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install parlando     # puts the parlando command on your PATH
```

or try it without installing anything permanent:

```bash
uvx parlando
```

or with pipx / pip:

```bash
pipx install parlando        # isolated install, commands on PATH
pip install parlando         # into your current Python environment
```

<details>
<summary>Other ways: one-line installer · from source · uninstall</summary>

```bash
# one-liner that also installs uv if you don't have it
curl -fsSL https://raw.githubusercontent.com/furkanc/parlando/main/install.sh | sh

# from source
git clone https://github.com/furkanc/parlando && cd parlando
uv run parlando              # run without installing
uv tool install .            # install commands from the checkout

# uninstall
uv tool uninstall parlando   # (or: pipx uninstall parlando / pip uninstall parlando)
```
</details>

## Quick start

```bash
parlando            # menu bar app: icon top right, settings in its menu
parlando --terminal # ...or dictate from this terminal window
```

1. On first run the speech model (~2 GB) downloads once; after that
   everything works offline.
2. macOS will ask for **Microphone** and **Accessibility** permissions
   (see below) — grant them and restart parlando.
3. Click the window you want to type into → tap **right ⌥ Option** →
   speak → tap again. Your words appear at the cursor.

### Permissions (one-time)

macOS will ask for two permissions, granted to the app that runs parlando
(Terminal, iTerm, VS Code, ...):

| Permission | Why | Where |
|------------|-----|-------|
| Microphone | hear you | Settings → Privacy & Security → Microphone |
| Accessibility | type keystrokes + global hotkey | Settings → Privacy & Security → Accessibility |

parlando detects missing permissions and tells you explicitly instead of
failing silently.

## Usage

```bash
parlando                      # menu bar app (default)
parlando --install-login      # start the menu bar app at login
parlando --terminal           # dictate from the terminal; the options below need it
parlando -t --language English   # dictate in another language
parlando -t --polish             # LLM cleanup (fillers, false starts, punctuation)
parlando -t --enter              # press Enter after each utterance
parlando -t --mode stream        # live word-by-word streaming
parlando --pipe                  # print to stdout (scriptable; implies --terminal)
parlando -t --hotkey cmd_r       # tap right Command instead
parlando --list-devices          # list microphones
```

### Voice commands

| Say (English) | Say (Turkish) | Result |
|---|---|---|
| period / comma | nokta / virgül | `.` `,` appended to previous word |
| question mark | soru işareti | `?` appended |
| new line / new paragraph | yeni satır / yeni paragraf | line break |
| send | gönder | presses Enter |

Disable with `--no-commands`.

### Models

**Speech recognition** (`--model`, MLX Qwen3-ASR — 30 languages):

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Qwen3-ASR-1.7B-8bit` | ~2 GB | **default** — best accuracy, recommended |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | ~700 MB | good balance for low-RAM machines |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | ~400 MB | fastest, lowest accuracy |
| `mlx-community/Qwen3-ASR-0.6B-bf16` | ~1.3 GB | higher fidelity 0.6B, more RAM |

**Polish LLM** (`--polish-model`, used only with `--polish`; scores from
`scripts/eval_polish.py`, an 11-case cleanup benchmark):

| Model | Size | Eval | Latency | Notes |
|---|---|---|---|---|
| `mlx-community/Qwen3-1.7B-4bit` | ~1 GB | 10/11 | ~0.4 s | **default** — fast, safe |
| `mlx-community/Qwen3-4B-4bit` | ~2.3 GB | 11/11 | ~0.9 s | best quality; use on 16 GB+ Macs |
| `mlx-community/Qwen3-0.6B-4bit` | ~350 MB | — | ~0.2 s | minimal RAM, weakest cleanup |

```bash
parlando -t --polish --polish-model mlx-community/Qwen3-4B-4bit
```

All models download once from Hugging Face and are cached in
`~/.cache/huggingface`; everything runs on-device.

### The two modes

**`record` (default)** — tap, speak freely (pauses are fine), tap again; the
whole recording is transcribed once and typed in one go. Recordings longer
than 28 s are split at natural silence points (2 min cap). This mode has no
VAD guessing, so it is the most robust.

**`stream`** — always listening; an energy VAD (hysteresis, adaptive noise
floor, optional `--silero` hybrid) segments utterances and words appear as
you speak. More "live", more sensitive to room noise.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Nothing typed, "only silence" warning | Grant microphone permission, restart |
| Nothing typed, no warning | Grant Accessibility permission (see startup warning) |
| Stalls after sleep | The watchdog reopens the stream within ~5 s automatically |
| Ghost text while silent (stream mode) | `--energy-floor 0.008` or `--silero` |
| Missing soft speech (stream mode) | `--energy-floor 0.002` |
| Wrong words | Move closer to the mic; prefer the built-in mic over AirPods (Bluetooth input drops to a low-quality codec) |
| Detailed trace | `~/Library/Logs/parlando.log` |

Note: `--device` indices shift when devices connect/disconnect, and virtual
devices ("Microsoft Teams Audio", "ZoomAudioDevice") are not microphones.
Prefer the default device.

## Architecture

```
microphone ── 30 ms frames ──▶ hotkey-gated recorder (record mode)
                              or energy VAD w/ hysteresis + adaptive floor,
                              optional Silero hybrid, 300 ms pre-roll (stream)
                │ segment boundaries
                ▼
        segment audio only ──▶ Qwen3-ASR (MLX, on-device GPU)
                │ readings
                ▼
        LocalAgreement-2 ──▶ append-only word commits ──▶ CGEvent unicode
        (commit on 2-reading    (typed text is never        keyboard events
         agreement)              retracted)
```

Key decisions (each one earned by a real failure during development):

- **Segment-based ASR, not a rolling window** — re-decoding a growing window
  revises earlier words (visible flicker) and hallucinates on silence.
- **Append-only, word-count-based commits** — exact-prefix matching stalls
  permanently when the ASR reshapes punctuation on committed words.
- **Normalized agreement + stall safety valve** — punctuation/case flapping
  between readings must not freeze the stream.
- **Asymmetric noise-floor learning** — a symmetric EMA lets speech scraps
  poison the floor and the app goes progressively deaf.
- **Every loop iteration guarded** — a single ASR exception must not silently
  kill the pipeline.
- **Chord-aware hotkey** — ⌥+Q-style character chords never toggle dictation.

## Development

```bash
uvx --with numpy pytest tests -q   # unit tests; no model/mic needed
uv run scripts/eval_polish.py      # polish quality benchmark (real local LLM)
uv build                           # build the wheel/sdist
```

Package layout: `src/parlando/` (engine, menubar, ASR engine). Entry point
`parlando` (menu bar app by default, `--terminal` for the CLI);
`parlando-menubar` is kept as an alias. CI runs the test suite and a packaging
build on macOS via GitHub Actions.

## Roadmap

- [x] One-line installer (`install.sh`) with `parlando` launcher command
- [x] Proper Python package (`uv tool install`, entry points)
- [x] Publish to PyPI (`uvx parlando` works)
- [ ] Homebrew tap (`brew install parlando`)
- [ ] End-to-end regression tests with recorded WAV fixtures
- [ ] Custom vocabulary / context biasing

## License & credits

MIT. The Qwen3-ASR MLX engine (`stt.py`) is from
[dictate.sh](https://github.com/mpuig/dictate.sh) by Marc Puig (MIT), with
local modifications (partial streaming, energy gating, error hardening).
Turn-taking research: [Whisper-Streaming /
LocalAgreement-2](https://arxiv.org/abs/2307.14743). VAD:
[Silero VAD](https://github.com/snakers4/silero-vad).
