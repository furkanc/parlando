"""parlando — local, fluent voice dictation for Apple Silicon (MLX).

Speak; your words are typed into the focused window (your editor, Claude
Code, any app). Fully local: audio never leaves your machine.

Design (dictation best practices):
  - Record mode (default): tap a hotkey, speak freely, tap again — the whole
    recording is transcribed once and typed at the cursor. This is the
    pattern commercial dictation apps use; there is no VAD guessing, so
    "it stopped hearing me" failure modes are structurally impossible.
  - Stream mode: utterance-based ASR. An energy VAD (hysteresis + adaptive
    noise floor, optional Silero hybrid) segments speech; ONLY that segment
    is transcribed, committed, and dropped. No rolling-window re-decoding.
  - Append-only commits: typed text is never retracted. Words are stabilized
    with LocalAgreement-2 (committed once two consecutive readings agree).
  - Self-healing: inference errors never kill the loop; an audio watchdog
    reopens the stream after sleep/device changes.
  - Typing: Quartz CGEvent unicode events (fast, layout-safe for any
    language); falls back to osascript automatically.
  - Guards: energy gate, repetition detection, sub-250ms noise rejection,
    capped normalization, microphone-permission diagnosis.
  - Cleanup: vocalized fillers (um, uh, eee) are removed deterministically;
    optional --polish runs a small local LLM at finalize to drop contextual
    fillers and false starts, guarded so it can only *edit*, never answer.

Usage:
    parlando                    # menu bar app: icon top right, settings in its menu
    parlando --install-app      # create ~/Applications/Parlando.app (recommended)
    parlando --install-login    # start the menu bar app at login
    parlando --terminal         # dictate from this terminal window instead
    parlando -t --language English  # terminal options (all need --terminal / -t)
    parlando -t --enter         # press Enter after each utterance
    parlando --pipe             # print to stdout instead of typing (implies -t)
    parlando -t --polish        # LLM cleanup of fillers and false starts
    parlando -t --silero        # hybrid Silero VAD (noisy rooms)
    parlando --list-devices     # list input devices

Global hotkey (default: single tap of right Option): start/stop recording.

Voice commands (disable with --no-commands) follow the language: the Turkish
set is active by default ("nokta", "virgül", "soru işareti", "yeni satır",
"gönder"); the English set below is active with --language English:
    "period" "comma" "exclamation mark"  -> append . , ! to previous word
    "question mark"                      -> append ? to previous word
    "new line" / "new paragraph"         -> line break
    "send" (at the end)                  -> press Enter

Requirements (macOS):
  - Microphone permission: Settings > Privacy & Security > Microphone.
  - Accessibility permission (typing + hotkey): ... > Accessibility.

Log: ~/Library/Logs/parlando.log
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parlando import __version__

# huggingface_hub reads this once at import: keep its progress bars off
# (the engine reports download progress itself; see _ensure_model).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Heavy dependencies (mlx, sounddevice, Quartz, pynput, onnxruntime, stt)
# are imported lazily on purpose: --help stays fast, unit tests need no
# model or permissions, and CI only needs numpy.

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
PREROLL_FRAMES = 10          # ~300ms of history prepended to each utterance
MIN_SPEECH_MS = 250          # anything shorter is a click/cough; dropped
MIN_PARTIAL_SECONDS = 0.6    # minimum audio before partial ASR
INT16_SCALE = 1.0 / 32768.0
WATCHDOG_SECONDS = 5.0       # reopen the stream if no frames for this long
AX_POLL_SECONDS = 2.0        # re-check Accessibility while it is missing
MIC_ZERO_FRAMES = 100        # ~3s of pure zeros -> permission warning
MAX_RECORD_SECONDS = 120     # hard safety cap for record mode
CHUNK_SECONDS = 28           # model window; longer recordings are split
ENTER_TOKEN = ""       # invisible marker for the "send" voice command
LOG_PATH = Path.home() / "Library" / "Logs" / "parlando.log"
VOCAB_FILE = Path.home() / ".config" / "parlando" / "vocabulary.txt"


def load_vocabulary(extra: str = "") -> list[str]:
    """Personal vocabulary: exact spellings the ASR should preserve.

    Merges ~/.config/parlando/vocabulary.txt (one term per line, `#`
    comments allowed) with the comma-separated `--vocab` argument. Fixes
    the classic ASR failure of spelling technical terms phonetically
    ("MLX" -> "Meleiks").
    """
    terms: list[str] = []
    try:
        if VOCAB_FILE.exists():
            for line in VOCAB_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    except OSError:
        pass
    for t in extra.split(","):
        t = t.strip()
        if t:
            terms.append(t)
    seen: set = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique


def vocab_context(terms: list[str]) -> str:
    """System-prompt context that biases Qwen3-ASR toward exact spellings."""
    if not terms:
        return ""
    # Deliberately minimal: Qwen3-ASR biases toward context text but does
    # not follow instructions — prose here just becomes parroting material
    # when the audio is quiet (it once typed the whole sentence verbatim).
    # The echo guard (is_context_echo) is the real protection.
    return "Vocabulary: " + ", ".join(terms)


def is_context_echo(text: str, context: str) -> bool:
    """Did the ASR echo its own context instead of transcribing?

    On quiet/contentless audio the decoder can copy the injected vocabulary
    context into the transcript. We know the exact injected string, so this
    is decidable: drop transcripts that are a substring of the context or
    consist almost entirely of context words. Short outputs are never
    dropped — dictating a single vocabulary term must keep working.
    """
    if not text or not context:
        return False
    tw = [_bare(w) for w in text.split() if _bare(w)]
    if len(tw) < 4:
        return False
    ctx_words = [_bare(w) for w in context.split() if _bare(w)]
    if " ".join(tw) in " ".join(ctx_words):
        return True
    ctx_set = set(ctx_words)
    inside = sum(1 for w in tw if w in ctx_set)
    return len(tw) >= 6 and inside / len(tw) >= 0.85

LOGGER = logging.getLogger("parlando")


@dataclass
class Config:
    model: str = "mlx-community/Qwen3-ASR-1.7B-8bit"
    language: str = "Turkish"
    device: int | None = None
    interval: float = 0.5        # partial ASR cadence (s), stream mode
    silence_ms: int = 500        # silence that ends an utterance, stream mode
    energy_floor: float = 0.004  # absolute minimum RMS threshold
    max_utterance_s: float = 20.0
    normalize: bool = True
    send_enter: bool = False
    pipe: bool = False           # print to stdout instead of typing
    # Single key names (alt_r, cmd_r, ctrl_r, shift_r) mean SINGLE TAP
    # (the pattern commercial dictation apps use); pynput combos such as
    # "<ctrl>+<alt>+d" also work.
    hotkey: str | None = "alt_r"
    commands: bool = True        # voice commands (period, new line, send)
    cleanup: bool = True         # drop unambiguous vocalized fillers (um, eee)
    polish: bool = False         # LLM cleanup pass at finalize (record mode)
    polish_model: str = "mlx-community/Qwen3-1.7B-4bit"
    vocab: str = ""              # extra terms, comma-separated (adds to file)
    silero: bool = False         # hybrid Silero VAD
    start_paused: bool = False
    # "record": tap to start/stop, transcribe once (default; most robust).
    # "stream": always listening, words appear as you speak.
    mode: str = "record"


@dataclass
class Status:
    """What the engine is doing right now, for shells (menu bar, terminal).

    phase: starting | downloading | loading | ready | recording |
           transcribing | paused
    problems: things the user must fix, e.g. "accessibility", "microphone".
    downloaded/total: bytes, meaningful while phase == "downloading".
    """

    phase: str = "starting"
    detail: str = ""
    downloaded: int = 0
    total: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def progress(self) -> float | None:
        if self.phase != "downloading" or not self.total:
            return None
        return min(1.0, self.downloaded / self.total)


def short_model_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / 1024:.0f} KB"


# Files a model download needs; the same patterns stt.py / mlx_lm use, so
# the pre-download here makes their own snapshot_download a cache hit.
ASR_MODEL_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]
LLM_MODEL_PATTERNS = ["*.json", "*.safetensors", "*.py", "tokenizer.model",
                      "*.tiktoken", "*.txt", "*.jsonl"]


def progress_tqdm_class(status: Status):
    """A tqdm subclass that reports byte progress into `status`.

    huggingface_hub drives several bars: one per snapshot (files, unit
    "it"), and byte bars (unit "B") whose `total` may be set after
    construction. With hf_xet there are two byte bars for the same bytes,
    "Downloading bytes" and "Reconstructing ..."; only the first counts.
    Display is disabled by HF_HUB_DISABLE_PROGRESS_BARS; update() is still
    called, so each bar keeps its own byte counter here.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    bars: dict[int, ProgressTqdm] = {}

    def recompute() -> None:
        status.downloaded = sum(b._bytes for b in bars.values())
        status.total = sum(int(b.total or 0) for b in bars.values())

    class ProgressTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            desc = str(kwargs.get("desc") or "").lower()
            self._counted = kwargs.get("unit") == "B" and "reconstruct" not in desc
            self._bytes = int(kwargs.get("initial") or 0)
            if self._counted:
                bars[id(self)] = self
                recompute()

        def update(self, n=1):
            if self._counted and n:
                self._bytes += int(n)
                recompute()
            return super().update(n)

    return ProgressTqdm


# -----------------------------------------------------------------------------
# Text stabilization, guards, voice commands
# -----------------------------------------------------------------------------


def collapse_repeats(words: list[str], max_run: int = 2) -> list[str]:
    """Collapse stutters and ASR stuck-loops deterministically.

    A run of 3+ identical words is essentially never legitimate — it is a
    stutter ("I I I want") or a hallucination loop — so it collapses to ONE
    occurrence. Doubles ("çok çok", "that that") can be real emphasis or
    grammar and are left for the context-aware LLM polish step.
    """
    out: list[str] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        run = j - i
        out.extend(words[i:j] if run <= max_run else [words[i]])
        i = j
    return out


def looks_hallucinated(words: list[str]) -> bool:
    """Long output made of almost a single word is a hallucination."""
    if len(words) < 8:
        return False
    unique = len({w.lower() for w in words})
    return unique / len(words) < 0.35


# Per-language voice command vocabularies. Transformation is deterministic:
# the same input always yields the same output, which keeps it compatible
# with LocalAgreement stabilization.
_VOICE_COMMANDS: dict[str, dict] = {
    "English": {
        "punct": {"period": ".", "comma": ",", "exclamation": "!"},
        "two_word": {
            ("question", "mark"): "?",
            ("exclamation", "mark"): "!",
            ("exclamation", "point"): "!",
            ("new", "line"): "\n",
            ("new", "paragraph"): "\n\n",
        },
        "enter": {"send"},
    },
    "Turkish": {
        "punct": {
            "nokta": ".",
            "virgül": ",",
            "virgul": ",",
            "ünlem": "!",
            "unlem": "!",
        },
        "two_word": {
            ("soru", "işareti"): "?",
            ("soru", "isareti"): "?",
            ("yeni", "satır"): "\n",
            ("yeni", "satir"): "\n",
            ("yeni", "paragraf"): "\n\n",
        },
        "enter": {"gönder", "gonder"},
    },
}


def _bare(word: str) -> str:
    return word.lower().strip(".,!?;:")


# Unambiguous vocalized fillers, safe to drop without context. Contextual
# fillers ("like", "you know", "yani", "şey") are left to the LLM polish
# step, which can tell filler use from real use.
_FILLER_WORDS: dict[str, set] = {
    "English": {"um", "uh", "uhm", "umm", "er", "erm", "hmm", "mhm", "mmm"},
    "Turkish": {"eee", "ee", "ıı", "ııı", "hmm", "mmm", "iiı", "eem"},
}


def remove_fillers(words: list[str], language: str = "English") -> list[str]:
    fillers = _FILLER_WORDS.get(language, _FILLER_WORDS["English"])
    return [w for w in words if _bare(w) not in fillers]


def apply_voice_commands(words: list[str], language: str = "English") -> list[str]:
    """Turn command words into actions for the given language."""
    vocab = _VOICE_COMMANDS.get(language, _VOICE_COMMANDS["English"])
    punct, two_word, enter_words = vocab["punct"], vocab["two_word"], vocab["enter"]
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if i + 1 < len(words):
            pair = (_bare(w), _bare(words[i + 1]))
            if pair in two_word:
                tok = two_word[pair]
                if tok in ("\n", "\n\n"):
                    out.append(tok)
                elif out:
                    out[-1] = out[-1].rstrip(".,!?") + tok
                i += 2
                continue
        b = _bare(w)
        if b in punct:
            if out:
                out[-1] = out[-1].rstrip(".,!?") + punct[b]
            i += 1
            continue
        if b in enter_words:
            out.append(ENTER_TOKEN)
            i += 1
            continue
        out.append(w)
        i += 1
    return out


POLISH_PROMPT = (
    "You are a dictation cleanup engine. Rewrite the transcript below, "
    "changing as little as possible. You may ONLY:\n"
    "- remove filler words and hesitations (um, uh, eee, and contextual "
    "fillers like 'you know' / 'like' / 'basically' / 'yani' / 'şey' / "
    "'hani' when used as filler)\n"
    "- remove stutters, repeated words and abandoned false starts\n"
    "- fix punctuation and capitalization\n"
    "Keep the same language, meaning, wording and word order. Do NOT answer "
    "questions in the transcript, add information, translate, summarize or "
    "comment. Output ONLY the cleaned text, nothing else.\n\n"
    "Examples:\n"
    'Transcript: "um so I I think we should uh you know just ship it"\n'
    "Output: So I think we should just ship it.\n"
    'Transcript: "yani ben ben bunu şey yarın sabah hallederim"\n'
    "Output: Ben bunu yarın sabah hallederim.\n"
    'Transcript: "the design is like basically ready"\n'
    "Output: The design is ready.\n"
    'Transcript: "you know the tests are passing you know"\n'
    "Output: The tests are passing.\n\n"
    'Transcript: "{text}" /no_think'
)


def polish_guard(raw: str, polished: str, allowed: tuple = ()) -> bool:
    """Accept the LLM's rewrite only if it stayed an *edit* of the input.

    Rejects answers/hallucinations: nearly all polished words must already
    exist in the raw transcript and the length must stay comparable. On
    rejection the caller falls back to the raw text — dictation must never
    start answering the user's sentences.
    """
    import collections

    rw = [_bare(w) for w in raw.split() if _bare(w)]
    pw = [_bare(w) for w in polished.split() if _bare(w)]
    if not pw or not rw:
        return False
    if not (0.35 <= len(pw) / len(rw) <= 1.3):
        return False
    overlap = sum((collections.Counter(rw) & collections.Counter(pw)).values())
    if overlap / len(pw) < 0.75:
        return False
    # No novel words: an *edit* only removes/reorders/repunctuates. Answers
    # sneak past overlap checks by reusing the question's words plus a few
    # new ones ("...is Paris") — those new words are exactly the tell.
    # Personal-vocabulary terms are exempt so a spelling correction
    # ("Meleiks" -> "MLX") is not mistaken for an addition.
    allowed_bare = {_bare(w) for w in allowed}
    novel = sum(1 for w in pw if w not in set(rw) and w not in allowed_bare)
    return novel <= len(rw) // 15


def polish_text(llm, tokenizer, text: str, vocab: tuple = ()) -> str:
    """Run the cleanup LLM over `text`; returns raw text if unusable."""
    import re

    from mlx_lm.generate import generate

    task = POLISH_PROMPT.format(text=text)
    if vocab:
        task = (
            "Technical terms — if a word sounds like one of these, spell it "
            "exactly as written here: " + ", ".join(vocab) + ".\n\n" + task
        )
    messages = [{"role": "user", "content": task}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_tokens = max(64, int(len(text.split()) * 3) + 32)
    response = generate(llm, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = response.strip('"“”').strip()
    if response and polish_guard(text, response, allowed=vocab):
        return response
    return text


def render_typing(joined: str) -> str:
    """Build the keyboard-bound text from joined tokens.

    ENTER_TOKEN is invisible (turned into a key press at finalize); spaces
    around line breaks are cleaned up.
    """
    s = joined.replace(" " + ENTER_TOKEN, "").replace(ENTER_TOKEN, "")
    s = s.replace(" \n", "\n").replace("\n ", "\n")
    return s


def _norm_word(word: str) -> str:
    """Normalize for agreement comparison.

    As context grows, the ASR shifts punctuation and casing on earlier words
    ("today" <-> "Today,"). Exact comparison therefore stalls the stream
    permanently (a bug we hit); agreement is checked on the bare word.
    """
    return word.lower().strip(".,!?;:")


def _word_prefix_len(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and _norm_word(a[i]) == _norm_word(b[i]):
        i += 1
    return i


class WordCommitter:
    """Append-only, word-count-based commits (battle-tested in this project).

    Tracks how many words have been keystroked and only ever returns words
    beyond that count. A later ASR revision of an earlier word can neither
    delete text nor block the stream. Partials use LocalAgreement (with
    normalized comparison); if readings still refuse to agree for a while, a
    safety valve types everything but the last word so the flow never stalls.
    """

    STALL_READINGS = 3

    def __init__(self) -> None:
        self.n_typed = 0
        self.prev: list[str] = []
        self.stalled = 0

    def reading(self, words: list[str]) -> str:
        k = _word_prefix_len(self.prev, words)
        self.prev = words
        out = self._take(words[:k])
        if out:
            self.stalled = 0
            return out
        self.stalled += 1
        if self.stalled >= self.STALL_READINGS and len(words) > self.n_typed + 1:
            # Safety valve: no agreement but speech keeps accumulating;
            # type all but the last (unstable) word so the flow continues.
            out = self._take(words[:-1])
            if out:
                self.stalled = 0
        return out

    def flush(self, words: list[str]) -> str:
        out = self._take(words)
        self.prev = []
        self.stalled = 0
        return out

    def _take(self, words: list[str]) -> str:
        if len(words) <= self.n_typed:
            return ""
        lead = " " if self.n_typed > 0 else ""
        new = words[self.n_typed:]
        self.n_typed = len(words)
        return render_typing(lead + " ".join(new))

    def reset(self) -> None:
        self.n_typed = 0
        self.prev = []
        self.stalled = 0


# -----------------------------------------------------------------------------
# VAD: energy (hysteresis + adaptive floor), optional Silero hybrid
# -----------------------------------------------------------------------------


class EnergyVAD:
    """RMS-based speech segmentation; can be hybridized with Silero.

    Hysteresis above an adaptive noise floor (start threshold > stop
    threshold) plus debounce. The floor is learned via EMA while idle.
    """

    START_MULT = 3.5
    STOP_MULT = 2.0
    DEBOUNCE_FRAMES = 2
    SILERO_START = 0.6
    SILERO_STOP = 0.35

    def __init__(self, floor: float, silence_ms: int) -> None:
        self.floor = floor
        self.noise = floor
        self.silence_frames_needed = max(1, round(silence_ms / FRAME_MS))
        self.in_speech = False
        self.silence_run = 0
        self.start_run = 0

    @property
    def start_threshold(self) -> float:
        return max(self.floor, self.noise * self.START_MULT)

    @property
    def stop_threshold(self) -> float:
        return max(self.floor * 0.6, self.noise * self.STOP_MULT)

    def update(self, rms: float, prob: float | None = None) -> str:
        """Per-frame event: 'start' | 'end' | 'speech' | 'idle'.

        If `prob` is given (Silero), the speech decision comes from it and
        energy only keeps learning the noise floor.
        """
        if prob is not None:
            above = prob > self.SILERO_START
            below = prob < self.SILERO_STOP
        else:
            above = rms > self.start_threshold
            below = rms < self.stop_threshold

        if not self.in_speech:
            # Noise floor: loud frames (likely speech scraps) must not poison
            # the floor and push the threshold out of reach — follow down
            # fast, up very slowly and capped. (The symmetric version caused
            # progressive deafness via positive feedback: a bug we hit.)
            if rms < self.noise:
                self.noise = max(5e-5, 0.95 * self.noise + 0.05 * rms)
            else:
                self.noise = min(self.noise * 1.002, self.floor * 1.5)
            if above:
                self.start_run += 1
                if self.start_run >= self.DEBOUNCE_FRAMES:
                    self.in_speech = True
                    self.silence_run = 0
                    self.start_run = 0
                    return "start"
            else:
                self.start_run = 0
            return "idle"

        if below:
            self.silence_run += 1
            if self.silence_run >= self.silence_frames_needed:
                self.in_speech = False
                self.silence_run = 0
                return "end"
        else:
            self.silence_run = 0
        return "speech"


class SileroVAD:
    """Silero VAD (ONNX, CPU). Wants 512-sample windows; stateful."""

    REPO_ID = "deepghs/silero-vad-onnx"
    MODEL_FILE = "silero_vad.onnx"
    WINDOW = 512
    CONTEXT = 64

    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=self.REPO_ID, filename=self.MODEL_FILE)
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=so)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def reset(self) -> None:
        self._state[:] = 0.0
        self._context[:] = 0.0

    def __call__(self, chunk_f32: np.ndarray) -> float:
        inp = np.concatenate([self._context, chunk_f32])[np.newaxis, :]
        outs = self.session.run(
            None, {"input": inp, "state": self._state, "sr": self._sr}
        )
        self._state = outs[1]
        self._context = chunk_f32[-self.CONTEXT:]
        return float(outs[0].item())


# -----------------------------------------------------------------------------
# Output: typing into the focused window
# -----------------------------------------------------------------------------


class QuartzTypist:
    """CGEvent unicode typing: no subprocess, fast, layout/character safe."""

    CHUNK = 20

    def __init__(self) -> None:
        import Quartz

        self._q = Quartz

    def _post_text(self, text: str) -> None:
        q = self._q
        for i in range(0, len(text), self.CHUNK):
            seg = text[i : i + self.CHUNK]
            for down in (True, False):
                ev = q.CGEventCreateKeyboardEvent(None, 0, down)
                q.CGEventKeyboardSetUnicodeString(ev, len(seg), seg)
                q.CGEventPost(q.kCGHIDEventTap, ev)
            time.sleep(0.005)

    def _post_key(self, code: int) -> None:
        q = self._q
        for down in (True, False):
            ev = q.CGEventCreateKeyboardEvent(None, code, down)
            q.CGEventPost(q.kCGHIDEventTap, ev)

    async def type(self, text: str) -> None:
        if text:
            await asyncio.to_thread(self._post_text, text)

    async def press_enter(self) -> None:
        await asyncio.to_thread(self._post_key, 36)


class OsascriptTypist:
    """Fallback typist used when Quartz (pyobjc) cannot be loaded."""

    _TYPE_SCRIPT = (
        "on run argv\n"
        'tell application "System Events" to keystroke (item 1 of argv)\n'
        "end run"
    )

    async def type(self, text: str) -> None:
        if not text:
            return
        await asyncio.to_thread(
            subprocess.run,
            ["osascript", "-e", self._TYPE_SCRIPT, text],
            check=False,
            capture_output=True,
        )

    async def press_enter(self) -> None:
        await asyncio.to_thread(
            subprocess.run,
            ["osascript", "-e", 'tell application "System Events" to key code 36'],
            check=False,
            capture_output=True,
        )


def make_typist():
    try:
        return QuartzTypist()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Quartz unavailable (%r); falling back to osascript", exc)
        return OsascriptTypist()


TAP_KEY_LABELS = {
    "alt_r": "right ⌥ Option",
    "cmd_r": "right ⌘ Command",
    "ctrl_r": "right ⌃ Control",
    "shift_r": "right ⇧ Shift",
}


def hotkey_label(hotkey: str | None) -> str:
    if not hotkey:
        return "no hotkey"
    if hotkey in TAP_KEY_LABELS:
        return f"{TAP_KEY_LABELS[hotkey]} (single tap)"
    return hotkey


class TapDetector:
    """Single-modifier 'tap' detection.

    Triggers only when the key is released quickly WITHOUT any other key
    pressed in between, so character chords such as ⌥+Q never toggle
    dictation by accident.
    """

    def __init__(self, timeout: float = 0.6) -> None:
        self.timeout = timeout
        self.down = False
        self.chorded = False
        self.t = 0.0

    def press(self, is_target: bool) -> None:
        if is_target:
            self.down = True
            self.chorded = False
            self.t = time.monotonic()
        elif self.down:
            self.chorded = True

    def release(self, is_target: bool) -> bool:
        """Returns True when the release counts as a tap (toggle)."""
        if not (is_target and self.down):
            return False
        self.down = False
        return not self.chorded and (time.monotonic() - self.t) < self.timeout


def accessibility_trusted(prompt: bool = False) -> bool | None:
    """Is Accessibility permission granted? None = could not determine.

    With `prompt=True` macOS shows its own "wants to control this computer"
    dialog and adds the responsible app (the terminal, or Parlando.app) to
    the Accessibility list, so the user only has to flip the switch.
    """
    try:
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )

        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))
    except Exception:  # noqa: BLE001
        return None


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------


class DictationEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.dropped = 0

        self.vad = EnergyVAD(cfg.energy_floor, cfg.silence_ms)
        self.silero: SileroVAD | None = None
        self._silero_buf = np.empty(0, dtype=np.float32)
        self.silero_prob: float | None = None

        self.preroll: collections.deque = collections.deque(maxlen=PREROLL_FRAMES)
        self.utt: list[np.ndarray] = []
        self.utt_samples = 0
        self.speech_frames = 0
        self.finalize_pending = False
        self.last_partial = 0.0

        self.committer = WordCommitter()
        self.typist = None  # created in run()
        self.vocab_terms = tuple(load_vocabulary(cfg.vocab))
        self._vocab_mtime = self._vocab_file_mtime()

        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self._stt = None
        self.llm = None            # polish LLM (loaded only with cfg.polish)
        self.llm_tokenizer = None

        self.paused = cfg.start_paused
        self.on_state = None  # callback for shells such as the menu bar app

        # Record-mode (tap-speak-tap) state.
        self.recording = False
        self.rec_frames: list[np.ndarray] = []
        self.rec_samples = 0
        self.rec_silence_marks: list[int] = []  # split-point candidates
        self._rec_silence_run = 0
        self._rec_finalize = False

        self.total_words = 0
        self.asr_ms: float | None = None
        self._status_tty = sys.stderr.isatty()
        self._last_status = ""
        self._mic_zero_count = 0
        self._mic_warned = False
        self._stream_restarts = 0
        self.accessibility_missing = False  # read by the menu bar shell
        self._hotkey = None                 # pynput listener, if any
        self.status = Status()              # read by shells (menu bar, tests)

    # -- status / logging -----------------------------------------------------

    def _status(self, text: str) -> None:
        if not self._status_tty or text == self._last_status:
            return
        self._last_status = text
        sys.stderr.write(f"\r\x1b[K{text}")
        sys.stderr.flush()

    def _err(self, message: str) -> None:
        LOGGER.error(message)
        sys.stderr.write(f"\r\x1b[K[error] {message}\n")
        sys.stderr.flush()
        self._last_status = ""

    def _info(self, message: str) -> None:
        LOGGER.info(message)
        sys.stderr.write(f"\r\x1b[K{message}\n")
        sys.stderr.flush()
        self._last_status = ""

    def _set_phase(self, phase: str, detail: str = "") -> None:
        self.status.phase = phase
        self.status.detail = detail
        if self.on_state:
            self.on_state(phase)

    def _add_problem(self, name: str) -> None:
        if name not in self.status.problems:
            self.status.problems.append(name)

    def _clear_problem(self, name: str) -> None:
        if name in self.status.problems:
            self.status.problems.remove(name)

    # -- pause / resume, record toggle ---------------------------------------

    def toggle_pause(self) -> None:
        """Not thread-safe; call from other threads via request_toggle()."""
        self.paused = not self.paused
        if self.paused:
            self.utt = []
            self.utt_samples = 0
            self.speech_frames = 0
            self.finalize_pending = False
            self.vad.in_speech = False
            self.committer.reset()
            if self.silero:
                self.silero.reset()
        sound = "Bottle" if self.paused else "Pop"
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LOGGER.info("dictation %s", "paused" if self.paused else "resumed")
        self._set_phase("paused" if self.paused else "ready")

    def _vocab_file_mtime(self) -> float:
        try:
            return VOCAB_FILE.stat().st_mtime
        except OSError:
            return 0.0

    def reload_vocab_if_changed(self) -> None:
        """Pick up edits to the vocabulary file without a restart.

        A cheap mtime check; called when a recording starts, so an edit is
        live by the next dictation.
        """
        mtime = self._vocab_file_mtime()
        if mtime != self._vocab_mtime:
            self._vocab_mtime = mtime
            self.vocab_terms = tuple(load_vocabulary(self.cfg.vocab))
            LOGGER.info("vocabulary reloaded: %d terms", len(self.vocab_terms))

    def add_vocab_term(self, term: str) -> bool:
        """Append a term to the vocabulary file and activate it immediately.

        Used by the menu bar's "Add Term…"; returns False for empty or
        already-known terms.
        """
        term = term.strip()
        if not term or term.lower() in {t.lower() for t in self.vocab_terms}:
            return False
        try:
            VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing = (
                VOCAB_FILE.read_text(encoding="utf-8") if VOCAB_FILE.exists() else ""
            )
            sep = "" if (not existing or existing.endswith("\n")) else "\n"
            with open(VOCAB_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"{sep}{term}\n")
        except OSError as exc:
            LOGGER.error("could not write vocabulary file: %r", exc)
            return False
        self._vocab_mtime = self._vocab_file_mtime()
        self.vocab_terms = tuple(load_vocabulary(self.cfg.vocab))
        LOGGER.info("vocabulary term added: %s", term)
        return True

    def toggle_record(self) -> None:
        """Record-mode hotkey: start recording / stop and type."""
        if self.recording:
            self.recording = False
            self._rec_finalize = True
            sound = "Bottle"
        else:
            self.reload_vocab_if_changed()
            self.recording = True
            self.rec_frames = []
            self.rec_samples = 0
            self.rec_silence_marks = []
            self._rec_silence_run = 0
            sound = "Pop"
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LOGGER.info("recording %s", "stopped" if not self.recording else "started")
        self._set_phase("recording" if self.recording else "transcribing")

    def toggle_action(self) -> None:
        if self.cfg.mode == "record":
            self.toggle_record()
        else:
            self.toggle_pause()

    def request_toggle(self) -> None:
        """Thread-safe start/stop from any thread."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.toggle_action)

    # -- audio input ----------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        data = indata.reshape(-1).copy()
        try:
            self.loop.call_soon_threadsafe(self._enqueue, data)
        except RuntimeError:
            pass  # loop is shutting down; drop the frame silently

    def _enqueue(self, data: np.ndarray) -> None:
        if self.queue.full():
            self.dropped += 1
            return
        self.queue.put_nowait(data)

    def _open_stream(self):
        import sounddevice as sd

        kwargs = {}
        if self.cfg.device is not None:
            kwargs["device"] = self.cfg.device
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
            **kwargs,
        )
        stream.start()
        return stream

    async def _reopen_stream(self, stream, quiet: bool = False):
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)
        try:
            stream = self._open_stream()
            if not quiet:
                self._info("Audio stream reopened.")
        except Exception as exc:  # noqa: BLE001
            self._err(f"Stream reopen failed: {exc!r}; retrying in 3s")
            await asyncio.sleep(3)
        return stream

    def _check_mic_silence(self, frame: np.ndarray) -> None:
        """Without mic permission macOS delivers zeros, not errors — warn.

        Checked during the first ~3 s only (real microphones never deliver
        exact zeros); once warned, the first real audio clears the problem.
        """
        if self._mic_zero_count > MIC_ZERO_FRAMES and not self._mic_warned:
            return  # audio was seen at startup; check done
        if int(np.max(np.abs(frame))) < 10:
            if self._mic_warned:
                return
            self._mic_zero_count += 1
            if self._mic_zero_count == MIC_ZERO_FRAMES:
                self._mic_warned = True
                self._add_problem("microphone")
                self._err(
                    "Only silence is coming from the microphone. Most likely "
                    "the mic permission is missing: System Settings > Privacy "
                    "& Security > Microphone > allow the app that runs "
                    "parlando (Parlando.app, or your terminal)."
                )
        else:
            if self._mic_warned:
                self._mic_warned = False
                self._clear_problem("microphone")
                self._info("Microphone audio is flowing.")
            self._mic_zero_count = MIC_ZERO_FRAMES + 1

    # -- ASR ------------------------------------------------------------------

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 0.03:
            return audio
        gain = min(0.3 / peak, 8.0)
        if gain <= 1.0:
            return audio
        return np.clip(audio * gain, -1.0, 1.0)

    def _transcribe_blocking(self, audio_int16: np.ndarray) -> str:
        audio = audio_int16.astype(np.float32) * INT16_SCALE
        if self.cfg.normalize:
            audio = self._normalize_audio(audio)
        ctx = vocab_context(list(self.vocab_terms))
        parts = []
        with self._stt._suppress_output():
            for token in self._stt.transcribe(
                self.model,
                self.tokenizer,
                self.feature_extractor,
                audio,
                self.cfg.language,
                context=ctx,
            ):
                parts.append(token)
        text = "".join(parts).strip()
        if is_context_echo(text, ctx):
            LOGGER.info("dropped context echo: %s", text)
            return ""
        return text

    async def _asr(self, audio_int16: np.ndarray) -> str:
        """ASR; an error must never kill the loop."""
        try:
            start = time.perf_counter()
            text = await asyncio.to_thread(self._transcribe_blocking, audio_int16)
            self.asr_ms = (time.perf_counter() - start) * 1000
            return text
        except Exception as exc:  # noqa: BLE001 — one failure must not stop the flow
            self._err(f"ASR: {exc!r}")
            return ""

    def _clean_words(self, text: str) -> list[str]:
        raw = text.split()
        # Check the RAW output first: collapsing repeats first would let the
        # hallucination slip under the uniqueness check.
        if looks_hallucinated(raw):
            return []
        words = collapse_repeats(raw)
        if self.cfg.cleanup:
            words = remove_fillers(words, self.cfg.language)
        if self.cfg.commands:
            words = apply_voice_commands(words, self.cfg.language)
        return words

    # -- segment handling -----------------------------------------------------

    def _on_frame_record(self, frame: np.ndarray) -> None:
        """Record mode: no VAD guessing; EVERYTHING between taps is kept."""
        if not self.recording:
            return
        self.rec_frames.append(frame)
        self.rec_samples += len(frame)
        # Mark silence moments so long recordings can be split at natural
        # boundaries to fit the model window.
        rms = float(
            np.sqrt(np.mean(np.square(frame.astype(np.float32) * INT16_SCALE)))
        )
        if rms < self.cfg.energy_floor:
            self._rec_silence_run += 1
            if self._rec_silence_run == 10:  # ~300ms of silence
                self.rec_silence_marks.append(self.rec_samples)
        else:
            self._rec_silence_run = 0
        if self.rec_samples >= MAX_RECORD_SECONDS * SAMPLE_RATE:
            self.recording = False
            self._rec_finalize = True

    def _on_frame(self, frame: np.ndarray) -> None:
        self._check_mic_silence(frame)
        if self.cfg.mode == "record":
            self._on_frame_record(frame)
            return
        if self.paused:
            return

        f32 = frame.astype(np.float32) * INT16_SCALE
        rms = float(np.sqrt(np.mean(np.square(f32))))

        prob = None
        if self.silero is not None:
            self._silero_buf = np.concatenate([self._silero_buf, f32])
            while len(self._silero_buf) >= SileroVAD.WINDOW:
                chunk = self._silero_buf[: SileroVAD.WINDOW]
                self._silero_buf = self._silero_buf[SileroVAD.WINDOW :]
                try:
                    self.silero_prob = self.silero(chunk)
                except Exception as exc:  # noqa: BLE001
                    self._err(f"Silero VAD: {exc!r}; falling back to energy VAD")
                    self.silero = None
                    self.silero_prob = None
            prob = self.silero_prob

        event = self.vad.update(rms, prob)

        if event == "start":
            self.reload_vocab_if_changed()
            self.utt = list(self.preroll)
            self.utt.append(frame)
            self.utt_samples = sum(len(f) for f in self.utt)
            self.speech_frames = 1
            self.last_partial = self.loop.time()
        elif event == "speech":
            self.utt.append(frame)
            self.utt_samples += len(frame)
            if rms >= self.vad.stop_threshold:
                self.speech_frames += 1
            if self.utt_samples >= int(self.cfg.max_utterance_s * SAMPLE_RATE):
                self.finalize_pending = True
        elif event == "end":
            self.utt.append(frame)
            self.utt_samples += len(frame)
            self.finalize_pending = True
            if self.silero:
                self.silero.reset()
                self.silero_prob = None
        else:  # idle
            self.preroll.append(frame)

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split recordings longer than the model window into chunks.

        Cuts happen at the marked silence moment closest to the target so no
        word gets sliced in half; falls back to a hard cut if none exists.
        """
        max_chunk = CHUNK_SECONDS * SAMPLE_RATE
        if len(audio) <= max_chunk:
            return [audio]
        chunks = []
        start = 0
        while len(audio) - start > max_chunk:
            target = start + max_chunk
            candidates = [m for m in self.rec_silence_marks if start < m <= target]
            cut = candidates[-1] if candidates else target
            chunks.append(audio[start:cut])
            start = cut
        chunks.append(audio[start:])
        return [c for c in chunks if len(c) > 0]

    def _recording_has_speech(self, audio: np.ndarray) -> bool:
        """Does the recording contain at least MIN_SPEECH_MS of audio above
        the energy floor? Deterministic; independent of the ASR."""
        n = len(audio) // FRAME_SAMPLES
        if n == 0:
            return False
        f32 = audio[: n * FRAME_SAMPLES].astype(np.float32) * INT16_SCALE
        frames = f32.reshape(n, FRAME_SAMPLES)
        rms = np.sqrt((frames * frames).mean(axis=1))
        loud_frames = int((rms > max(self.cfg.energy_floor, 0.004)).sum())
        return loud_frames * FRAME_MS >= MIN_SPEECH_MS

    async def _finalize_record(self) -> None:
        """Recording ended: transcribe it all (chunked if long), type once."""
        audio = (
            np.concatenate(self.rec_frames)
            if self.rec_frames
            else np.array([], dtype=np.int16)
        )
        self.rec_frames = []
        self._rec_finalize = False
        rec_seconds = len(audio) / SAMPLE_RATE
        self.rec_samples = 0

        if rec_seconds < 0.4:
            self._set_phase("ready")
            return

        # Energy gate: a recording with no speech-level audio must never
        # reach the ASR. On silence the model hallucinates — and with a
        # vocabulary in its context it parrots exactly those terms.
        if not self._recording_has_speech(audio):
            self._info("(no speech in the recording)")
            self._set_phase("ready")
            return

        self._status(f"✎ transcribing ({rec_seconds:.1f}s)...")
        self._set_phase("transcribing", f"{rec_seconds:.1f} s of audio")
        try:
            await self._transcribe_record(audio)
        finally:
            self._set_phase("ready")

    async def _transcribe_record(self, audio: np.ndarray) -> None:
        words: list[str] = []
        for chunk in self._split_audio(audio):
            text = await self._asr(chunk)
            if text:
                words.extend(self._clean_words(text))
        self.rec_silence_marks = []
        if not words:
            self._info("(no intelligible speech found)")
            return

        wants_enter = words[-1] == ENTER_TOKEN

        # Optional LLM cleanup: only on plain text (no structural tokens),
        # guarded so a hallucinating rewrite falls back to the raw words.
        if self.llm is not None and "\n" not in words and "\n\n" not in words:
            plain = " ".join(w for w in words if w != ENTER_TOKEN)
            if plain:
                self._status("✎ polishing...")
                self.status.detail = "polishing"
                try:
                    polished = await asyncio.to_thread(
                        polish_text, self.llm, self.llm_tokenizer, plain, self.vocab_terms
                    )
                    words = polished.split() + ([ENTER_TOKEN] if wants_enter else [])
                except Exception as exc:  # noqa: BLE001
                    self._err(f"polish: {exc!r}; using raw transcript")
        typed_text = render_typing(" ".join(words))
        printable = " ".join(w for w in words if w != ENTER_TOKEN)
        printable = printable.replace("\n", " ⏎ ")
        LOGGER.info("recording transcript: %s", printable)

        if self.cfg.pipe:
            sys.stdout.write(printable + "\n")
            sys.stdout.flush()
        else:
            await self.typist.type(typed_text)
            if self.cfg.send_enter or wants_enter:
                await self.typist.press_enter()
            else:
                await self.typist.type(" ")
            if not sys.stdout.isatty():
                sys.stdout.write(printable + "\n")
                sys.stdout.flush()
        self.total_words += len(words)

    async def _emit_partial(self) -> None:
        audio = np.concatenate(self.utt)
        text = await self._asr(audio)
        if not text:
            return
        words = self._clean_words(text)
        increment = self.committer.reading(words)
        if increment and not self.cfg.pipe:
            await self.typist.type(increment)
            self.total_words = self.committer.n_typed

    async def _finalize(self) -> None:
        audio = np.concatenate(self.utt) if self.utt else np.array([], dtype=np.int16)
        speech_ms = self.speech_frames * FRAME_MS
        had_commits = self.committer.n_typed > 0
        self.utt = []
        self.utt_samples = 0
        self.speech_frames = 0
        self.finalize_pending = False

        # Click/cough: drop silently when nothing has been committed.
        if speech_ms < MIN_SPEECH_MS and not had_commits:
            self.committer.reset()
            return

        text = await self._asr(audio)
        words = self._clean_words(text) if text else []

        tail = self.committer.flush(words)
        typed_any = self.committer.n_typed > 0
        wants_enter = bool(words) and words[-1] == ENTER_TOKEN
        printable = " ".join(w for w in words if w != ENTER_TOKEN)
        printable = printable.replace("\n", " ⏎ ")

        if typed_any:
            LOGGER.info("utterance: %s", printable)
            if self.cfg.pipe:
                sys.stdout.write(printable + "\n")
                sys.stdout.flush()
            else:
                if tail:
                    await self.typist.type(tail)
                if self.cfg.send_enter or wants_enter:
                    await self.typist.press_enter()
                else:
                    await self.typist.type(" ")
                if not sys.stdout.isatty():
                    sys.stdout.write(printable + "\n")
                    sys.stdout.flush()
            self.total_words += len(words)

        self.committer.reset()

    # -- setup / main loop ----------------------------------------------------

    async def _ensure_model(self, repo_id: str, patterns: list[str], label: str) -> None:
        """Download `repo_id` if it is not cached, reporting progress.

        Later loaders call snapshot_download themselves; after this it is a
        cache hit. Progress goes to self.status (menu bar) and the terminal
        status line.
        """
        from huggingface_hub import snapshot_download

        if Path(repo_id).exists():
            return
        try:
            await asyncio.to_thread(
                snapshot_download, repo_id, allow_patterns=patterns, local_files_only=True
            )
            return
        except Exception:  # noqa: BLE001 — not (fully) cached: download
            pass

        name = short_model_name(repo_id)
        self._info(f"Downloading {label} {name} (first run only)...")
        self.status.downloaded = self.status.total = 0
        self._set_phase("downloading", f"{label} · {name}")
        task = asyncio.ensure_future(asyncio.to_thread(
            snapshot_download, repo_id, allow_patterns=patterns,
            tqdm_class=progress_tqdm_class(self.status),
        ))
        while not task.done():
            st = self.status
            pct = f" {st.progress * 100:3.0f}%" if st.progress is not None else ""
            size = (f" {human_bytes(st.downloaded)} / {human_bytes(st.total)}"
                    if st.total else "")
            self._status(f"⬇ downloading {label}{pct}{size}")
            await asyncio.sleep(0.3)
        await task  # re-raise download errors
        # hf_xet serves some chunks from its local cache without a transfer
        # update, so the counter can stop short of the total; it is done.
        self.status.downloaded = self.status.total
        self._info(f"Downloaded {label} {name}.")

    async def _load_models(self) -> None:
        await self._ensure_model(self.cfg.model, ASR_MODEL_PATTERNS, "speech model")
        self._info(f"Loading model: {self.cfg.model}...")
        self._set_phase("loading", f"speech model · {short_model_name(self.cfg.model)}")
        from parlando import stt

        self._stt = stt
        self.model, self.tokenizer, self.feature_extractor = await asyncio.to_thread(
            stt.load_qwen3_asr, self.cfg.model
        )
        warm = (np.random.randn(SAMPLE_RATE) * 100).astype(np.int16)
        await self._asr(warm)

        if self.cfg.silero:
            try:
                self.silero = await asyncio.to_thread(SileroVAD)
                self._info("Silero VAD enabled (hybrid mode).")
            except Exception as exc:  # noqa: BLE001
                self._err(f"Silero unavailable ({exc!r}); using energy VAD.")

        if self.cfg.polish:
            try:
                await self._ensure_model(
                    self.cfg.polish_model, LLM_MODEL_PATTERNS, "polish model"
                )
                self._info(f"Loading polish LLM: {self.cfg.polish_model}...")
                self._set_phase(
                    "loading", f"polish model · {short_model_name(self.cfg.polish_model)}"
                )
                from mlx_lm.utils import load as load_llm

                self.llm, self.llm_tokenizer = await asyncio.to_thread(
                    load_llm, self.cfg.polish_model
                )
                self._info("Polish enabled (LLM cleanup at finalize).")
            except Exception as exc:  # noqa: BLE001
                self._err(f"Polish LLM unavailable ({exc!r}); polish disabled.")
                self.llm = None

    def _setup_hotkey(self):
        if not self.cfg.hotkey:
            return None
        try:
            from pynput import keyboard

            action = (
                "start/stop recording" if self.cfg.mode == "record"
                else "pause/resume"
            )
            name = self.cfg.hotkey.strip()
            if name in TAP_KEY_LABELS:
                target = getattr(keyboard.Key, name)
                tap = TapDetector()

                def on_press(key):
                    tap.press(key == target)

                def on_release(key):
                    if tap.release(key == target):
                        self.request_toggle()

                hk = keyboard.Listener(on_press=on_press, on_release=on_release)
                hk.start()
            else:
                hk = keyboard.GlobalHotKeys({name: self.request_toggle})
                hk.start()
            self._info(f"Hotkey ready: {hotkey_label(name)} ({action})")
            return hk
        except Exception as exc:  # noqa: BLE001
            self._err(f"Hotkey setup failed ({exc!r}); disable with --no-hotkey.")
            return None

    def _check_permissions(self) -> None:
        # Accessibility gates BOTH the global hotkey (pynput sees no key
        # events without it; it fails silently) and CGEvent typing. Ask
        # macOS to show its prompt so the app lands in the list directly.
        need_hotkey = bool(self.cfg.hotkey)
        if self.cfg.pipe and not need_hotkey:
            return
        trusted = accessibility_trusted(prompt=True)
        if trusted is False:
            self.accessibility_missing = True
            self._add_problem("accessibility")
            self._err(
                "Accessibility permission missing: the hotkey will NOT be "
                "heard and text CANNOT be typed. System Settings > Privacy & "
                "Security > Accessibility > enable the app that runs parlando "
                "(Parlando.app, or your terminal). No restart needed: parlando "
                "picks the permission up within a few seconds."
            )

    def _poll_accessibility(self) -> bool:
        """While the permission is missing, notice when it gets granted.

        A pynput listener created before the grant never receives events,
        so the hotkey is set up again. Returns True when recovered.
        """
        if not self.accessibility_missing or accessibility_trusted() is not True:
            return False
        self.accessibility_missing = False
        self._clear_problem("accessibility")
        if self._hotkey is not None:
            try:
                self._hotkey.stop()
            except Exception:  # noqa: BLE001
                pass
        self._hotkey = self._setup_hotkey()
        self._info("Accessibility permission granted; hotkey and typing active.")
        return True

    def _render_status(self) -> None:
        if self.cfg.mode == "record":
            key = hotkey_label(self.cfg.hotkey) if self.cfg.hotkey else "menu"
            if self.recording:
                secs = self.rec_samples / SAMPLE_RATE
                self._status(f"🔴 recording {secs:5.1f}s  ({key}: stop and type)")
            else:
                asr = f" | ASR {self.asr_ms:.0f}ms" if self.asr_ms else ""
                self._status(
                    f"○ ready  ({key}: start recording) | "
                    f"{self.total_words} words typed{asr}"
                )
            return
        if self.paused:
            self._status(f"⏸ paused ({hotkey_label(self.cfg.hotkey)})")
            return
        if self.vad.in_speech:
            secs = self.utt_samples / SAMPLE_RATE
            state = f"● speech {secs:4.1f}s"
        else:
            state = "○ listening"
        asr = f" | ASR {self.asr_ms:.0f}ms" if self.asr_ms else ""
        drop = f" | dropped {self.dropped}" if self.dropped else ""
        rst = f" | stream restarts {self._stream_restarts}" if self._stream_restarts else ""
        self._status(f"{state} | {self.total_words} words typed{asr}{drop}{rst}")

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        await self._load_models()
        if not self.cfg.pipe:
            self.typist = make_typist()
        self._check_permissions()
        self._hotkey = self._setup_hotkey()

        stream = self._open_stream()
        last_frame_time = self.loop.time()
        last_ax_check = last_frame_time

        stop_event = asyncio.Event()
        try:
            self.loop.add_signal_handler(signal.SIGINT, stop_event.set)
        except (NotImplementedError, RuntimeError, ValueError):
            # No signal handling on non-main threads (menu bar mode); the
            # shell owns the lifecycle there.
            pass

        if self.cfg.mode == "record":
            key = hotkey_label(self.cfg.hotkey) if self.cfg.hotkey else "the menu bar"
            self._info(
                f"Ready. Click the target window; use {key} to start "
                f"recording, speak, tap again to stop — the text is typed in "
                f"one go. (Ctrl+C to quit)"
            )
            if not self.cfg.hotkey:
                self._err(
                    "record mode needs a hotkey but --no-hotkey was given. "
                    "Use the menu bar app or switch to --mode stream."
                )
        else:
            self._info(
                "Ready. Click the target window and speak. (Ctrl+C to quit)"
            )
        self._set_phase("paused" if self.paused else "ready")
        last_mic_reopen = self.loop.time()

        try:
            while not stop_event.is_set():
                # Every iteration is guarded: one error cannot stop dictation.
                try:
                    try:
                        frame = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    except TimeoutError:
                        frame = None

                    now = self.loop.time()
                    if frame is not None:
                        last_frame_time = now
                        self._on_frame(frame)
                        while True:  # drain the backlog after inference
                            try:
                                self._on_frame(self.queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    elif now - last_frame_time > WATCHDOG_SECONDS:
                        # Sleep/device change: CoreAudio may have died silently.
                        self._stream_restarts += 1
                        self._err(
                            f"No audio for {WATCHDOG_SECONDS:.0f}s; "
                            "reopening the stream..."
                        )
                        stream = await self._reopen_stream(stream)
                        last_frame_time = self.loop.time()

                    if self._mic_warned and now - last_mic_reopen >= WATCHDOG_SECONDS:
                        # A stream opened before the mic permission was granted
                        # stays silent; a fresh one picks the grant up.
                        last_mic_reopen = now
                        stream = await self._reopen_stream(stream, quiet=True)

                    if self.accessibility_missing and now - last_ax_check >= AX_POLL_SECONDS:
                        last_ax_check = now
                        self._poll_accessibility()

                    if self._rec_finalize:
                        await self._finalize_record()
                    elif self.finalize_pending:
                        await self._finalize()
                    elif (
                        self.vad.in_speech
                        and not self.paused
                        and self.utt_samples >= int(MIN_PARTIAL_SECONDS * SAMPLE_RATE)
                        and self.loop.time() - self.last_partial >= self.cfg.interval
                    ):
                        self.last_partial = self.loop.time()
                        await self._emit_partial()

                    self._render_status()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    self._err("internal error (continuing):\n" + traceback.format_exc())
                    await asyncio.sleep(0.2)
        finally:
            if self._hotkey is not None:
                try:
                    self._hotkey.stop()
                except Exception:  # noqa: BLE001
                    pass
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._info("Stopped.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _setup_logging() -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
    except Exception:  # noqa: BLE001 — continue silently if logging can't be set up
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parlando",
        description="Local voice dictation: speak, and the text is typed "
                    "into the focused window. Without options this opens "
                    "the menu bar app (icon top right, settings in its menu); "
                    "with --terminal it dictates from this terminal window, "
                    "configured by the options below.",
    )
    p.add_argument("--version", action="version", version=f"parlando {__version__}")
    p.add_argument("--terminal", "-t", action="store_true",
                   help="Dictate from this terminal instead of the menu bar "
                        "(status here, Ctrl+C to quit). Implied by --pipe.")
    p.add_argument("--install-app", action="store_true",
                   help="Create ~/Applications/Parlando.app so macOS asks for "
                        "permissions in Parlando's name, then exit")
    p.add_argument("--uninstall-app", action="store_true",
                   help="Remove Parlando.app and its login item, then exit")
    p.add_argument("--install-login", action="store_true",
                   help="Start the menu bar app at login (LaunchAgent), then exit")
    p.add_argument("--uninstall-login", action="store_true",
                   help="Remove the login item, then exit")
    p.add_argument("--list-devices", action="store_true", help="List input devices")
    g = p.add_argument_group(
        "terminal mode options",
        "Only with --terminal (or --pipe); the menu bar app takes its "
        "settings from its menu.",
    )
    p.terminal_group = g
    g.add_argument("--mode", choices=["record", "stream"], default=Config.mode,
                   help="record: tap-speak-tap, typed in one go (default, most "
                        "robust). stream: always listening, word by word.")
    g.add_argument("--model", default=Config.model, help="ASR model")
    g.add_argument("--language", default=Config.language,
                   help=f"Language (default: {Config.language}; e.g. English, German)")
    g.add_argument("--device", type=int, default=None, help="Input device index")
    g.add_argument("--interval", type=float, default=Config.interval,
                   help="Partial ASR cadence in seconds, stream mode "
                        "(default: 0.5)")
    g.add_argument("--silence-ms", type=int, default=Config.silence_ms,
                   help="Silence that ends an utterance in stream mode, ms "
                        "(default: 500)")
    g.add_argument("--energy-floor", type=float, default=Config.energy_floor,
                   help="Minimum RMS threshold (default: 0.004)")
    g.add_argument("--max-utterance", type=float, default=Config.max_utterance_s,
                   help="Maximum utterance length in stream mode, seconds "
                        "(default: 20)")
    g.add_argument("--no-normalize", action="store_true",
                   help="Disable low-level microphone compensation")
    g.add_argument("--enter", action="store_true",
                   help="Press Return after each utterance")
    g.add_argument("--pipe", action="store_true",
                   help="Do not type; print utterances to stdout")
    g.add_argument("--hotkey", default=Config.hotkey,
                   help="Hotkey. A single key name (alt_r/cmd_r/ctrl_r/shift_r) "
                        "means single tap; combos like '<ctrl>+<alt>+d' also "
                        "work (default: alt_r = tap right Option)")
    g.add_argument("--no-hotkey", action="store_true", help="Disable the hotkey")
    g.add_argument("--no-commands", action="store_true",
                   help="Disable voice commands (period, new line, send)")
    g.add_argument("--vocab", default=Config.vocab,
                   help="Comma-separated terms to spell exactly (adds to "
                        "~/.config/parlando/vocabulary.txt)")
    g.add_argument("--no-cleanup", action="store_true",
                   help="Keep vocalized fillers (um, uh, eee) in the output")
    g.add_argument("--polish", action="store_true",
                   help="LLM cleanup at finalize: removes contextual fillers "
                        "and false starts, fixes punctuation (record mode)")
    g.add_argument("--polish-model", default=Config.polish_model,
                   help=f"Polish LLM (default: {Config.polish_model})")
    g.add_argument("--silero", action="store_true",
                   help="Hybrid Silero VAD (for noisy environments)")
    g.add_argument("--paused", action="store_true",
                   help="Start paused (resume with the hotkey)")
    return p


def terminal_options_given(parser: argparse.ArgumentParser,
                           args: argparse.Namespace) -> list[str]:
    """Terminal-mode options (except --pipe) that differ from their defaults."""
    given = []
    for action in parser.terminal_group._group_actions:
        if action.dest == "pipe":
            continue
        if getattr(args, action.dest, action.default) != action.default:
            given.append(action.option_strings[0])
    return given


def wants_terminal(args: argparse.Namespace) -> bool:
    return bool(args.terminal or args.pipe)


def list_devices() -> None:
    import sounddevice as sd

    default_in = sd.default.device[0]
    print("Input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <- default" if i == default_in else ""
            print(f"  [{i}] {d['name']}{mark}")


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        model=args.model,
        language=args.language,
        device=args.device,
        interval=args.interval,
        silence_ms=args.silence_ms,
        energy_floor=args.energy_floor,
        max_utterance_s=args.max_utterance,
        normalize=not args.no_normalize,
        send_enter=args.enter,
        pipe=args.pipe,
        hotkey=None if args.no_hotkey else args.hotkey,
        commands=not args.no_commands,
        cleanup=not args.no_cleanup,
        vocab=args.vocab,
        polish=args.polish,
        polish_model=args.polish_model,
        silero=args.silero,
        start_paused=args.paused,
        mode=args.mode,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.list_devices:
        list_devices()
        return 0

    from parlando import menubar  # lazy: the engine stays importable alone

    if args.install_app:
        return menubar.install_app()
    if args.uninstall_app:
        return menubar.uninstall_app()
    if args.install_login:
        return menubar.install_login()
    if args.uninstall_login:
        return menubar.uninstall_login()
    if not wants_terminal(args):
        given = terminal_options_given(parser, args)
        if given:
            parser.error(
                f"{', '.join(given)}: terminal option(s) given without "
                "--terminal. The menu bar app takes its settings from its "
                "menu; add --terminal (-t) to dictate from this window with "
                "these options."
            )
        return menubar.run_menubar()

    _setup_logging()
    LOGGER.info("parlando %s starting: %s", __version__, vars(args))
    try:
        asyncio.run(DictationEngine(config_from_args(args)).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
