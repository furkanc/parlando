"""Regressions for Whisper routing, long recordings and clipboard ownership."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from parlando import engine
from parlando.clipboard import TemporaryClipboard


@pytest.mark.parametrize("language, expected", [("Turkish", "turkish"), ("Auto", None)])
def test_whisper_receives_audio_language_and_transcription_task(language, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "VOCAB_FILE", tmp_path / "vocabulary.txt")
    calls = []
    eng = engine.DictationEngine(engine.Config(language=language, vocab="BigQuery", normalize=False))
    eng.model = "/cached/whisper"
    eng._whisper = SimpleNamespace(transcribe=lambda audio, **kw: (
        calls.append((audio, kw)) or {"text": " BigQuery deployment "}
    ))
    assert eng._transcribe_blocking(np.array([16384], dtype=np.int16)) == "BigQuery deployment"
    assert calls[0][0].tolist() == [0.5]
    assert calls[0][1] == {
        "path_or_hf_repo": "/cached/whisper", "language": expected,
        "task": "transcribe", "temperature": 0.0, "verbose": None,
        "initial_prompt": "BigQuery",
    }


def test_qwen_still_uses_its_own_engine():
    from contextlib import nullcontext

    eng = engine.DictationEngine(engine.Config(backend="qwen"))
    eng._stt = SimpleNamespace(_suppress_output=nullcontext,
                               transcribe=lambda *a, **kw: iter(["Qwen ", "works"]))
    assert eng._transcribe_blocking(np.array([1], dtype=np.int16)) == "Qwen works"


def test_qwen_auto_removes_detected_language_header():
    from contextlib import nullcontext

    eng = engine.DictationEngine(engine.Config(backend="qwen", language="Auto"))
    eng._stt = SimpleNamespace(
        _suppress_output=nullcontext,
        transcribe=lambda *a, **kw: iter(["language English<asr", "_text>Hello world"]),
    )
    assert eng._transcribe_blocking(np.array([1], dtype=np.int16)) == "Hello world"


def test_hotkey_change_restores_previous_listener_on_failure():
    eng = engine.DictationEngine(engine.Config())
    eng.status.phase = "ready"
    events = []
    eng._hotkey = SimpleNamespace(stop=lambda: events.append("stopped"))
    restored = object()
    results = iter([None, restored])
    eng._setup_hotkey = lambda: next(results)
    eng._replace_hotkey("cmd_r")
    assert events == ["stopped"]
    assert eng.cfg.hotkey == "alt_r" and eng._hotkey is restored


def test_hotkey_change_does_not_interrupt_recording():
    eng = engine.DictationEngine(engine.Config())
    eng.status.phase = "recording"
    eng._hotkey = SimpleNamespace(stop=lambda: pytest.fail("listener stopped during recording"))
    eng._replace_hotkey("cmd_r")
    assert eng.cfg.hotkey == "alt_r"


def test_cli_selects_qwen_default_and_can_disable_clipboard():
    args = engine.build_arg_parser().parse_args(["-t", "--backend", "qwen", "--no-clipboard"])
    cfg = engine.config_from_args(args)
    assert cfg.backend == "qwen" and "Qwen3-ASR" in cfg.model
    assert not cfg.copy_clipboard


def test_recording_continues_past_two_minutes_then_stops_at_ten(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: None)
    eng = engine.DictationEngine(engine.Config())
    eng.toggle_record()
    frame = np.ones(engine.FRAME_SAMPLES, dtype=np.int16) * 1000
    eng.rec_samples = 120 * engine.SAMPLE_RATE
    eng._on_frame_record(frame)
    assert eng.recording and not eng._rec_finalize
    eng.rec_samples = 600 * engine.SAMPLE_RATE - len(frame)
    eng._on_frame_record(frame)
    assert not eng.recording and eng._rec_finalize


def test_ten_minute_recording_chunks_keep_every_sample():
    eng = engine.DictationEngine(engine.Config())
    audio = np.zeros(600 * engine.SAMPLE_RATE, dtype=np.int16)
    chunks = eng._split_audio(audio)
    assert sum(len(chunk) for chunk in chunks) == len(audio)
    assert all(len(chunk) <= engine.CHUNK_SECONDS * engine.SAMPLE_RATE for chunk in chunks)


def test_final_text_copied_after_typing_with_real_newlines():
    eng = engine.DictationEngine(engine.Config(language="English"))
    events = []

    async def asr(_audio):
        return "first new line second send"

    async def type_text(text):
        events.append(("type", text))

    async def enter():
        events.append(("enter", None))

    eng._asr = asr
    eng.typist = SimpleNamespace(type=type_text, press_enter=enter)
    eng.clipboard = SimpleNamespace(copy=lambda text: events.append(("copy", text)))
    asyncio.run(eng._transcribe_record(np.zeros(16000, dtype=np.int16)))
    assert events == [("type", "first\nsecond"), ("enter", None), ("copy", "first\nsecond")]


def test_silent_chunks_in_long_recording_do_not_reach_asr():
    eng = engine.DictationEngine(engine.Config(pipe=True))
    seen = []

    async def asr(audio):
        seen.append(len(audio))
        return "speech"

    eng._asr = asr
    speech = np.ones(28 * engine.SAMPLE_RATE, dtype=np.int16) * 1000
    silence = np.zeros(28 * engine.SAMPLE_RATE, dtype=np.int16)
    asyncio.run(eng._transcribe_record(np.concatenate([speech, silence])))
    assert seen == [len(speech)]


class PasteboardItem:
    def __init__(self, data=None):
        self.data = data or {}

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def types(self):
        return list(self.data)

    def dataForType_(self, kind):
        return self.data[kind]

    def setData_forType_(self, data, kind):
        self.data[kind] = data


class Pasteboard:
    def __init__(self):
        self.items = [PasteboardItem({"text": b"original", "image": b"image data"})]
        self.count = 1

    def pasteboardItems(self):
        return self.items

    def clearContents(self):
        self.items = []
        self.count += 1

    def setString_forType_(self, text, kind):
        self.items = [PasteboardItem({kind: text.encode()})]
        self.count += 1
        return True

    def writeObjects_(self, items):
        self.items = items
        self.count += 1
        return True

    def changeCount(self):
        return self.count

    def stringForType_(self, kind):
        return self.items[0].data.get(kind, b"").decode() if self.items else None


@pytest.fixture
def clipboard(monkeypatch):
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(
        NSPasteboard=None, NSPasteboardItem=PasteboardItem, NSPasteboardTypeString="text"))
    board = Pasteboard()
    value = TemporaryClipboard(duration=3600, pasteboard=board)
    yield value, board
    value.restore()


def test_clipboard_restores_all_previous_representations(clipboard):
    value, board = clipboard
    value.copy("dictation")
    assert board.items[0].data == {"text": b"dictation"}
    value.restore()
    assert board.items[0].data == {"text": b"original", "image": b"image data"}


def test_engine_uses_custom_clipboard_duration(clipboard):
    value, board = clipboard
    eng = engine.DictationEngine(engine.Config(clipboard_seconds=5))
    eng.clipboard = value
    eng._copy_output("dictation")
    assert board.items[0].data == {"text": b"dictation"}
    assert value._timer.interval == 5


def test_clipboard_preserves_user_copy(clipboard):
    value, board = clipboard
    value.copy("dictation")
    board.setString_forType_("user copy", "text")
    value.restore()
    assert board.items[0].data == {"text": b"user copy"}


def test_clipboard_preserves_content_change_without_new_owner(clipboard):
    value, board = clipboard
    value.copy("dictation")
    board.items[0].data["text"] = b"in-place update"
    value.restore()
    assert board.items[0].data == {"text": b"in-place update"}


def test_late_timer_cannot_erase_next_dictation(clipboard):
    value, board = clipboard
    value.copy("first")
    first_count = board.changeCount()
    value.copy("second")
    value.restore(first_count)
    assert board.items[0].data == {"text": b"second"}
    value.restore()
    assert board.items[0].data["text"] == b"original"
