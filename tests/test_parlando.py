"""parlando core unit tests.

No model, microphone or permissions needed: heavy dependencies are imported
lazily inside parlando.py, so numpy + pytest are enough (this is how CI runs).
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from parlando import engine as vt  # noqa: E402

E = vt.ENTER_TOKEN


# -----------------------------------------------------------------------------
# Guards
# -----------------------------------------------------------------------------


def test_collapse_repeats():
    # 3+ runs are stutters/hallucinations -> one occurrence
    assert vt.collapse_repeats("a a a a a b".split()) == ["a", "b"]
    assert vt.collapse_repeats("I I I want to to go".split()) == \
        "I want to to go".split()  # doubles stay (may be legitimate)
    assert vt.collapse_repeats("çok çok güzel".split()) == "çok çok güzel".split()
    assert vt.collapse_repeats([]) == []
    assert vt.collapse_repeats("x y z".split()) == ["x", "y", "z"]


def test_looks_hallucinated():
    assert vt.looks_hallucinated(["okay"] * 12)
    assert not vt.looks_hallucinated("today the weather is really quite nice".split())
    assert not vt.looks_hallucinated(["short", "text"])


# -----------------------------------------------------------------------------
# Voice commands
# -----------------------------------------------------------------------------


def test_voice_commands_english_punctuation():
    f = vt.apply_voice_commands
    assert f("hello period".split()) == ["hello."]
    assert f("really question mark".split()) == ["really?"]
    assert f("yes comma no".split()) == ["yes,", "no"]


def test_voice_commands_english_newline_and_enter():
    f = vt.apply_voice_commands
    assert f("first line new line second".split()) == ["first", "line", "\n", "second"]
    assert f("do this send".split()) == ["do", "this", E]


def test_voice_commands_turkish():
    f = vt.apply_voice_commands
    assert f("merhaba nokta".split(), "Turkish") == ["merhaba."]
    assert f("tamam mı soru işareti".split(), "Turkish") == ["tamam", "mı?"]
    assert f("bunu yap gönder".split(), "Turkish") == ["bunu", "yap", E]


def test_voice_commands_unknown_language_falls_back_to_english():
    assert vt.apply_voice_commands("hello period".split(), "German") == ["hello."]


def test_voice_commands_orphan_punct_dropped():
    assert vt.apply_voice_commands(["comma"]) == []


def test_render_typing():
    assert vt.render_typing("first \n second") == "first\nsecond"
    assert vt.render_typing("do " + E) == "do"


# -----------------------------------------------------------------------------
# WordCommitter: never deletes, never blocks, never repeats
# -----------------------------------------------------------------------------


def test_committer_monotonic_growth():
    c = vt.WordCommitter()
    out = []
    for p in ["hello", "hello there", "hello there friend"]:
        out.append(c.reading(p.split()))
    out.append(c.flush("hello there friend today".split()))
    assert "".join(out) == "hello there friend today"


def test_committer_no_block_on_committed_drift():
    """A revised leading word must not block the flow (regression)."""
    c = vt.WordCommitter()
    out = [c.reading(p.split()) for p in [
        "hello", "hello there", "Hello there friend",
        "hello there friend today",
    ]]
    out.append(c.flush("hello there friend today ok".split()))
    assert "".join(out) == "hello there friend today ok"


def test_committer_agreement_ignores_punct_and_case():
    """Punctuation/case flapping must not stall the stream (regression)."""
    c = vt.WordCommitter()
    out = [c.reading(r.split()) for r in [
        "today the",
        "Today, the weather",
        "today the weather is",
        "Today, the weather is quite",
        "today the weather is quite nice",
    ]]
    assert c.n_typed >= 4, f"stream stalled: {c.n_typed} words, out={out}"


def test_committer_stall_safety_valve():
    """Even with zero agreement, the flow continues after 3 readings."""
    c = vt.WordCommitter()
    # The first word is completely different each reading: agreement impossible.
    c.reading("aa one two three".split())
    c.reading("bb one two three four".split())
    out = c.reading("cc one two three four five".split())  # 3rd stalled reading
    assert out, "safety valve did not fire"
    assert c.n_typed >= 4  # everything but the last word was typed


def test_committer_never_retracts():
    c = vt.WordCommitter()
    c.reading("a b c".split())
    c.reading("a b c d".split())  # 'a b c' committed
    assert c.reading(["x"]) == ""  # total divergence: no delete, no append
    assert c.n_typed == 3


# -----------------------------------------------------------------------------
# EnergyVAD
# -----------------------------------------------------------------------------


def test_vad_start_end_cycle():
    v = vt.EnergyVAD(floor=0.004, silence_ms=300)
    events = []
    for _ in range(20):
        events.append(v.update(0.001))
    for _ in range(3):
        events.append(v.update(0.05))
    for _ in range(30):
        events.append(v.update(0.05))
    for _ in range(12):
        events.append(v.update(0.001))
    assert "start" in events and "end" in events
    assert events.index("start") < events.index("end")
    assert events[:20] == ["idle"] * 20


def test_vad_debounce_rejects_single_click():
    v = vt.EnergyVAD(0.004, 300)
    for _ in range(10):
        v.update(0.001)
    assert v.update(0.08) == "idle"


def test_vad_silero_prob_overrides_energy():
    v = vt.EnergyVAD(0.004, 300)
    assert v.update(0.001, prob=0.9) == "idle"
    assert v.update(0.001, prob=0.9) == "start"
    assert v.update(0.001, prob=0.9) == "speech"
    for _ in range(9):
        v.update(0.001, prob=0.1)
    assert v.update(0.001, prob=0.1) == "end"


def test_vad_noise_floor_does_not_go_deaf():
    """Speech scraps must not poison the noise floor (regression)."""
    v = vt.EnergyVAD(floor=0.004, silence_ms=300)
    for _ in range(500):
        v.update(0.012)
        if v.in_speech:
            for _ in range(11):
                v.update(0.0005)
    assert v.noise <= v.floor * 1.5 + 1e-9
    v.in_speech = False
    v.update(0.05)
    assert v.update(0.05) == "start"


# -----------------------------------------------------------------------------
# Engine flows (fake ASR/typist)
# -----------------------------------------------------------------------------


class FakeTypist:
    def __init__(self):
        self.out = []

    async def type(self, t):
        self.out.append(t)

    async def press_enter(self):
        self.out.append("<ENTER>")


def _speech_engine(cfg=None, asr_text="hello world"):
    eng = vt.DictationEngine(cfg or vt.Config(mode="stream"))
    eng.typist = FakeTypist()

    async def fake_asr(_a):
        return asr_text

    eng._asr = fake_asr
    eng.utt = [np.ones(480, np.int16) * 1000] * 40
    eng.utt_samples = 480 * 40
    eng.speech_frames = 40
    return eng


def test_finalize_types_and_separates():
    eng = _speech_engine(asr_text="hello world how are you")
    asyncio.new_event_loop().run_until_complete(eng._finalize())
    assert "".join(eng.typist.out) == "hello world how are you "


def test_finalize_send_presses_enter():
    eng = _speech_engine(vt.Config(mode="stream", language="English"), "do this send")
    asyncio.new_event_loop().run_until_complete(eng._finalize())
    assert eng.typist.out == ["do this", "<ENTER>"]


def test_finalize_discards_click():
    eng = _speech_engine()

    async def boom(_a):
        raise AssertionError("clicks must not reach ASR")

    eng._asr = boom
    eng.utt = [np.ones(480, np.int16)] * 3
    eng.utt_samples = 480 * 3
    eng.speech_frames = 3  # 90ms < MIN_SPEECH_MS
    asyncio.new_event_loop().run_until_complete(eng._finalize())
    assert eng.typist.out == []


def test_partial_plus_finalize_stream():
    eng = vt.DictationEngine(vt.Config(mode="stream"))
    eng.typist = FakeTypist()
    readings = iter([
        "hello", "hello there", "hello there friend",
        "hello there friend today",
    ])

    async def fake_asr(_a):
        return next(readings)

    eng._asr = fake_asr
    eng.utt = [np.ones(480, np.int16) * 1000] * 40
    eng.utt_samples = 480 * 40
    eng.speech_frames = 40
    loop = asyncio.new_event_loop()
    for _ in range(3):
        loop.run_until_complete(eng._emit_partial())
    loop.run_until_complete(eng._finalize())
    assert "".join(eng.typist.out) == "hello there friend today "


def test_asr_error_does_not_raise():
    eng = vt.DictationEngine(vt.Config())
    eng._stt = None  # _transcribe_blocking will blow up

    async def run():
        return await eng._asr(np.ones(16000, np.int16))

    assert asyncio.new_event_loop().run_until_complete(run()) == ""


def test_paused_ignores_frames():
    eng = vt.DictationEngine(vt.Config(mode="stream", start_paused=True))
    eng._on_frame(np.ones(480, np.int16) * 5000)
    assert eng.utt == [] and not eng.vad.in_speech


def test_toggle_pause_clears_utterance(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="stream"))
    eng.utt = [np.ones(480, np.int16)]
    eng.utt_samples = 480
    eng.vad.in_speech = True
    eng.toggle_pause()
    assert eng.paused and eng.utt == [] and not eng.vad.in_speech


def test_hallucinated_reading_commits_nothing():
    eng = vt.DictationEngine(vt.Config(mode="stream"))
    eng.typist = FakeTypist()

    async def fake_asr(_a):
        return "okay " * 12

    eng._asr = fake_asr
    eng.utt = [np.ones(480, np.int16) * 1000] * 40
    eng.utt_samples = 480 * 40
    eng.speech_frames = 40
    loop = asyncio.new_event_loop()
    loop.run_until_complete(eng._emit_partial())
    loop.run_until_complete(eng._emit_partial())
    assert eng.typist.out == []


# -----------------------------------------------------------------------------
# Record mode (tap-speak-tap)
# -----------------------------------------------------------------------------


def test_record_toggle_and_finalize(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record"))
    eng.typist = FakeTypist()

    async def fake_asr(_a):
        return "hello this is a recording test"

    eng._asr = fake_asr
    eng.toggle_action()  # recording started
    assert eng.recording
    for _ in range(50):
        eng._on_frame(np.ones(480, np.int16) * 1000)
    assert eng.rec_samples == 50 * 480
    eng.toggle_action()  # recording stopped
    assert not eng.recording and eng._rec_finalize
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert "".join(eng.typist.out) == "hello this is a recording test "


def test_record_short_recording_discarded(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record"))
    eng.typist = FakeTypist()

    async def boom(_a):
        raise AssertionError("short recordings must not reach ASR")

    eng._asr = boom
    eng.toggle_action()
    for _ in range(5):  # ~150ms
        eng._on_frame(np.ones(480, np.int16) * 1000)
    eng.toggle_action()
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert eng.typist.out == []


def test_split_audio_cuts_at_silence_marks():
    eng = vt.DictationEngine(vt.Config(mode="record"))
    sr = vt.SAMPLE_RATE
    audio = np.zeros(60 * sr, dtype=np.int16)  # 60s > 28s window
    eng.rec_silence_marks = [25 * sr, 50 * sr]
    chunks = eng._split_audio(audio)
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [25 * sr, 25 * sr, 10 * sr]
    assert sum(len(c) for c in chunks) == len(audio)


def test_split_audio_short_passthrough():
    eng = vt.DictationEngine(vt.Config(mode="record"))
    audio = np.zeros(10 * vt.SAMPLE_RATE, dtype=np.int16)
    assert len(eng._split_audio(audio)) == 1


def test_record_send_presses_enter(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record", language="English"))
    eng.typist = FakeTypist()

    async def fake_asr(_a):
        return "write this to claude send"

    eng._asr = fake_asr
    eng.toggle_action()
    for _ in range(50):
        eng._on_frame(np.ones(480, np.int16) * 1000)
    eng.toggle_action()
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert eng.typist.out == ["write this to claude", "<ENTER>"]


def test_default_language_is_turkish(monkeypatch):
    """Default config dictates Turkish: Turkish fillers and commands apply."""
    assert vt.Config().language == "Turkish"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record"))
    eng.typist = FakeTypist()

    async def fake_asr(_a):
        return "eee bunu yaz gönder"

    eng._asr = fake_asr
    eng.toggle_action()
    for _ in range(50):
        eng._on_frame(np.ones(480, np.int16) * 1000)
    eng.toggle_action()
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert eng.typist.out == ["bunu yaz", "<ENTER>"]


# -----------------------------------------------------------------------------
# TapDetector: single-modifier tap
# -----------------------------------------------------------------------------


def test_tap_simple_tap_triggers():
    t = vt.TapDetector(timeout=0.6)
    t.press(True)
    assert t.release(True) is True


def test_tap_chord_does_not_trigger():
    """Character chords such as Option+Q must not toggle dictation."""
    t = vt.TapDetector(timeout=0.6)
    t.press(True)      # right Option down
    t.press(False)     # another key pressed (chord)
    assert t.release(True) is False


def test_tap_long_hold_does_not_trigger():
    t = vt.TapDetector(timeout=0.6)
    t.press(True)
    t.t -= 1.0  # backdate the press by 1s (long hold)
    assert t.release(True) is False


def test_tap_other_key_release_ignored():
    t = vt.TapDetector(timeout=0.6)
    t.press(True)
    assert t.release(False) is False  # releasing another key never triggers
    assert t.release(True) is True    # releasing the target does


def test_hotkey_label():
    assert "Option" in vt.hotkey_label("alt_r")
    assert vt.hotkey_label("<ctrl>+<alt>+d") == "<ctrl>+<alt>+d"
    assert vt.hotkey_label(None) == "no hotkey"


# -----------------------------------------------------------------------------
# Cleanup (lexicon fillers) and polish guardrail
# -----------------------------------------------------------------------------


def test_remove_fillers_english():
    words = "um so uh we should umm ship it".split()
    assert vt.remove_fillers(words, "English") == "so we should ship it".split()


def test_remove_fillers_turkish():
    words = "eee toplantıyı ııı yarına alalım".split()
    assert vt.remove_fillers(words, "Turkish") == "toplantıyı yarına alalım".split()


def test_remove_fillers_keeps_contextual_words():
    # Contextual fillers are the LLM's job; the lexicon must not touch them.
    words = "you know I like the design yani şey güzel".split()
    assert vt.remove_fillers(words, "English") == words
    assert vt.remove_fillers(words, "Turkish") == words


def test_polish_guard_accepts_edit():
    raw = "um so I think we should ship the release this week"
    polished = "So I think we should ship the release this week."
    assert vt.polish_guard(raw, polished)


def test_polish_guard_rejects_answer():
    raw = "write down what is the capital of France"
    answered = "The capital of France is Paris."
    assert not vt.polish_guard(raw, answered)


def test_polish_guard_rejects_expansion():
    raw = "send the report"
    essay = ("I will gladly send the report tomorrow morning after I have "
             "reviewed all of the numbers and formatted everything nicely")
    assert not vt.polish_guard(raw, essay)


def test_polish_guard_rejects_empty():
    assert not vt.polish_guard("hello world", "")


def test_clean_words_pipeline_removes_fillers():
    eng = vt.DictationEngine(vt.Config(language="Turkish"))
    words = eng._clean_words("eee toplantıyı ııı yarına alalım nokta")
    assert words == ["toplantıyı", "yarına", "alalım."]


def test_clean_words_cleanup_can_be_disabled():
    eng = vt.DictationEngine(vt.Config(language="English", cleanup=False))
    assert "um" in eng._clean_words("um ship it")


def test_polish_disabled_by_default():
    eng = vt.DictationEngine(vt.Config())
    assert eng.cfg.polish is False and eng.llm is None


# -----------------------------------------------------------------------------
# CLI entry: plain `parlando` is the menu bar app, --terminal is the CLI
# -----------------------------------------------------------------------------


def test_cli_default_opens_menubar(monkeypatch):
    from parlando import menubar

    calls = []
    monkeypatch.setattr(menubar, "run_menubar", lambda: calls.append("menubar") or 0)
    monkeypatch.setattr(sys, "argv", ["parlando"])
    assert vt.main() == 0 and calls == ["menubar"]


def test_cli_login_flags_dispatch(monkeypatch):
    from parlando import menubar

    calls = []
    monkeypatch.setattr(menubar, "install_login", lambda: calls.append("install") or 0)
    monkeypatch.setattr(menubar, "uninstall_login", lambda: calls.append("uninstall") or 0)
    monkeypatch.setattr(sys, "argv", ["parlando", "--install-login"])
    assert vt.main() == 0
    monkeypatch.setattr(sys, "argv", ["parlando", "--uninstall-login"])
    assert vt.main() == 0
    assert calls == ["install", "uninstall"]


def test_cli_terminal_options_require_terminal_flag(monkeypatch):
    from parlando import menubar

    monkeypatch.setattr(menubar, "run_menubar", lambda: pytest.fail("must not open"))
    monkeypatch.setattr(sys, "argv", ["parlando", "--language", "Turkish", "--polish"])
    with pytest.raises(SystemExit) as exc:
        vt.main()
    assert exc.value.code == 2


def test_terminal_options_given_and_wants_terminal():
    parser = vt.build_arg_parser()
    args = parser.parse_args([])
    assert vt.terminal_options_given(parser, args) == [] and not vt.wants_terminal(args)
    args = parser.parse_args(["--pipe"])
    assert vt.terminal_options_given(parser, args) == [] and vt.wants_terminal(args)
    args = parser.parse_args(["-t", "--language", "English", "--polish", "--hotkey", "alt_r"])
    assert vt.wants_terminal(args)
    assert vt.terminal_options_given(parser, args) == ["--language", "--polish"]
