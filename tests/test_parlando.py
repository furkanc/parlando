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


# -----------------------------------------------------------------------------
# Parlando.app bundle (menu bar shell)
# -----------------------------------------------------------------------------


def test_app_bundle_layout(tmp_path):
    import plistlib

    from parlando import menubar

    dest = menubar.build_app_bundle(
        tmp_path / "Parlando.app", python="/opt/py/bin/python3", sign=False
    )
    contents = dest / "Contents"
    with open(contents / "Info.plist", "rb") as fh:
        info = plistlib.load(fh)
    assert info["CFBundleIdentifier"] == menubar.BUNDLE_ID
    assert info["CFBundleExecutable"] == "parlando"
    assert info["LSUIElement"] is True
    assert "NSMicrophoneUsageDescription" in info
    assert (contents / "Resources" / (info["CFBundleIconFile"] + ".icns")).exists()

    exe = contents / "MacOS" / "parlando"
    assert exe.stat().st_mode & 0o111, "main executable must be executable"
    # A real Mach-O, not a script: macOS refuses script main executables.
    assert exe.read_bytes()[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")

    script = contents / "Resources" / "launch.sh"
    assert script.stat().st_mode & 0o111
    text = script.read_text()
    assert text.startswith("#!/bin/sh")
    assert "'/opt/py/bin/python3' -m parlando.menubar" in text


def test_app_bundle_launcher_quotes_python_path(tmp_path):
    from parlando import menubar

    dest = menubar.build_app_bundle(
        tmp_path / "P.app", python="/Users/o'brien/py/bin/python", sign=False
    )
    text = (dest / "Contents" / "Resources" / "launch.sh").read_text()
    assert "'/Users/o'\\''brien/py/bin/python'" in text


def test_app_bundle_is_stable_across_rebuilds(tmp_path):
    """Same inputs -> identical files, so re-running --install-app never
    changes the bundle hash the privacy database keys permissions on."""
    from parlando import menubar

    a = menubar.build_app_bundle(tmp_path / "A.app", python="/x/python", sign=False)
    b = menubar.build_app_bundle(tmp_path / "B.app", python="/x/python", sign=False)
    for rel in ("Contents/Info.plist", "Contents/Resources/launch.sh",
                "Contents/MacOS/parlando"):
        assert (a / rel).read_bytes() == (b / rel).read_bytes()


@pytest.mark.skipif(sys.platform != "darwin", reason="codesign is macOS-only")
def test_app_bundle_adhoc_signature_verifies(tmp_path):
    from parlando import menubar

    dest = menubar.build_app_bundle(tmp_path / "S.app", python="/x/python", sign=True)
    r = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(dest)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_cli_app_flags_dispatch(monkeypatch):
    from parlando import menubar

    calls = []
    monkeypatch.setattr(menubar, "install_app", lambda: calls.append("install") or 0)
    monkeypatch.setattr(menubar, "uninstall_app", lambda: calls.append("uninstall") or 0)
    monkeypatch.setattr(sys, "argv", ["parlando", "--install-app"])
    assert vt.main() == 0
    monkeypatch.setattr(sys, "argv", ["parlando", "--uninstall-app"])
    assert vt.main() == 0
    assert calls == ["install", "uninstall"]


# -----------------------------------------------------------------------------
# Accessibility auto-recovery: no restart after granting the permission
# -----------------------------------------------------------------------------


class _FakeListener:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_poll_accessibility_recovers_and_rebuilds_hotkey(monkeypatch):
    eng = vt.DictationEngine(vt.Config())
    old = _FakeListener()
    eng._hotkey = old
    eng.accessibility_missing = True
    built = []
    monkeypatch.setattr(eng, "_setup_hotkey", lambda: built.append(1) or _FakeListener())

    monkeypatch.setattr(vt, "accessibility_trusted", lambda prompt=False: False)
    assert eng._poll_accessibility() is False
    assert eng.accessibility_missing and built == [] and not old.stopped

    monkeypatch.setattr(vt, "accessibility_trusted", lambda prompt=False: True)
    assert eng._poll_accessibility() is True
    assert not eng.accessibility_missing
    assert old.stopped, "the dead pre-grant listener must be stopped"
    assert built == [1] and eng._hotkey is not old
    # Already recovered: nothing more happens.
    assert eng._poll_accessibility() is False and built == [1]


def test_poll_accessibility_noop_when_not_missing(monkeypatch):
    eng = vt.DictationEngine(vt.Config())
    monkeypatch.setattr(vt, "accessibility_trusted", lambda prompt=False: pytest.fail("no check"))
    assert eng._poll_accessibility() is False


def test_app_bundle_embedded_runtime_uses_relative_python(tmp_path):
    """scripts/build_app.sh passes a Resources-relative interpreter path."""
    from parlando import menubar

    dest = menubar.build_app_bundle(
        tmp_path / "E.app", python="runtime/bin/python3", sign=False, version="9.9.9"
    )
    text = (dest / "Contents" / "Resources" / "launch.sh").read_text()
    assert 'RES=$(cd "$(dirname "$0")" && pwd)' in text
    assert '"$RES"/\'runtime/bin/python3\' -m parlando.menubar' in text
    import plistlib
    with open(dest / "Contents" / "Info.plist", "rb") as fh:
        assert plistlib.load(fh)["CFBundleShortVersionString"] == "9.9.9"


# -----------------------------------------------------------------------------
# Status: engine phases, download progress, menu bar presentation
# -----------------------------------------------------------------------------


def test_status_progress_and_human_bytes():
    st = vt.Status()
    assert st.progress is None
    st.phase, st.total, st.downloaded = "downloading", 2000, 500
    assert st.progress == 0.25
    assert vt.human_bytes(2_400_000_000) == "2.2 GB"
    assert vt.human_bytes(50 * 1024 * 1024) == "50 MB"
    assert vt.short_model_name("mlx-community/Qwen3-ASR-1.7B-8bit") == "Qwen3-ASR-1.7B-8bit"


def test_progress_tqdm_aggregates_byte_bars_only():
    pytest.importorskip("huggingface_hub")
    st = vt.Status(phase="downloading")
    T = vt.progress_tqdm_class(st)
    files = T(total=3, unit="it", disable=True)      # snapshot-level bar: ignored
    a = T(total=1000, unit="B", disable=True)
    b = T(total=500, unit="B", initial=100, disable=True)
    r = T(total=0, unit="B", desc="Reconstructing (incomplete total...)", disable=True)
    assert st.total == 1500 and st.downloaded == 100
    a.update(400)
    b.update(50)
    files.update(1)
    r.update(450)  # r: the same bytes again; must not count
    assert st.downloaded == 550
    assert abs(st.progress - 550 / 1500) < 1e-9
    # hf_xet sets the total after construction.
    c = T(total=0, unit="B", desc="Downloading bytes", disable=True)
    c.total = 2000
    c.update(1000)
    assert st.total == 3500 and st.downloaded == 1550


def test_record_flow_phases(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record"))
    eng.typist = FakeTypist()
    phases = []
    eng.on_state = phases.append

    async def fake_asr(_a):
        return "hello"

    eng._asr = fake_asr
    eng.toggle_action()
    assert eng.status.phase == "recording"
    for _ in range(50):
        eng._on_frame(np.ones(480, np.int16) * 1000)
    eng.toggle_action()
    assert eng.status.phase == "transcribing"
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert eng.status.phase == "ready"
    assert phases == ["recording", "transcribing", "transcribing", "ready"]


def test_short_recording_returns_to_ready(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    eng = vt.DictationEngine(vt.Config(mode="record"))
    eng.toggle_action()
    eng.toggle_action()
    asyncio.new_event_loop().run_until_complete(eng._finalize_record())
    assert eng.status.phase == "ready"


def test_mic_silence_problem_is_recoverable():
    eng = vt.DictationEngine(vt.Config())
    zeros = np.zeros(480, np.int16)
    for _ in range(vt.MIC_ZERO_FRAMES):
        eng._check_mic_silence(zeros)
    assert eng.status.problems == ["microphone"] and eng._mic_warned
    eng._check_mic_silence(np.ones(480, np.int16) * 500)
    assert eng.status.problems == [] and not eng._mic_warned
    # Later silence (a quiet room, a muted mic) must not re-trigger it.
    for _ in range(vt.MIC_ZERO_FRAMES + 5):
        eng._check_mic_silence(zeros)
    assert eng.status.problems == []


def test_accessibility_problem_tracks_recovery(monkeypatch):
    eng = vt.DictationEngine(vt.Config())
    monkeypatch.setattr(vt, "accessibility_trusted", lambda prompt=False: False)
    eng._check_permissions()
    assert eng.status.problems == ["accessibility"]
    monkeypatch.setattr(eng, "_setup_hotkey", lambda: None)
    monkeypatch.setattr(vt, "accessibility_trusted", lambda prompt=False: True)
    eng._poll_accessibility()
    assert eng.status.problems == []


def test_ensure_model_skips_when_cached(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    eng = vt.DictationEngine(vt.Config())
    calls = []

    def fake_snapshot(repo_id, **kw):
        calls.append(kw.get("local_files_only", False))
        return "/cached"

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    asyncio.new_event_loop().run_until_complete(
        eng._ensure_model("org/model", vt.ASR_MODEL_PATTERNS, "speech model")
    )
    assert calls == [True] and eng.status.phase == "starting"


def test_ensure_model_downloads_with_progress(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    eng = vt.DictationEngine(vt.Config())
    seen = []

    def fake_snapshot(repo_id, **kw):
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        bar = kw["tqdm_class"](total=100, unit="B", disable=True)
        bar.update(100)
        seen.append(eng.status.phase)
        return "/downloaded"

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    asyncio.new_event_loop().run_until_complete(
        eng._ensure_model("org/model", vt.ASR_MODEL_PATTERNS, "speech model")
    )
    assert seen == ["downloading"]
    assert eng.status.downloaded == eng.status.total == 100


def _menu_engine(**cfg):
    eng = vt.DictationEngine(vt.Config(**cfg))
    eng.status.phase = "ready"
    return eng


def test_describe_status_ready_and_recording():
    from parlando import menubar

    eng = _menu_engine(mode="record")
    v = menubar.describe_status(eng)
    assert v["primary"].startswith("Ready — tap right ⌥ Option")
    assert v["badge"] == "" and v["icon"] == "idle" and v["can_toggle"]
    assert v["toggle"] == "Start recording" and v["action"] is None

    eng.recording = True
    eng.status.phase = "recording"
    eng.rec_samples = 3 * vt.SAMPLE_RATE
    v = menubar.describe_status(eng)
    assert v["primary"] == "● Recording 3 s" and v["icon"] == "recording"
    assert v["toggle"] == "Stop recording and type"


def test_describe_status_downloading_and_loading():
    from parlando import menubar

    eng = _menu_engine()
    eng.status.phase = "downloading"
    eng.status.detail = "speech model · Qwen3-ASR-1.7B-8bit"
    eng.status.total, eng.status.downloaded = 2 * (1 << 30), 1 << 30
    v = menubar.describe_status(eng)
    assert v["badge"] == "50%" and v["primary"] == "Downloading speech model… 50%"
    assert v["secondary"] == "1.0 GB of 2.0 GB · first run only"
    assert not v["can_toggle"]

    eng.status.total = 0
    v = menubar.describe_status(eng)
    assert v["badge"] == "⬇" and v["primary"] == "Downloading speech model…"

    eng.status.phase, eng.status.detail = "loading", "speech model · X"
    v = menubar.describe_status(eng)
    assert v["badge"] == "…" and v["primary"] == "Loading speech model · X…"


def test_describe_status_problem_offers_settings_action():
    from parlando import menubar

    eng = _menu_engine()
    eng.status.problems = ["accessibility"]
    v = menubar.describe_status(eng)
    assert v["badge"] == "!" and v["action"] == "accessibility"
    assert "Accessibility" in v["primary"]
    assert "Accessibility" in menubar._PROBLEM_TEXT["accessibility"][2]
    assert "accessibility" in menubar.SETTINGS_PANES
    assert v["can_toggle"]  # menu-driven recording still works


def test_describe_status_language_badge_and_stream_mode():
    from parlando import menubar

    eng = _menu_engine(mode="stream", language="English")
    v = menubar.describe_status(eng)
    assert v["badge"] == menubar.LANGUAGE_BADGES["English"]
    assert v["toggle"] == "Pause" and v["primary"].startswith("Listening")
    eng.paused = True
    eng.status.phase = "paused"
    v = menubar.describe_status(eng)
    assert v["toggle"] == "Resume" and v["icon"] == "paused" and v["primary"] == "Paused"


# -----------------------------------------------------------------------------
# Relaunch guard (macOS kills the app when Accessibility is toggled)
# -----------------------------------------------------------------------------


def test_guard_plist_waits_for_pid_then_opens_app():
    from parlando import menubar

    pl = menubar.guard_plist("com.x.guard", 4242, Path("/Applications/Parlando.app"))
    assert pl["Label"] == "com.x.guard" and pl["RunAtLoad"] is True
    sh = pl["ProgramArguments"]
    assert sh[:2] == ["/bin/sh", "-c"]
    assert "kill -0 4242" in sh[2] and "/usr/bin/open -g '/Applications/Parlando.app'" in sh[2]


def test_relaunch_guard_start_stop(monkeypatch, tmp_path):
    from parlando import menubar

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[:2])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(menubar.subprocess, "run", fake_run)
    monkeypatch.setattr(menubar, "GUARD_PLIST", tmp_path / "guard.plist")
    g = menubar.RelaunchGuard(Path("/Applications/Parlando.app"))
    g.start()
    assert g.active and (tmp_path / "guard.plist").exists()
    assert ["launchctl", "bootstrap"] in calls
    g.start()  # idempotent
    assert calls.count(["launchctl", "bootstrap"]) == 1
    g.stop()
    assert not g.active and calls[-1] == ["launchctl", "bootout"]
    g.stop()  # idempotent
    assert calls.count(["launchctl", "bootout"]) == 2  # one pre-clean in start, one in stop


def test_relaunch_guard_inactive_outside_bundle(monkeypatch):
    from parlando import menubar

    monkeypatch.delenv("PARLANDO_APP_BUNDLE", raising=False)
    assert menubar.app_bundle_path() is None
    monkeypatch.setattr(menubar.subprocess, "run",
                        lambda *a, **k: pytest.fail("no launchctl without a bundle"))
    g = menubar.RelaunchGuard(menubar.app_bundle_path())
    g.start()
    assert not g.active
    monkeypatch.setenv("PARLANDO_APP_BUNDLE", "/Applications/Parlando.app")
    assert menubar.app_bundle_path() == Path("/Applications/Parlando.app")


# -----------------------------------------------------------------------------
# Personal vocabulary (ASR context biasing)
# -----------------------------------------------------------------------------


def test_load_vocabulary_merges_and_dedupes(monkeypatch, tmp_path):
    f = tmp_path / "vocabulary.txt"
    f.write_text("MLX\n# comment\nPyPI\n\nClaude Code\n", encoding="utf-8")
    monkeypatch.setattr(vt, "VOCAB_FILE", f)
    terms = vt.load_vocabulary("uv, mlx, Wispr Flow")
    assert terms == ["MLX", "PyPI", "Claude Code", "uv", "Wispr Flow"]  # mlx dedup


def test_load_vocabulary_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(vt, "VOCAB_FILE", tmp_path / "nope.txt")
    assert vt.load_vocabulary("MLX") == ["MLX"]
    assert vt.load_vocabulary() == []


def test_vocab_context():
    assert vt.vocab_context([]) == ""
    ctx = vt.vocab_context(["MLX", "PyPI"])
    assert "MLX, PyPI" in ctx and "exactly" in ctx


def test_polish_guard_allows_vocab_correction():
    """'Meleiks' -> 'MLX' is a spelling fix, not a hallucinated addition."""
    raw = "we use meleiks for on device inference"
    fixed = "We use MLX for on device inference."
    assert not vt.polish_guard(raw, fixed)                     # vocab yokken red
    assert vt.polish_guard(raw, fixed, allowed=("MLX",))       # vocab ile kabul


def test_engine_loads_vocab_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(vt, "VOCAB_FILE", tmp_path / "nope.txt")
    eng = vt.DictationEngine(vt.Config(vocab="MLX, uv"))
    assert eng.vocab_terms == ("MLX", "uv")


def test_vocab_hot_reload(monkeypatch, tmp_path):
    f = tmp_path / "vocabulary.txt"
    f.write_text("MLX\n", encoding="utf-8")
    monkeypatch.setattr(vt, "VOCAB_FILE", f)
    eng = vt.DictationEngine(vt.Config())
    assert eng.vocab_terms == ("MLX",)
    # dosya değişir -> mtime farkı -> yeniden yüklenir
    f.write_text("MLX\nPyPI\n", encoding="utf-8")
    import os
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    eng.reload_vocab_if_changed()
    assert eng.vocab_terms == ("MLX", "PyPI")


def test_add_vocab_term(monkeypatch, tmp_path):
    f = tmp_path / "vocabulary.txt"
    monkeypatch.setattr(vt, "VOCAB_FILE", f)
    eng = vt.DictationEngine(vt.Config())
    assert eng.add_vocab_term("  MLX  ") is True
    assert eng.vocab_terms == ("MLX",)
    assert eng.add_vocab_term("mlx") is False   # tekrar eklenmez
    assert eng.add_vocab_term("   ") is False   # boş reddedilir
    assert eng.add_vocab_term("PyPI") is True
    assert f.read_text(encoding="utf-8") == "MLX\nPyPI\n"
