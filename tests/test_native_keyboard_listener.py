"""Guard the actual pynput macOS startup path against Carbon layout lookups."""

import sys
import threading

import pytest

pytest.importorskip("AppKit")
from pynput import keyboard  # noqa: E402

from parlando.keyboard_listener import GlobalHotKeys, Listener  # noqa: E402


@pytest.mark.parametrize("upstream, fixed", [
    (lambda: keyboard.Listener(), lambda: Listener()),
    (lambda: keyboard.GlobalHotKeys({"<alt>+<space>": lambda: None}),
     lambda: GlobalHotKeys({"<alt>+<space>": lambda: None})),
    (lambda: keyboard.Listener(darwin_intercept=lambda kind, event: event),
     lambda: Listener(darwin_intercept=lambda kind, event: event)),
])
def test_background_startup_skips_faulting_carbon_lookup(upstream, fixed, monkeypatch):
    backend = sys.modules[keyboard.Listener.__module__]
    event_loop_class = keyboard.Listener.__mro__[1]
    entered_event_loop = []

    def forbidden_layout_lookup():
        # A Python exception makes the old failure path testable without
        # deliberately causing a native SIGTRAP in the pytest process.
        raise RuntimeError("Carbon keyboard layout called from listener thread")

    monkeypatch.setattr(backend, "keycode_context", forbidden_layout_lookup)
    monkeypatch.setattr(event_loop_class, "_run", lambda self: entered_event_loop.append(self))

    def start_on_worker(listener):
        failures = []

        def run():
            try:
                listener._run()
            except RuntimeError as exc:
                failures.append(str(exc))

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
        return failures

    assert start_on_worker(upstream()) == ["Carbon keyboard layout called from listener thread"]
    safe = fixed()
    assert start_on_worker(safe) == []
    assert entered_event_loop == [safe]


def test_global_shortcuts_keep_canonicalization_and_release_behavior():
    toggles = []
    listener = GlobalHotKeys({"<alt>+<space>": lambda: toggles.append(True)})
    for alt in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        listener.on_press(alt, False)
        listener.on_press(keyboard.Key.space, False)
        listener.on_press(keyboard.Key.space, False)
        listener.on_release(keyboard.Key.space, False)
        listener.on_release(alt, False)
    assert toggles == [True, True]
