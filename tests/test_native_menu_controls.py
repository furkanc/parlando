"""Exercise shortcut delivery while Cocoa is running in menu tracking mode."""

import threading
from types import SimpleNamespace

import pytest

AppKit = pytest.importorskip("AppKit")
Quartz = pytest.importorskip("Quartz")
from Foundation import NSDate, NSRunLoop  # noqa: E402
from rumps import MenuItem  # noqa: E402
from rumps.rumps import Menu  # noqa: E402

from parlando import engine, menu_controls  # noqa: E402


@pytest.mark.parametrize("codes, expected", [
    ([(54, 1 << 20, True), (54, 0, True)], "cmd_r"),
    ([(55, 1 << 20, True), (55, 0, True)], "cmd_l"),
    ([(58, 1 << 19, True), (49, 1 << 19, False)], "<alt>+<space>"),
    ([(53, 0, False)], "alt_r"),  # Escape cancels and keeps the original setting.
])
def test_event_tap_delivers_shortcuts_during_menu_tracking(codes, expected, monkeypatch):
    AppKit.NSApplication.sharedApplication()
    monkeypatch.setattr(menu_controls, "save_menu_config", lambda cfg: None)
    listener_calls = []
    main_thread = threading.get_ident()

    def make_listener(**kwargs):
        listener_calls.append(kwargs)
        return SimpleNamespace(start=lambda: None, stop=lambda: listener_calls.append(threading.get_ident()))

    monkeypatch.setattr(menu_controls, "Listener", make_listener)
    eng = SimpleNamespace(cfg=engine.Config(), status=SimpleNamespace(phase="ready"), clipboard=None,
                          begin_hotkey_capture=lambda: None, end_hotkey_capture=lambda: None)
    eng.request_hotkey = lambda key: setattr(eng.cfg, "hotkey", key)
    items = [MenuItem("Enter"), MenuItem("Copy"), MenuItem("Hotkey")]
    menu = Menu()
    menu.update(items)
    controls = menu_controls.MenuControls.alloc().init()
    controls.get_engine = lambda: eng
    controls.attach(*items, menu._menu)
    controls.recordHotkey_(None)
    callback = listener_calls[0]["darwin_intercept"]
    events = []
    for code, flags, modifier in codes:
        event = Quartz.CGEventCreateKeyboardEvent(None, code, True)
        event_type = Quartz.kCGEventFlagsChanged if modifier else Quartz.kCGEventKeyDown
        Quartz.CGEventSetType(event, event_type)
        Quartz.CGEventSetFlags(event, flags)
        events.append((event_type, event))
    results = []
    worker = threading.Thread(target=lambda: results.extend(callback(*event) for event in events))
    try:
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert results == [None] * len(events)
        # Delivery must run in the menu's mode, without returning to the default loop.
        assert controls.capturing
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            AppKit.NSEventTrackingRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1))
        assert not controls.capturing
        assert eng.cfg.hotkey == expected
        assert controls.capture_listener is None
        assert listener_calls[-1] == main_thread
        assert callback(*events[-1]) == events[-1][1]  # Monitoring ends with capture.
    finally:
        controls.end_capture()
