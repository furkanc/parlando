"""Native inline preferences for the menu bar app."""

import objc
from AppKit import (
    NSBeep,
    NSButton,
    NSColor,
    NSControlSizeSmall,
    NSEvent,
    NSEventTrackingRunLoopMode,
    NSEventTypeFlagsChanged,
    NSEventTypeKeyDown,
    NSFont,
    NSMenuDidEndTrackingNotification,
    NSNotificationCenter,
    NSObject,
    NSSwitchButton,
    NSTextField,
    NSView,
)
from Foundation import NSRunLoopCommonModes

from parlando import engine
from parlando.hotkeys import CANCEL_CAPTURE, HotkeyCapture, shortcut_symbols
from parlando.keyboard_listener import Listener
from parlando.menubar import format_hotkey, parse_clipboard_seconds, save_menu_config


def label(text, frame):
    field = NSTextField.labelWithString_(text)
    field.setFrame_(frame)
    field.setFont_(NSFont.menuFontOfSize_(0))
    return field


class CheckboxLabel(NSTextField):
    def hitTest_(self, point):
        # The checkbox extends behind its label, so the entire title stays clickable.
        return None


class MenuControls(NSObject):
    def init(self):
        self = objc.super(MenuControls, self).init()
        if self is None:
            return None
        self.get_engine = lambda: None
        self.capture_listener = None
        self.capture_generation = 0
        self.capturing = False
        self.editing = False
        self.capture = HotkeyCapture()
        return self

    @objc.python_method
    def attach(self, enter_item, clipboard_item, hotkey_item, menu):
        self.enter_view = NSView.alloc().initWithFrame_(((0, 0), (350, 28)))
        self.enter = self.make_checkbox("Press Enter after each utterance", ((12, 3), (326, 22)), "toggleEnter:")
        self.enter_view.addSubview_(self.enter)
        enter_item._menuitem.setView_(self.enter_view)

        self.clipboard_view = NSView.alloc().initWithFrame_(((0, 0), (350, 30)))
        self.copy = self.make_checkbox("Copy to clipboard", ((12, 4), (178, 22)), "toggleCopy:")
        self.clipboard_view.addSubview_(self.copy)
        self.seconds = NSTextField.alloc().initWithFrame_(((200, 4), (48, 22)))
        self.seconds.setFont_(NSFont.menuFontOfSize_(0))
        self.seconds.setControlSize_(NSControlSizeSmall)
        self.seconds.setAlignment_(2)  # right aligned
        self.seconds.setStringValue_("3")
        self.seconds.setAccessibilityLabel_("Change copy to clipboard duration")
        self.seconds.setToolTip_("Change copy to clipboard duration")
        self.seconds.setDelegate_(self)
        self.clipboard_view.addSubview_(self.seconds)
        self.clipboard_view.addSubview_(label("seconds", ((256, 4), (80, 22))))
        clipboard_item._menuitem.setView_(self.clipboard_view)

        self.hotkey_view = NSView.alloc().initWithFrame_(((0, 0), (350, 30)))
        self.hotkey_view.addSubview_(label("Change hotkey", ((14, 4), (110, 22))))
        self.current_key = label("Right ⌥", ((122, 4), (133, 22)))
        self.current_key.setAlignment_(2)
        self.hotkey_view.addSubview_(self.current_key)
        self.record = NSButton.alloc().initWithFrame_(((265, 2), (73, 26)))
        self.record.setTitle_("Record")
        self.record.setBezelStyle_(1)
        self.record.setControlSize_(NSControlSizeSmall)
        self.record.setTarget_(self)
        self.record.setAction_("recordHotkey:")
        self.hotkey_view.addSubview_(self.record)
        hotkey_item._menuitem.setView_(self.hotkey_view)
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self, "menuClosed:", NSMenuDidEndTrackingNotification, menu)
        self.refresh()

    @objc.python_method
    def make_checkbox(self, title, frame, action):
        button = NSButton.alloc().initWithFrame_(frame)
        button.setButtonType_(NSSwitchButton)
        button.setContentTintColor_(NSColor.systemGrayColor())
        button.setTitle_("")
        button.setAccessibilityLabel_(title)
        button.setFont_(NSFont.menuFontOfSize_(0))
        button.setTarget_(self)
        button.setAction_(action)
        # NSButton's content tint also affects its title. Draw the title as an
        # independent native label so only the checkbox uses the gray tint.
        text = CheckboxLabel.labelWithString_(title)
        text.setFrame_(((20, 0), (frame[1][0] - 20, frame[1][1])))
        text.setFont_(NSFont.menuFontOfSize_(0))
        text.setTextColor_(NSColor.labelColor())
        button.addSubview_(text)
        return button

    @objc.python_method
    def save(self):
        eng = self.get_engine()
        if eng:
            try:
                save_menu_config(eng.cfg)
            except OSError as exc:
                engine.LOGGER.warning("Could not save menu settings: %s", exc)

    @objc.python_method
    def refresh(self):
        eng = self.get_engine()
        cfg = eng.cfg if eng else engine.Config()
        ready = bool(eng and eng.status.phase in ("ready", "paused"))
        self.enter.setState_(int(cfg.send_enter))
        self.enter.setEnabled_(ready and not self.capturing)
        self.copy.setState_(int(cfg.copy_clipboard))
        self.copy.setEnabled_(ready and not self.capturing)
        self.seconds.setEnabled_(ready and cfg.copy_clipboard and not self.capturing)
        if not self.editing:
            self.seconds.setStringValue_(f"{cfg.clipboard_seconds:g}")
        self.record.setEnabled_(ready or self.capturing)
        if not self.capturing:
            self.current_key.setStringValue_(shortcut_symbols(cfg.hotkey))
            self.current_key.setAccessibilityLabel_(format_hotkey(cfg.hotkey))
            self.current_key.setToolTip_(format_hotkey(cfg.hotkey))

    def toggleEnter_(self, sender):
        eng = self.get_engine()
        if eng:
            eng.cfg.send_enter = bool(sender.state())
            self.save()
            self.refresh()

    def toggleCopy_(self, sender):
        eng = self.get_engine()
        if eng:
            eng.cfg.copy_clipboard = bool(sender.state())
            if not eng.cfg.copy_clipboard and eng.clipboard is not None:
                eng.clipboard.restore()
            self.save()
            self.refresh()

    def controlTextDidBeginEditing_(self, notification):
        self.editing = True

    def controlTextDidChange_(self, notification):
        # Valid edits are saved while typing; incomplete edits wait for completion.
        self.commit_duration(False)

    def controlTextDidEndEditing_(self, notification):
        self.editing = False
        self.commit_duration(True)

    @objc.python_method
    def commit_duration(self, final):
        eng = self.get_engine()
        if not eng:
            return
        try:
            eng.cfg.clipboard_seconds = parse_clipboard_seconds(self.seconds.stringValue())
            self.save()
        except (TypeError, ValueError):
            if final:
                self.seconds.setStringValue_(f"{eng.cfg.clipboard_seconds:g}")
                NSBeep()

    def recordHotkey_(self, sender):
        if self.capturing:
            self.end_capture()
            return
        eng = self.get_engine()
        if not eng or eng.status.phase not in ("ready", "paused"):
            return
        self.capture = HotkeyCapture()
        self.capturing = True
        self.capture_generation += 1
        eng.begin_hotkey_capture()
        self.record.setTitle_("Cancel")
        self.current_key.setStringValue_("Press keys…")
        self.current_key.setAccessibilityLabel_("Press a shortcut; Escape cancels")
        # NSMenu tracks keyboard events outside the application's normal local
        # NSEvent monitor. Use the same system event tap as the dictation hotkey,
        # and deliver results on the main thread in the menu's tracking mode.
        self.capture_listener = Listener(darwin_intercept=self.intercept_event)
        self.capture_listener.start()
        self.refresh()

    @objc.python_method
    def intercept_event(self, event_type, cg_event):
        if not self.capturing:
            return cg_event
        generation = self.capture_generation
        if event_type not in (NSEventTypeFlagsChanged, NSEventTypeKeyDown):
            return None
        event = NSEvent.eventWithCGEvent_(cg_event)
        self.performSelectorOnMainThread_withObject_waitUntilDone_modes_(
            "captureEvent:", [generation, event], False,
            [NSRunLoopCommonModes, NSEventTrackingRunLoopMode])
        return None  # Suppress captured keys before macOS dispatches menu commands.

    def captureEvent_(self, payload):
        generation, event = payload
        if not self.capturing or generation != self.capture_generation:
            return
        if event.type() == NSEventTypeFlagsChanged:
            result = self.capture.flags_changed(event.keyCode(), event.modifierFlags())
        else:
            result = self.capture.key_down(event.keyCode(), event.charactersByApplyingModifiers_(0) or "",
                                           event.modifierFlags())
        if result == CANCEL_CAPTURE:
            self.end_capture()
        elif result:
            eng = self.get_engine()
            if eng:
                eng.request_hotkey(result)
            self.end_capture()
        elif event.type() == NSEventTypeKeyDown:
            self.current_key.setStringValue_("Use a modifier")

    @objc.python_method
    def end_capture(self):
        if self.capture_listener is not None:
            self.capture_listener.stop()
            self.capture_listener = None
        if self.capturing:
            eng = self.get_engine()
            if eng:
                eng.end_hotkey_capture()
        self.capturing = False
        self.capture_generation += 1
        self.record.setTitle_("Record")
        self.refresh()

    def menuClosed_(self, notification):
        self.end_capture()
        if self.editing:
            self.editing = False
            self.commit_duration(True)
