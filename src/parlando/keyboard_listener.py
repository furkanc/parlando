"""Parlando's keyboard listeners, including the macOS 26 startup workaround."""

import sys

from pynput import keyboard

if sys.platform == "darwin":
    class Listener(keyboard.Listener):
        def _run(self):
            # pynput 1.8.2 reads a Carbon keyboard-layout context here on its
            # background thread. macOS can abort in TSMGetInputSourceProperty
            # with a main-queue assertion. The listener never uses that context:
            # event decoding reads CGEvent data instead. Skip only this wrapper
            # and retain pynput's event tap, callbacks and shutdown loop.
            return super(keyboard.Listener, self)._run()

    class GlobalHotKeys(keyboard.GlobalHotKeys):
        _run = Listener._run
else:
    Listener = keyboard.Listener
    GlobalHotKeys = keyboard.GlobalHotKeys
