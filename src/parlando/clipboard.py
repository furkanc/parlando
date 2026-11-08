"""Temporarily copy dictation without replacing a later user copy."""

import threading


class TemporaryClipboard:
    def __init__(self, duration: float = 3.0, pasteboard=None) -> None:
        from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString

        self.duration = duration
        self.pasteboard = pasteboard or NSPasteboard.generalPasteboard()
        self.item_class = NSPasteboardItem
        self.text_type = NSPasteboardTypeString
        self._lock = threading.RLock()
        self._timer = None
        self._previous = None
        self._change_count = None
        self._text = None

    def copy(self, text: str) -> None:
        with self._lock:
            self.restore()
            # Keep all representations, including images and file references.
            previous = []
            for item in self.pasteboard.pasteboardItems() or []:
                saved = self.item_class.alloc().init()
                for kind in item.types():
                    data = item.dataForType_(kind)
                    if data is not None:
                        saved.setData_forType_(data, kind)
                previous.append(saved)
            self.pasteboard.clearContents()
            if not self.pasteboard.setString_forType_(text, self.text_type):
                self.pasteboard.writeObjects_(previous)
                raise RuntimeError("Could not copy dictation to clipboard")
            self._previous = previous
            self._change_count = self.pasteboard.changeCount()
            self._text = text
            self._timer = threading.Timer(self.duration, self.restore, args=(self._change_count,))
            self._timer.daemon = True
            self._timer.start()

    def restore(self, expected_count=None) -> None:
        with self._lock:
            # A cancelled timer may already be waiting for this lock. It must
            # not restore a newer dictation's clipboard snapshot.
            if expected_count is not None and expected_count != self._change_count:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._previous is not None:
                # A new copy belongs to the user or another app; leave it alone.
                if (self.pasteboard.changeCount() == self._change_count
                        and self.pasteboard.stringForType_(self.text_type) == self._text):
                    self.pasteboard.clearContents()
                    if self._previous:
                        self.pasteboard.writeObjects_(self._previous)
                self._previous = None
                self._change_count = None
                self._text = None
