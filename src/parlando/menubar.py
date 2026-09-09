"""parlando menu bar app.

Runs the dictation engine in the background; the menu bar offers start/stop,
language switching, Enter mode and quit. No terminal window needed.

Usage:
    parlando-menubar                 # start in the menu bar
    parlando-menubar --install-login # auto-start at login
    parlando-menubar --uninstall-login

Menu bar icon (template, adapts to light/dark): outline mic = ready,
filled mic = recording, slashed mic = paused; a small "ᵗʳ" badge appears
when the language is Turkish. The global hotkey works too (default:
single tap of right Option).
"""

from __future__ import annotations

import asyncio
import plistlib
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from parlando import engine

LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "com.parlando.menubar.plist"

LANGUAGES = {
    "English": "English",
    "Türkçe": "Turkish",
}

# Template icons (black + alpha): macOS recolors them to match the menu bar
# in light and dark mode. Regenerate with scripts/make_icons.py.
_ASSETS = Path(__file__).resolve().parent / "assets"
ICONS = {
    "idle": str(_ASSETS / "mic.png"),
    "recording": str(_ASSETS / "mic-recording.png"),
    "paused": str(_ASSETS / "mic-paused.png"),
}


# -----------------------------------------------------------------------------
# Start at login (LaunchAgent)
# -----------------------------------------------------------------------------


def install_login() -> int:
    exe = shutil.which("parlando-menubar")
    if not exe:
        print(
            "error: 'parlando-menubar' not found in PATH; install parlando first",
            file=sys.stderr,
        )
        return 1
    plist = {
        "Label": "com.parlando.menubar",
        "ProgramArguments": [exe],
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardErrorPath": str(Path.home() / "Library/Logs/parlando-agent.log"),
    }
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    with open(LAUNCH_AGENT, "wb") as fh:
        plistlib.dump(plist, fh)
    subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT)], capture_output=True)
    subprocess.run(["launchctl", "load", str(LAUNCH_AGENT)], check=False)
    print(f"Installed: {LAUNCH_AGENT}\nparlando will start at your next login.")
    return 0


def uninstall_login() -> int:
    subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT)], capture_output=True)
    if LAUNCH_AGENT.exists():
        LAUNCH_AGENT.unlink()
        print("Uninstalled.")
    else:
        print("Not installed.")
    return 0


# -----------------------------------------------------------------------------
# Menu bar
# -----------------------------------------------------------------------------


def run_menubar() -> int:
    import rumps

    engine_holder: dict = {"engine": None}

    class ParlandoApp(rumps.App):
        def __init__(self) -> None:
            super().__init__(
                "parlando",
                title=None,
                icon=ICONS["idle"],
                template=True,
                quit_button=None,
            )
            self._icon_state = "idle"
            self.toggle_item = rumps.MenuItem(
                "Start recording", callback=self.on_toggle
            )
            self.enter_item = rumps.MenuItem(
                "Press Enter after each utterance", callback=self.on_enter_mode
            )
            self.lang_items = {
                asr_name: rumps.MenuItem(label, callback=self.on_language)
                for label, asr_name in LANGUAGES.items()
            }
            self.menu = [
                self.toggle_item,
                self.enter_item,
                {"Language": list(self.lang_items.values())},
                None,
                rumps.MenuItem("Open log file", callback=self.on_log),
                rumps.MenuItem("Quit", callback=self.on_quit),
            ]
            # UI updates must happen on the main thread: polling engine state
            # from a main-loop timer is the thread-safe pattern.
            self._poll = rumps.Timer(self._refresh, 0.5)
            self._poll.start()

        def _set_icon(self, state: str) -> None:
            if state != self._icon_state:
                self._icon_state = state
                self.icon = ICONS[state]

        def _refresh(self, _timer) -> None:
            eng = engine_holder.get("engine")
            if eng is None:
                return
            for lang, item in self.lang_items.items():
                item.state = int(eng.cfg.language == lang)
            # Small text badge next to the icon for non-default language.
            badge = "" if eng.cfg.language == "English" else "ᵗʳ"
            if self.title != badge:
                self.title = badge
            if eng.cfg.mode == "record":
                if eng.recording:
                    self._set_icon("recording")
                    self.toggle_item.title = "Stop recording and type"
                else:
                    self._set_icon("idle")
                    self.toggle_item.title = "Start recording"
                return
            if eng.paused:
                self._set_icon("paused")
                self.toggle_item.title = "Resume"
            elif eng.vad.in_speech:
                self._set_icon("recording")
                self.toggle_item.title = "Pause"
            else:
                self._set_icon("idle")
                self.toggle_item.title = "Pause"

        def on_toggle(self, _item) -> None:
            eng = engine_holder["engine"]
            if eng:
                eng.request_toggle()

        def on_enter_mode(self, item) -> None:
            eng = engine_holder["engine"]
            if eng:
                eng.cfg.send_enter = not eng.cfg.send_enter
                item.state = eng.cfg.send_enter

        def on_language(self, item) -> None:
            eng = engine_holder.get("engine")
            if not eng:
                return
            # Menu title -> ASR language tag; the model is multilingual, so
            # switching is instant with no model reload.
            for lang, mi in self.lang_items.items():
                if mi is item:
                    eng.cfg.language = lang
                    engine.LOGGER.info("language changed: %s", lang)
                    break

        def on_log(self, _item) -> None:
            subprocess.run(["open", str(engine.LOG_PATH)], check=False)

        def on_quit(self, _item) -> None:
            rumps.quit_application()

    app = ParlandoApp()

    def engine_thread() -> None:
        engine._setup_logging()
        eng = engine.DictationEngine(engine.Config())
        engine_holder["engine"] = eng
        try:
            asyncio.run(eng.run())
        except Exception:  # noqa: BLE001
            # rumps notifications need an app bundle; a log line is enough.
            engine.LOGGER.exception("engine crashed")
            print(
                f"[error] engine stopped — details: {engine.LOG_PATH}",
                file=sys.stderr,
            )

    threading.Thread(target=engine_thread, daemon=True).start()
    app.run()
    return 0


def main() -> int:
    if "--install-login" in sys.argv:
        return install_login()
    if "--uninstall-login" in sys.argv:
        return uninstall_login()
    return run_menubar()


if __name__ == "__main__":
    raise SystemExit(main())
