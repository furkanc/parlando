"""Menu preferences persist without applying malformed or unrelated settings."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from parlando import engine, menubar  # noqa: E402


def test_menu_settings_remember_choices(tmp_path):
    path = tmp_path / "settings.json"
    cfg = engine.Config(language="Auto", hotkey="<cmd>+<shift>+d", copy_clipboard=False,
                        clipboard_seconds=5)
    menubar.save_menu_config(cfg, path)
    loaded = menubar.load_menu_config(path)
    assert menubar.menu_preferences(loaded) == menubar.menu_preferences(cfg)


def test_invalid_menu_settings_preserve_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hotkey": "invalid", "language": "invalid",
                                "copy_clipboard": "false", "send_enter": 1,
                                "pipe": True}))
    cfg = menubar.load_menu_config(path)
    assert cfg.hotkey == "alt_r" and cfg.language == "Turkish"
    assert cfg.copy_clipboard and not cfg.send_enter and not cfg.pipe
    path.write_text("{invalid json")
    assert menubar.load_menu_config(path) == engine.Config()


@pytest.mark.parametrize("value, expected", [("Option+Space", "<alt>+<space>"),
                                             ("Command + Shift + D", "<cmd>+<shift>+d"),
                                             ("Right Option", "alt_r"),
                                             ("<ctrl>+<alt>+d", "<ctrl>+<alt>+d")])
def test_hotkey_input_accepts_readable_shortcuts(value, expected):
    assert menubar.parse_hotkey(value) == expected
    assert menubar.parse_hotkey(menubar.format_hotkey(expected)) == expected


@pytest.mark.parametrize("value", ["a", "Command+Command+A", "Shift", "Command+A+B", "Fn", None])
def test_hotkey_input_rejects_invalid_shortcuts(value):
    with pytest.raises(ValueError):
        menubar.parse_hotkey(value)


@pytest.mark.parametrize("value", [0, -1, "NaN", "inf", True, "three"])
def test_clipboard_duration_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        menubar.parse_clipboard_seconds(value)


def test_clipboard_duration_accepts_custom_seconds():
    assert menubar.parse_clipboard_seconds("5") == 5
    assert menubar.parse_clipboard_seconds("0.5") == 0.5
