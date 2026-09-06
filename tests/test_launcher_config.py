"""Round-trip della configurazione: launcher -> SimulationConfig -> config.properties."""

from __future__ import annotations

import pytest

from antelligent.starter import _parse_properties, _read_bool


def _tk_root():
    """Radice Tk nascosta, o skip se non c'e' un display (CI headless)."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("nessun display disponibile per Tk")
    root.withdraw()
    return root


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("captureScreenshots=1", True),
        ("captureScreenshots=true", True),
        ("captureScreenshots=ON", True),
        ("captureScreenshots=0", False),
        ("captureScreenshots=false", False),
        ("", False),  # chiave assente (config.properties precedente): disattivato
    ],
)
def test_read_bool_parses_capture_flag(text: str, expected: bool) -> None:
    assert _read_bool(_parse_properties(text), "captureScreenshots") is expected


@pytest.mark.parametrize("default", [False, True])
def test_launcher_checkbox_reflects_default_and_is_returned(make_config, default: bool) -> None:
    from antelligent.launcher import AntelligentLauncher

    root = _tk_root()
    try:
        launched: list = []
        cfg = make_config(capture_screenshots=default)
        launcher = AntelligentLauncher(root, cfg, launched.append)
        assert launcher._capture_var.get() is default  # noqa: SLF001
        launcher._on_start()  # noqa: SLF001
        assert launched[0].capture_screenshots is default
    finally:
        root.destroy()


def test_launcher_checkbox_toggle_is_picked_up(make_config) -> None:
    from antelligent.launcher import AntelligentLauncher

    root = _tk_root()
    try:
        launched: list = []
        launcher = AntelligentLauncher(root, make_config(capture_screenshots=False), launched.append)
        launcher._capture_var.set(True)  # noqa: SLF001 - l'utente flagga la casella
        launcher._on_start()  # noqa: SLF001
        assert launched[0].capture_screenshots is True
    finally:
        root.destroy()


def test_saved_properties_contain_the_capture_key(tmp_path, monkeypatch, make_config) -> None:
    from antelligent import paths
    from antelligent.starter import AntelligentStarter

    config_path = tmp_path / "config.properties"
    monkeypatch.setattr(paths, "CONFIG_PATH", config_path)

    starter = AntelligentStarter.__new__(AntelligentStarter)
    starter._save_config(make_config(capture_screenshots=True))  # noqa: SLF001
    assert _read_bool(_parse_properties(config_path.read_text(encoding="utf-8")),
                      "captureScreenshots") is True

    starter._save_config(make_config(capture_screenshots=False))  # noqa: SLF001
    assert _read_bool(_parse_properties(config_path.read_text(encoding="utf-8")),
                      "captureScreenshots") is False
