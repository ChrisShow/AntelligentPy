"""GUI di confronto: run **sequenziali** (prima euristica, poi RL) e cronometri per copia.

Le due copie non girano piu' in parallelo: l'RL parte solo quando l'euristica ha
concluso. Ogni copia ha la sua etichetta del tempo, che riporta lo stato
(in attesa / in corso / conclusa) e si **ferma** sul valore finale a fine run.
"""

from __future__ import annotations

import time

import pytest

from antelligent.simulation import simulation as sim_mod
from antelligent.simulation.simulation import _DONE, _RUNNING, _WAITING, ComparisonSimulation


def _tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("nessun display disponibile per Tk")
    root.withdraw()
    return root


@pytest.fixture
def gui(make_config, tmp_path, monkeypatch):
    """`ComparisonSimulation` isolata: niente scritture in results/, niente dialog modale."""
    monkeypatch.setattr(sim_mod.paths, "RESULTS_FILE", tmp_path / "results.txt")
    monkeypatch.setattr(sim_mod.paths, "SCREENSHOTS_DIR", tmp_path / "screenshots")
    dialogs: list = []
    monkeypatch.setattr(sim_mod, "SimulationResultsDialog",
                        lambda *a, **k: dialogs.append((a, k)))

    root = _tk_root()
    config = make_config(cols=8, rows=6, n_ants=3, n_seeds=10, seed_types=3,
                         stop_criterion=0, max_iterations=150, refresh_rate=1000)
    gui = ComparisonSimulation(root, config, lambda: None)
    gui._dialogs = dialogs  # type: ignore[attr-defined]
    try:
        yield gui
    finally:
        gui._stop_requested = True
        for pane in gui._panes:
            if pane.worker is not None:
                pane.worker.join(timeout=5)
        root.destroy()


def _pump(gui, seconds: float, observer=None) -> None:
    """Fa girare il loop di eventi Tk per un po', campionando lo stato."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        gui._root.update()
        if observer is not None:
            observer()
        time.sleep(0.005)


# --------------------------------------------------------------------------- #
# Cronometro per copia
# --------------------------------------------------------------------------- #

def test_panes_start_waiting_with_their_own_time_label(gui) -> None:
    heuristic, rl = gui._panes
    assert heuristic.mode == "heuristic" and rl.mode == "rl"
    assert heuristic.status == rl.status == _WAITING
    for pane in gui._panes:
        assert "in attesa" in gui._time_text(pane)
        assert gui._elapsed_seconds(pane) == 0.0
    # etichette di riga distinte per le due copie
    assert sim_mod._TIME_ROW_LABEL["heuristic"] != sim_mod._TIME_ROW_LABEL["rl"]


def test_time_label_wording_differs_per_state(gui) -> None:
    pane = gui._panes[0]
    seen = {}
    pane.status, pane.start_time = _WAITING, None
    seen[_WAITING] = gui._time_text(pane)
    pane.status, pane.start_time = _RUNNING, time.monotonic()
    seen[_RUNNING] = gui._time_text(pane)
    pane.status, pane.duration_ms = _DONE, 65_000
    seen[_DONE] = gui._time_text(pane)

    assert len(set(seen.values())) == 3, seen
    assert "in attesa" in seen[_WAITING]
    assert "in corso" in seen[_RUNNING]
    assert seen[_DONE].startswith("01:05") and "conclusa" in seen[_DONE]


def test_finished_pane_clock_is_frozen_on_its_final_value(gui) -> None:
    pane = gui._panes[0]
    pane.start_time = time.monotonic() - 3600  # partita un'ora fa
    pane.duration_ms = 12_000
    pane.status = _DONE
    first = gui._elapsed_seconds(pane)
    time.sleep(0.05)
    assert gui._elapsed_seconds(pane) == first == 12.0  # non avanza piu'


def test_running_pane_clock_advances(gui) -> None:
    pane = gui._panes[0]
    pane.status, pane.start_time = _RUNNING, time.monotonic()
    first = gui._elapsed_seconds(pane)
    time.sleep(0.05)
    assert gui._elapsed_seconds(pane) > first


# --------------------------------------------------------------------------- #
# Esecuzione sequenziale
# --------------------------------------------------------------------------- #

def test_start_launches_only_the_first_copy(gui) -> None:
    gui._on_start_close()
    heuristic, rl = gui._panes
    assert heuristic.status == _RUNNING
    assert rl.status == _WAITING          # l'RL non parte finche' l'euristica non finisce
    assert rl.worker is None
    assert rl.env.iterations == 0


def _run_to_completion(gui, observer=None, timeout: float = 20.0) -> bool:
    gui._on_start_close()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _pump(gui, 0.02, observer)
        if all(p.status == _DONE for p in gui._panes):
            _pump(gui, 0.05, observer)  # lascia sfilare gli ultimi eventi
            return True
    return False


def test_runs_never_overlap_and_keep_their_order(gui) -> None:
    heuristic, rl = gui._panes
    overlaps: list = []

    def observe() -> None:
        running = [p.mode for p in gui._panes if p.status == _RUNNING]
        if len(running) > 1:
            overlaps.append(tuple(running))

    assert _run_to_completion(gui, observe), "le run non sono terminate in tempo"

    assert not overlaps, f"le due copie hanno girato insieme: {overlaps}"
    assert heuristic.status == rl.status == _DONE
    assert heuristic.env.iterations > 0 and rl.env.iterations > 0
    # L'ordine e l'assenza di sovrapposizione, verificati sui tempi registrati:
    # l'RL parte solo *dopo* che l'euristica ha concluso.
    assert heuristic.start_time is not None and rl.start_time is not None
    assert heuristic.start_time < rl.start_time
    heuristic_end = heuristic.start_time + heuristic.duration_ms / 1000.0
    assert rl.start_time >= heuristic_end, (
        f"l'RL e' partito {heuristic_end - rl.start_time:.3f}s prima della fine dell'euristica"
    )


def test_second_copy_is_untouched_while_the_first_runs(gui) -> None:
    gui._on_start_close()
    heuristic, rl = gui._panes
    _pump(gui, 0.08)
    if heuristic.status == _RUNNING:  # se l'euristica e' ancora in corso, l'RL e' fermo
        assert rl.status == _WAITING
        assert rl.env.iterations == 0
        assert rl.env.total_moves == 0
        assert "in attesa" in gui._time_text(rl)


def test_results_dialog_only_after_both_copies_are_done(gui) -> None:
    seen_early: list = []

    def observe() -> None:
        if gui._dialogs and not all(p.status == _DONE for p in gui._panes):
            seen_early.append([p.status for p in gui._panes])

    assert _run_to_completion(gui, observe), "le run non sono terminate in tempo"
    assert not seen_early, f"dialog aperto con una copia ancora in corso: {seen_early}"
    assert gui._dialogs, "il dialog dei risultati non e' mai stato aperto"


def test_clocks_freeze_after_both_copies_are_done(gui) -> None:
    assert _run_to_completion(gui), "le run non sono terminate in tempo"

    frozen = [p.stat_time.cget("text") for p in gui._panes]
    assert all("conclusa" in text for text in frozen), frozen
    _pump(gui, 1.3)  # oltre un giro del timer a 1 Hz
    assert [p.stat_time.cget("text") for p in gui._panes] == frozen


def test_the_stats_panel_shows_the_iteration_count(gui) -> None:
    """Un'iterazione = un giro di ``tick()``, cioe' una mossa per ogni formica."""
    for pane in gui._panes:
        assert pane.stat_iterations.cget("text") == "0"

    assert _run_to_completion(gui), "le run non sono terminate in tempo"

    for pane in gui._panes:
        assert pane.env.iterations > 0
        assert pane.stat_iterations.cget("text") == str(pane.env.iterations)


def test_results_file_records_the_iterations_of_both_copies(gui, tmp_path) -> None:
    assert _run_to_completion(gui), "le run non sono terminate in tempo"

    rows = (tmp_path / "results.txt").read_text(encoding="utf-8").strip().splitlines()
    header, records = rows[0].split(";"), [r.split(";") for r in rows[1:]]
    mode_col, iter_col = header.index("mode"), header.index("iterazioni")
    written = {r[mode_col]: int(r[iter_col]) for r in records}

    assert written == {p.mode: p.env.iterations for p in gui._panes}
    assert all(value > 0 for value in written.values())
