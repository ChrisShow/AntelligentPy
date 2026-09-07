"""GUI di confronto affiancato euristica vs RL.

Una sola finestra con **due griglie gemelle** (stesso ``master_seed``), i pulsanti
Start/Visibility condivisi in basso e, sotto ciascuna griglia, il pannello
statistiche della rispettiva copia: tempo (aggiornato **1 volta al secondo**),
mosse totali, semi raccolti, entropia totale.

I due ambienti girano su **due thread worker indipendenti**; Tkinter viene toccato
solo dal thread principale via ``root.after``.
"""

from __future__ import annotations

import logging
import queue
import random
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

from PIL import Image, ImageGrab, ImageTk

from .. import paths
from ..config import SimulationConfig
from ..environment import Environment
from ..policies.heuristic import HeuristicPolicy
from ..policies.lumer_tabular import load_policy
from ..policies.tabular import QConfig, TabularQPolicy
from ..seeds.seed import BLANK_SEED_IMAGE
from ..seeds.seed_type import SeedType
from ..world import build_initial_state, resolve_master_seed
from .ant import BLACK_ANT_IMAGE
from .results_dialog import SimulationResultsDialog
from .simulation_result import SimulationResult

_LOGGER = logging.getLogger(__name__)

_RESULTS_HEADER = (
    "timestamp;mode;durataMs;iterazioni;righe;colonne;thread;formiche;semi;"
    "mosseTotali;semiRaccolti;entropiaIniziale;entropiaFinale;masterSeed;tipo"
)
_SCREENSHOT_EVERY_N_ITERATIONS = 200

#: Ogni quanti ms il thread principale drena la coda degli eventi dei worker.
#: Tkinter non e' thread-safe e ``root.after`` da un thread worker solleva
#: ``RuntimeError: main thread is not in main loop``: i worker quindi non toccano
#: Tk, accodano un evento e basta.
_EVENT_POLL_MS = 50
_EV_RENDER, _EV_SCREENSHOT, _EV_FINISHED = "render", "screenshot", "finished"

# Stati di una copia: le run sono SEQUENZIALI (prima l'euristica, poi l'RL), quindi
# una copia e' "in attesa" finche' l'altra non ha finito.
_WAITING, _RUNNING, _DONE = "waiting", "running", "done"

#: Etichetta della riga del tempo, diversa per copia (cosi' i due cronometri non si
#: confondono) e testo/colore dello stato accanto al valore.
_TIME_ROW_LABEL = {"heuristic": "tempo euristica", "rl": "tempo RL"}
_STATUS_TEXT = {_WAITING: "in attesa", _RUNNING: "in corso", _DONE: "conclusa"}
_STATUS_COLOR = {_WAITING: "gray50", _RUNNING: "#1a6f1a", _DONE: "#8a4b00"}


class _Pane:
    def __init__(self, mode: str, label: str, env: Environment, policy: object) -> None:
        self.mode = mode
        self.label = label
        self.env = env
        self.policy = policy
        self.canvas: Optional[tk.Canvas] = None
        self.stat_time: Optional[ttk.Label] = None
        self.stat_moves: Optional[ttk.Label] = None
        self.stat_seeds: Optional[ttk.Label] = None
        self.stat_entropy: Optional[ttk.Label] = None
        self.title_label: Optional[ttk.Label] = None
        self.worker: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.duration_ms = 0
        self.status = _WAITING
        self.ticks_since_paint = 0

    @property
    def done(self) -> bool:
        return self.status == _DONE


class ComparisonSimulation:
    def __init__(
        self,
        root: tk.Tk,
        config: SimulationConfig,
        on_new_simulation: Callable[[], None],
        *,
        rl_policy_path: Optional[Path] = None,
    ) -> None:
        self._root = root
        self._config = config
        self._on_new_simulation = on_new_simulation
        self._refresh_rate = max(1, config.refresh_rate)

        self._master_seed = resolve_master_seed(config.master_seed)
        init = build_initial_state(config, self._master_seed)
        _LOGGER.info("master_seed = %d", self._master_seed)

        env_h = Environment.from_initial_state(init, rng=random.Random())
        env_rl = Environment.from_initial_state(init, rng=random.Random())

        rl_policy, rl_label = self._make_rl_policy(env_rl, config, rl_policy_path)
        self._panes = [
            _Pane("heuristic", "Euristica", env_h, HeuristicPolicy(env_h.matrix, config.seed_types)),
            _Pane("rl", rl_label, env_rl, rl_policy),
        ]

        self._started = False
        self._gui_visible = True
        self._stop_requested = False
        self._closing = False
        self._finished = False
        self._screenshot_number = 0
        self._formatted_datetime = ""
        self._board_images: list[Image.Image] = []
        # handoff worker -> thread principale (i worker non chiamano mai Tk)
        self._events: "queue.Queue[tuple[str, Optional[_Pane]]]" = queue.Queue()

        self._cell_side = self._compute_cell_side()
        self._build_gui()
        self._photo_cache: dict[str, ImageTk.PhotoImage] = {}
        self._preload_images()
        for pane in self._panes:
            self._draw_pane(pane)

    # ------------------------------------------------------------------ #
    # Costruzione
    # ------------------------------------------------------------------ #

    def _make_rl_policy(
        self, env: Environment, config: SimulationConfig, rl_policy_path: Optional[Path]
    ) -> tuple[object, str]:
        heuristic = HeuristicPolicy(env.matrix, config.seed_types)
        if rl_policy_path is not None and Path(rl_policy_path).is_file():
            try:
                # load_policy sceglie la classe dalla famiglia scritta nel pickle
                # (tabellare base o variante Lumer-Faieta): codifiche di stato diverse.
                policy = load_policy(rl_policy_path, heuristic=heuristic)
                policy.freeze()
                return policy, "RL (addestrata)"
            except Exception:  # noqa: BLE001
                _LOGGER.warning("impossibile caricare %s, uso apprendimento live", rl_policy_path, exc_info=True)
        # nessuna politica addestrata: il pannello RL impara dal vivo durante la run
        # (unico thread che tocca la tabella -> sicuro).
        _LOGGER.warning(
            "nessuna politica RL addestrata (%s): il pannello RL apprende dal vivo. "
            "Genera una politica con: python -m antelligent.train",
            rl_policy_path,
        )
        return TabularQPolicy(config.seed_types, QConfig(), heuristic=heuristic), "RL (apprendimento live)"

    def _compute_cell_side(self) -> int:
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        cell_w = (screen_w // 2 - 120) // self._config.cols
        cell_h = (screen_h - 260) // self._config.rows
        return max(3, min(cell_w, cell_h))

    def _build_gui(self) -> None:
        self._toplevel = tk.Toplevel(self._root)
        self._toplevel.title("Antelligent - Euristica vs RL")
        self._toplevel.resizable(False, False)
        self._toplevel.protocol("WM_DELETE_WINDOW", self._quit_now)

        board_w = self._cell_side * self._config.cols + 20
        board_h = self._cell_side * self._config.rows + 20

        for col, pane in enumerate(self._panes):
            pane.title_label = ttk.Label(
                self._toplevel, text=pane.label, font=("TkDefaultFont", 11, "bold")
            )
            pane.title_label.grid(row=0, column=col, pady=(8, 2))
            pane.canvas = tk.Canvas(
                self._toplevel, width=board_w, height=board_h, background="white", highlightthickness=0
            )
            pane.canvas.grid(row=1, column=col, padx=6)
            pane.canvas._pane_label = pane.label  # type: ignore[attr-defined]

            stats = ttk.Frame(self._toplevel, padding=(6, 4))
            stats.grid(row=2, column=col, sticky="ew")
            pane.stat_time = self._add_stat(stats, 0, _TIME_ROW_LABEL.get(pane.mode, "tempo"))
            pane.stat_moves = self._add_stat(stats, 1, "mosse totali")
            pane.stat_seeds = self._add_stat(stats, 2, "semi raccolti")
            pane.stat_entropy = self._add_stat(stats, 3, "entropia totale")

        controls = ttk.Frame(self._toplevel)
        controls.grid(row=3, column=0, columnspan=len(self._panes), pady=8)
        self._start_close_btn = ttk.Button(controls, text="Start", command=self._on_start_close)
        self._start_close_btn.grid(row=0, column=0, padx=8)
        ttk.Button(controls, text="Visibility", command=self._on_toggle_visibility).grid(
            row=0, column=1, padx=8
        )
        ttk.Label(controls, text=f"seed: {self._master_seed}").grid(row=0, column=2, padx=8)

        self._toplevel.update_idletasks()
        self._center_window()
        self._refresh_stats()
        self._pump_events()

    @staticmethod
    def _add_stat(parent: ttk.Frame, row: int, name: str) -> ttk.Label:
        ttk.Label(parent, text=f"{name}:").grid(row=row, column=0, sticky="w", padx=(0, 8))
        value = ttk.Label(parent, text="-")
        value.grid(row=row, column=1, sticky="w")
        return value

    def _center_window(self) -> None:
        self._toplevel.update_idletasks()
        w, h = self._toplevel.winfo_width(), self._toplevel.winfo_height()
        x = (self._toplevel.winfo_screenwidth() - w) // 2
        y = (self._toplevel.winfo_screenheight() - h) // 2
        self._toplevel.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

    def _preload_images(self) -> None:
        size = max(1, self._cell_side - 5)
        rel_paths = {BLACK_ANT_IMAGE, BLANK_SEED_IMAGE}
        for seed_type in SeedType:
            rel_paths.add(seed_type.seed_image_path)
            rel_paths.add(seed_type.ant_image_path)
        for rel_path in rel_paths:
            image = Image.open(paths.IMAGES_DIR / rel_path).convert("RGBA")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            self._photo_cache[rel_path] = ImageTk.PhotoImage(image)

    # ------------------------------------------------------------------ #
    # Rendering (thread principale)
    # ------------------------------------------------------------------ #

    def _draw_pane(self, pane: _Pane) -> None:
        canvas = pane.canvas
        if canvas is None:
            return
        try:
            canvas.delete("all")
            cs = self._cell_side
            cols, rows = self._config.cols, self._config.rows
            right, bottom = 10 + cs * cols, 10 + cs * rows
            for i in range(rows + 1):
                canvas.create_line(10, 10 + cs * i, right, 10 + cs * i, fill="black")
            for j in range(cols + 1):
                canvas.create_line(10 + cs * j, 10, 10 + cs * j, bottom, fill="black")
            for seed in pane.env.seeds:
                pos = seed.position
                canvas.create_image(
                    13 + pos.x * cs, 14 + pos.y * cs, image=self._photo_cache[seed.image_path], anchor="nw"
                )
            for ant in pane.env.ants:
                pos = ant.position
                canvas.create_image(
                    13 + pos.x * cs, 14 + pos.y * cs, image=self._photo_cache[ant.image_path], anchor="nw"
                )
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ #
    # Handoff worker -> thread principale
    # ------------------------------------------------------------------ #

    def _post(self, event: str, pane: Optional[_Pane] = None) -> None:
        """Chiamabile dai worker: accoda e basta, nessuna chiamata a Tk."""
        self._events.put((event, pane))

    def _pump_events(self) -> None:
        """Drena la coda sul thread principale, l'unico che puo' toccare Tk.

        Il riarmo del timer e' in ``finally``: se una callback fallisce il pump
        **non** deve morire, altrimenti gli eventi successivi (fra cui la fine di
        una copia) resterebbero in coda per sempre e la run si pianterebbe.
        """
        try:
            while True:
                try:
                    event, pane = self._events.get_nowait()
                except queue.Empty:
                    break
                try:
                    if event == _EV_RENDER and self._gui_visible and pane is not None:
                        self._draw_pane(pane)
                    elif event == _EV_SCREENSHOT:
                        self._take_screenshot()
                    elif event == _EV_FINISHED and pane is not None:
                        self._pane_finished(pane)
                except tk.TclError:
                    return  # finestra distrutta: inutile insistere
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("errore gestendo l'evento %r", event)
        finally:
            if not self._closing:
                try:
                    self._root.after(_EVENT_POLL_MS, self._pump_events)
                except (RuntimeError, tk.TclError):
                    pass

    @staticmethod
    def _mmss(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def _elapsed_seconds(self, pane: _Pane) -> float:
        """Cronometro della copia: **fermo** al valore finale una volta conclusa."""
        if pane.status == _DONE:
            return pane.duration_ms / 1000.0
        if pane.status == _RUNNING and pane.start_time is not None:
            return time.monotonic() - pane.start_time
        return 0.0  # in attesa: non e' ancora partita

    def _time_text(self, pane: _Pane) -> str:
        state = _STATUS_TEXT[pane.status]
        if pane.status == _WAITING:
            return f"--:--  ({state})"
        return f"{self._mmss(self._elapsed_seconds(pane))}  ({state})"

    def _update_stat_labels(self) -> None:
        for pane in self._panes:
            if pane.stat_time is None:
                continue
            try:
                pane.stat_time.config(text=self._time_text(pane),
                                      foreground=_STATUS_COLOR[pane.status])
                pane.stat_moves.config(text=f"{pane.env.total_moves}")
                pane.stat_seeds.config(text=f"{pane.env.seeds_collected}")
                pane.stat_entropy.config(text=f"{pane.env.entropy:.1f}")
                if pane.title_label is not None:
                    pane.title_label.config(text=f"{pane.label} - {_STATUS_TEXT[pane.status]}")
            except tk.TclError:
                return

    def _refresh_stats(self) -> None:
        self._update_stat_labels()
        # Tutte le copie concluse: le etichette restano congelate, niente altro timer.
        if self._started and all(pane.status == _DONE for pane in self._panes):
            return
        try:
            self._root.after(1000, self._refresh_stats)  # 1 Hz: non a decimi/millesimi
        except (RuntimeError, tk.TclError):
            pass

    # ------------------------------------------------------------------ #
    # Screenshot
    # ------------------------------------------------------------------ #

    def _take_screenshot(self) -> None:
        if not self._config.capture_screenshots:
            return
        try:
            self._toplevel.update_idletasks()
            x, y = self._toplevel.winfo_rootx(), self._toplevel.winfo_rooty()
            w, h = self._toplevel.winfo_width(), self._toplevel.winfo_height()
            self._board_images.append(ImageGrab.grab((x, y, x + w, y + h)))
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Impossibile catturare lo screenshot", exc_info=True)

    def _save_screenshots(self, datetime_str: str) -> None:
        if not self._config.capture_screenshots or not self._board_images:
            return
        try:
            directory = paths.SCREENSHOTS_DIR / datetime_str
            directory.mkdir(parents=True, exist_ok=True)
            for image in self._board_images:
                image.save(directory / f"{self._screenshot_number}.png")
                self._screenshot_number += 1
            self._board_images.clear()
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Impossibile salvare gli screenshot", exc_info=True)

    # ------------------------------------------------------------------ #
    # Loop di simulazione (un thread per copia)
    # ------------------------------------------------------------------ #

    def _start_next_pane(self) -> None:
        """Avvia la prossima copia **in coda**, o conclude se non ce ne sono piu'.

        Le due run non si sovrappongono: l'RL parte solo quando l'euristica ha
        finito. Cosi' il tempo di ciascuna copia e' misurato su una macchina
        scarica, senza contesa del GIL con l'altra griglia.
        Da chiamare solo sul thread principale.
        """
        if self._closing or self._stop_requested:
            return
        for pane in self._panes:
            if pane.status == _WAITING:
                pane.status = _RUNNING
                pane.start_time = time.monotonic()
                self._update_stat_labels()
                pane.worker = threading.Thread(
                    target=self._run_pane, args=(pane,), name=f"sim-{pane.mode}", daemon=True
                )
                pane.worker.start()
                return
        self._maybe_finish()

    def _pane_finished(self, pane: _Pane) -> None:
        """Chiusura di una copia (thread principale): congela le etichette e passa alla prossima."""
        pane.status = _DONE
        self._update_stat_labels()  # cronometro fermo sul valore finale, subito
        self._draw_pane(pane)
        self._start_next_pane()

    def _run_pane(self, pane: _Pane) -> None:
        cfg = self._config
        while not self._stop_requested:
            try:
                pane.env.tick(pane.policy)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("errore nel tick (%s)", pane.mode)
            pane.ticks_since_paint += 1
            if pane.ticks_since_paint >= self._refresh_rate:
                pane.ticks_since_paint = 0
                self._post(_EV_RENDER, pane)
            if self._config.capture_screenshots and pane.env.iterations % _SCREENSHOT_EVERY_N_ITERATIONS == 0:
                self._post(_EV_SCREENSHOT)
            if pane.env.is_done(cfg.stop_criterion, cfg.max_iterations, cfg.entropy_threshold):
                break
        pane.env.refresh_entropy()
        pane.duration_ms = int((time.monotonic() - (pane.start_time or time.monotonic())) * 1000)
        # il passaggio a _DONE e l'avvio della copia successiva avvengono sul thread principale
        self._post(_EV_FINISHED, pane)

    def _maybe_finish(self) -> None:
        if self._closing or self._finished:
            return
        if not all(pane.done for pane in self._panes):
            return
        self._finished = True
        self._take_screenshot()
        self._save_screenshots(self._formatted_datetime)
        results = [self._result_for(pane) for pane in self._panes]
        for result in results:
            self._append_result(result)
        SimulationResultsDialog(
            self._toplevel, results, self._handle_new_simulation,
            with_screenshots=self._config.capture_screenshots,
        )

    def _result_for(self, pane: _Pane) -> SimulationResult:
        cfg = self._config
        return SimulationResult(
            mode=pane.mode,
            variant_label=pane.label,
            duration_ms=pane.duration_ms,
            iterations=pane.env.iterations,
            rows=cfg.rows,
            cols=cfg.cols,
            n_thread=cfg.n_thread,
            n_ants=cfg.n_ants,
            n_seeds=cfg.n_seeds,
            total_moves=pane.env.total_moves,
            seeds_collected=pane.env.seeds_collected,
            initial_entropy=pane.env.initial_entropy,
            final_entropy=pane.env.entropy,
            master_seed=self._master_seed,
        )

    def _append_result(self, result: SimulationResult) -> None:
        try:
            paths.RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            write_header = not paths.RESULTS_FILE.exists() or paths.RESULTS_FILE.stat().st_size == 0
            with paths.RESULTS_FILE.open("a", encoding="utf-8") as handle:
                if write_header:
                    handle.write(_RESULTS_HEADER + "\n")
                handle.write(
                    ";".join(
                        [
                            self._formatted_datetime,
                            result.mode,
                            str(result.duration_ms),
                            str(result.iterations),
                            str(result.rows),
                            str(result.cols),
                            str(result.n_thread),
                            str(result.n_ants),
                            str(result.n_seeds),
                            str(result.total_moves),
                            str(result.seeds_collected),
                            f"{result.initial_entropy:.2f}",
                            f"{result.final_entropy:.2f}",
                            str(result.master_seed),
                            "1",
                        ]
                    )
                    + "\n"
                )
        except OSError:
            _LOGGER.exception("Impossibile scrivere %s", paths.RESULTS_FILE)

    # ------------------------------------------------------------------ #
    # Pulsanti
    # ------------------------------------------------------------------ #

    def _on_start_close(self) -> None:
        if not self._started:
            self._started = True
            self._formatted_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            self._start_close_btn.config(text="Close")
            self._start_next_pane()  # una copia alla volta, in ordine
        else:
            self._quit_now()

    def _on_toggle_visibility(self) -> None:
        self._gui_visible = not self._gui_visible
        for pane in self._panes:
            if self._gui_visible:
                self._draw_pane(pane)
            elif pane.canvas is not None:
                try:
                    pane.canvas.delete("all")
                except tk.TclError:
                    pass

    def _quit_now(self) -> None:
        self._closing = True
        self._stop_requested = True
        stamp = self._formatted_datetime or datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self._take_screenshot()
        self._save_screenshots(stamp)
        try:
            self._root.destroy()
        except tk.TclError:
            pass
        sys.exit(0)

    def _handle_new_simulation(self) -> None:
        self._stop_requested = True
        try:
            self._toplevel.destroy()
        except tk.TclError:
            pass
        self._on_new_simulation()
