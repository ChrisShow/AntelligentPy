"""GUI di confronto affiancato euristica vs RL (``docs/rl-design.md`` §4.5).

Una sola finestra con **due griglie gemelle** (stesso ``master_seed``), i pulsanti
Start/Visibility condivisi in basso e, sotto ciascuna griglia, il pannello
statistiche della rispettiva copia: tempo (aggiornato **1 volta al secondo**),
mosse totali, semi raccolti, entropia totale.

I due ambienti girano su **due thread worker indipendenti**; Tkinter viene toccato
solo dal thread principale via ``root.after``.
"""

from __future__ import annotations

import logging
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
        self.worker: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.duration_ms = 0
        self.done = False
        self.ticks_since_paint = 0


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
                policy = TabularQPolicy.load(rl_policy_path, heuristic=heuristic)
                policy.freeze()
                return policy, "RL (addestrata)"
            except Exception:  # noqa: BLE001
                _LOGGER.warning("impossibile caricare %s, uso apprendimento live", rl_policy_path, exc_info=True)
        # nessuna politica addestrata: il pannello RL impara dal vivo durante la run
        # (unico thread che tocca la tabella -> sicuro). Vedi docs/rl-design.md §4.5.
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
            ttk.Label(self._toplevel, text=pane.label, font=("TkDefaultFont", 11, "bold")).grid(
                row=0, column=col, pady=(8, 2)
            )
            pane.canvas = tk.Canvas(
                self._toplevel, width=board_w, height=board_h, background="white", highlightthickness=0
            )
            pane.canvas.grid(row=1, column=col, padx=6)
            pane.canvas._pane_label = pane.label  # type: ignore[attr-defined]

            stats = ttk.Frame(self._toplevel, padding=(6, 4))
            stats.grid(row=2, column=col, sticky="ew")
            pane.stat_time = self._add_stat(stats, 0, "tempo")
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

    def _schedule_render(self, pane: _Pane) -> None:
        if self._gui_visible:
            self._schedule(self._draw_pane, pane)

    def _schedule(self, func: Callable[..., None], *args: object) -> None:
        try:
            self._root.after(0, func, *args)
        except (RuntimeError, tk.TclError):
            pass

    def _refresh_stats(self) -> None:
        for pane in self._panes:
            if pane.stat_time is None:
                continue
            try:
                elapsed = int(time.monotonic() - pane.start_time) if pane.start_time else 0
                pane.stat_time.config(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")
                pane.stat_moves.config(text=f"{pane.env.total_moves}")
                pane.stat_seeds.config(text=f"{pane.env.seeds_collected}")
                pane.stat_entropy.config(text=f"{pane.env.entropy:.1f}")
            except tk.TclError:
                return
        try:
            self._root.after(1000, self._refresh_stats)  # 1 Hz: non a decimi/millesimi
        except (RuntimeError, tk.TclError):
            pass

    # ------------------------------------------------------------------ #
    # Screenshot
    # ------------------------------------------------------------------ #

    def _take_screenshot(self) -> None:
        try:
            self._toplevel.update_idletasks()
            x, y = self._toplevel.winfo_rootx(), self._toplevel.winfo_rooty()
            w, h = self._toplevel.winfo_width(), self._toplevel.winfo_height()
            self._board_images.append(ImageGrab.grab((x, y, x + w, y + h)))
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Impossibile catturare lo screenshot", exc_info=True)

    def _save_screenshots(self, datetime_str: str) -> None:
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

    def _run_pane(self, pane: _Pane) -> None:
        cfg = self._config
        pane.start_time = time.monotonic()
        while not self._stop_requested:
            try:
                pane.env.tick(pane.policy)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("errore nel tick (%s)", pane.mode)
            pane.ticks_since_paint += 1
            if pane.ticks_since_paint >= self._refresh_rate:
                pane.ticks_since_paint = 0
                self._schedule_render(pane)
            if pane.env.iterations % _SCREENSHOT_EVERY_N_ITERATIONS == 0:
                self._schedule(self._take_screenshot)
            if pane.env.is_done(cfg.stop_criterion, cfg.max_iterations, cfg.entropy_threshold):
                break
        pane.env.refresh_entropy()
        pane.duration_ms = int((time.monotonic() - (pane.start_time or time.monotonic())) * 1000)
        pane.done = True
        self._schedule(self._draw_pane, pane)
        self._schedule(self._maybe_finish)

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
        SimulationResultsDialog(self._toplevel, results, self._handle_new_simulation)

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
            for pane in self._panes:
                pane.worker = threading.Thread(
                    target=self._run_pane, args=(pane,), name=f"sim-{pane.mode}", daemon=True
                )
                pane.worker.start()
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
