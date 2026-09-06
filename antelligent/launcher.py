"""Launcher di configurazione pre-simulazione.

Raccoglie i parametri della run (griglia, formiche, semi, criterio di arresto,
seed iniziale) e li salva in ``config.properties``. Il controllo
``nAnts <= righe*colonne`` garantisce che ogni formica possa occupare in
esclusiva una cella.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from .config import SimulationConfig

_MATRIX_TYPE_LOCK = 1


class AntelligentLauncher(tk.Toplevel):
    def __init__(
        self,
        root: tk.Tk,
        defaults: SimulationConfig,
        on_launch: Callable[[SimulationConfig], None],
    ) -> None:
        super().__init__(root)
        self.title("Antelligent - Configurazione")
        self.resizable(False, False)
        self._on_launch = on_launch
        self.protocol("WM_DELETE_WINDOW", root.destroy)

        content = ttk.Frame(self, padding=12)
        content.grid(sticky="nsew")

        form = ttk.Frame(content)
        form.grid(row=0, column=0, sticky="ew")

        self._cols = self._add_spinbox(form, 0, "Colonne:", _clamp(defaults.cols, 1, 300), 1, 300)
        self._rows = self._add_spinbox(form, 1, "Righe:", _clamp(defaults.rows, 1, 300), 1, 300)
        self._n_thread = self._add_spinbox(form, 2, "Numero di thread:", _clamp(defaults.n_thread, 1, 64), 1, 64)
        self._n_ants = self._add_spinbox(form, 3, "Numero di formiche:", _clamp(defaults.n_ants, 1, 10_000), 1, 10_000)
        self._n_seeds = self._add_spinbox(form, 4, "Numero di semi:", _clamp(defaults.n_seeds, 1, 50_000), 1, 50_000)
        self._seed_types = self._add_spinbox(form, 5, "Tipi di seme (1-5):", _clamp(defaults.seed_types, 1, 5), 1, 5)
        self._refresh_rate = self._add_spinbox(
            form, 6, "Refresh GUI (iterazioni):", _clamp(defaults.refresh_rate, 1, 1_000_000), 1, 1_000_000
        )
        self._master_seed = self._add_spinbox(
            form, 7, "Seed iniziale (0 = casuale):", max(0, defaults.master_seed), 0, 2_147_483_646
        )

        # --- Criterio di arresto ---
        stop_frame = ttk.LabelFrame(content, text="Criterio di arresto", padding=8)
        stop_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self._stop_var = tk.IntVar(value=0 if defaults.stop_criterion != 1 else 1)
        self._max_iterations_var = tk.StringVar(value=str(defaults.max_iterations))
        self._entropy_threshold_var = tk.StringVar(value=str(defaults.entropy_threshold))

        iterations_radio = ttk.Radiobutton(
            stop_frame, text="Ferma dopo un numero di iterazioni:", variable=self._stop_var,
            value=0, command=self._update_stop_fields,
        )
        iterations_radio.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self._max_iterations_entry = ttk.Entry(stop_frame, textvariable=self._max_iterations_var, width=12)
        self._max_iterations_entry.grid(row=0, column=1, sticky="ew", pady=4)

        entropy_radio = ttk.Radiobutton(
            stop_frame, text="Ferma al raggiungimento di una soglia di entropia:",
            variable=self._stop_var, value=1, command=self._update_stop_fields,
        )
        entropy_radio.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self._entropy_threshold_entry = ttk.Entry(stop_frame, textvariable=self._entropy_threshold_var, width=12)
        self._entropy_threshold_entry.grid(row=1, column=1, sticky="ew", pady=4)

        self._update_stop_fields()

        # --- Opzioni della run ---
        options_frame = ttk.LabelFrame(content, text="Opzioni", padding=8)
        options_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self._capture_var = tk.BooleanVar(value=bool(defaults.capture_screenshots))
        ttk.Checkbutton(
            options_frame,
            text="Cattura schermate della finestra (results/screenshots/)",
            variable=self._capture_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            options_frame,
            text="Rallenta la simulazione; su macOS serve il permesso \"Registrazione schermo\".",
            foreground="gray40",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        start_button = ttk.Button(content, text="Avvia simulazione", command=self._on_start)
        start_button.grid(row=3, column=0, sticky="e", pady=(12, 0))
        self.bind("<Return>", lambda _event: self._on_start())

        self.update_idletasks()
        self._center(root)

    # ------------------------------------------------------------------ #

    def _add_spinbox(
        self, parent: ttk.Frame, row: int, label: str, value: int, minimum: int, maximum: int
    ) -> tk.IntVar:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.IntVar(value=value)
        spin = ttk.Spinbox(parent, from_=minimum, to=maximum, textvariable=var, width=12)
        spin.grid(row=row, column=1, sticky="ew", pady=4)
        return var

    def _update_stop_fields(self) -> None:
        on_iterations = self._stop_var.get() == 0
        self._max_iterations_entry.config(state="normal" if on_iterations else "disabled")
        self._entropy_threshold_entry.config(state="disabled" if on_iterations else "normal")

    def _center(self, root: tk.Tk) -> None:
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

    def _on_start(self) -> None:
        stop_on_iterations = self._stop_var.get() == 0

        max_iterations = _parse_int(self._max_iterations_var.get())
        if stop_on_iterations and (max_iterations is None or max_iterations <= 0):
            messagebox.showerror(
                "Valore non valido",
                "Inserisci un numero massimo di iterazioni valido (intero positivo).",
                parent=self,
            )
            return

        entropy_threshold = _parse_float(self._entropy_threshold_var.get())
        if not stop_on_iterations and (
            entropy_threshold is None or entropy_threshold < 0 or entropy_threshold > 100
        ):
            messagebox.showerror(
                "Valore non valido",
                "Inserisci una soglia di entropia valida (numero tra 0 e 100).",
                parent=self,
            )
            return

        try:
            cols = int(self._cols.get())
            rows = int(self._rows.get())
            n_seeds = int(self._n_seeds.get())
            n_ants = int(self._n_ants.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Valore non valido", "Controlla i campi numerici.", parent=self)
            return

        cells = cols * rows
        if n_seeds > cells:
            messagebox.showerror(
                "Valore non valido",
                f"Il numero di semi ({n_seeds}) supera le celle disponibili "
                f"({cols}x{rows} = {cells}):\nla simulazione resterebbe bloccata a cercare "
                "una cella libera che non esiste.\nRiduci i semi o aumenta la griglia.",
                parent=self,
            )
            return
        if n_ants > cells:
            messagebox.showerror(
                "Valore non valido",
                f"Il numero di formiche ({n_ants}) supera le celle disponibili "
                f"({cols}x{rows} = {cells}):\nogni formica occupa in esclusiva una cella.\n"
                "Riduci le formiche o aumenta la griglia.",
                parent=self,
            )
            return

        config = SimulationConfig(
            cols=cols,
            rows=rows,
            n_thread=int(self._n_thread.get()),
            n_ants=n_ants,
            n_seeds=n_seeds,
            seed_types=int(self._seed_types.get()),
            matrix_type=_MATRIX_TYPE_LOCK,
            refresh_rate=int(self._refresh_rate.get()),
            stop_criterion=0 if stop_on_iterations else 1,
            max_iterations=max_iterations if max_iterations is not None else 20_000,
            entropy_threshold=entropy_threshold if entropy_threshold is not None else 5.0,
            master_seed=int(self._master_seed.get()),
            capture_screenshots=bool(self._capture_var.get()),
        )
        self.destroy()
        self._on_launch(config)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _parse_int(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return None
