"""Schermata dei risultati finali (confronto affiancato), ``docs/rl-design.md`` §4.5."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

from .simulation_result import SimulationResult


class SimulationResultsDialog(tk.Toplevel):
    def __init__(
        self,
        owner: tk.Misc,
        results: Sequence[SimulationResult],
        on_new_simulation: Callable[[], None],
        *,
        with_screenshots: bool = False,
    ) -> None:
        super().__init__(owner)
        self.title("Simulazione completata")
        self.resizable(False, False)
        self.transient(owner.winfo_toplevel())

        container = ttk.Frame(self, padding=16)
        container.grid(sticky="nsew")

        ttk.Label(
            container, text="Simulazione completata", font=("TkDefaultFont", 14, "bold")
        ).grid(row=0, column=0, columnspan=1 + len(results), pady=(0, 12))

        master_seed = results[0].master_seed if results else 0
        rows: list[tuple[str, Callable[[SimulationResult], str]]] = [
            ("Modalita'", lambda r: r.variant_label),
            ("Durata", lambda r: f"{r.duration_ms} ms"),
            ("Iterazioni", lambda r: str(r.iterations)),
            ("Mosse totali", lambda r: str(r.total_moves)),
            ("Semi raccolti", lambda r: str(r.seeds_collected)),
            ("Entropia iniziale", lambda r: f"{r.initial_entropy:.2f}"),
            ("Entropia finale", lambda r: f"{r.final_entropy:.2f}"),
        ]

        ttk.Label(container, text=f"seed iniziale: {master_seed}", font=("TkDefaultFont", 9)).grid(
            row=1, column=0, columnspan=1 + len(results), sticky="w", pady=(0, 6)
        )
        for col, result in enumerate(results, start=1):
            ttk.Label(container, text=result.mode.upper(), font=("TkDefaultFont", 10, "bold")).grid(
                row=2, column=col, padx=8
            )
        for i, (label, getter) in enumerate(rows, start=3):
            ttk.Label(container, text=label).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=2)
            for col, result in enumerate(results, start=1):
                ttk.Label(container, text=getter(result)).grid(row=i, column=col, padx=8, pady=2)

        note_text = "Risultati salvati in results/results.txt"
        if with_screenshots:
            note_text += ", screenshot in results/screenshots/"
        note = ttk.Label(container, text=note_text, font=("TkDefaultFont", 8, "italic"))
        note.grid(row=3 + len(rows), column=0, columnspan=1 + len(results), pady=(12, 8))

        buttons = ttk.Frame(container)
        buttons.grid(row=4 + len(rows), column=0, columnspan=1 + len(results))

        def new_run() -> None:
            self.destroy()
            on_new_simulation()

        new_button = ttk.Button(buttons, text="Nuova iterazione", command=new_run)
        new_button.grid(row=0, column=0, padx=6)
        ttk.Button(buttons, text="Esci", command=lambda: sys.exit(0)).grid(row=0, column=1, padx=6)

        self.protocol("WM_DELETE_WINDOW", lambda: None)
        new_button.focus_set()
        self.bind("<Return>", lambda _event: new_run())
        self.update_idletasks()
        self.grab_set()
