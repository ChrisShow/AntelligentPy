"""Esito di una copia della simulazione (euristica o RL)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:
    mode: str                 # "heuristic" | "rl"
    variant_label: str
    duration_ms: int
    iterations: int
    rows: int
    cols: int
    n_thread: int
    n_ants: int
    n_seeds: int
    total_moves: int
    seeds_collected: int
    initial_entropy: float
    final_entropy: float
    master_seed: int
