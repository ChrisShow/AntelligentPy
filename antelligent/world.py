"""Stato iniziale condiviso dalle due copie gemelle (``docs/rl-design.md`` §1, §4.3).

Da un solo ``master_seed`` si genera un unico :class:`InitialState` (posizioni dei
semi, celle e heading di partenza delle formiche); poi ``Environment`` lo
materializza in due ambienti identici, uno per l'euristica e uno per l'RL.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import SimulationConfig
from .seeds.seed_type import SeedType

_DIRECTIONS = 4


@dataclass(frozen=True)
class InitialState:
    rows: int
    cols: int
    seed_types: int
    n_seeds: int
    n_ants: int
    master_seed: int
    seed_cells: tuple[tuple[int, int, int], ...]  # (x, y, codice tipo)
    ant_cells: tuple[tuple[int, int], ...]         # (x, y)
    ant_dirs: tuple[int, ...]                       # heading iniziale per formica


def resolve_master_seed(config_seed: int) -> int:
    """``0`` -> seed casuale (32 bit, > 0); altrimenti il valore dato."""
    if config_seed and config_seed > 0:
        return config_seed
    return random.Random().randrange(1, 2**31)


def build_initial_state(config: SimulationConfig, master_seed: int) -> InitialState:
    rng = random.Random(master_seed)
    cols, rows = config.cols, config.rows

    used_seed: set[tuple[int, int]] = set()
    seed_cells: list[tuple[int, int, int]] = []
    for _ in range(config.n_seeds):
        seed_type = SeedType.random(rng, config.seed_types)
        while True:
            cell = (rng.randrange(cols), rng.randrange(rows))
            if cell not in used_seed:
                used_seed.add(cell)
                seed_cells.append((cell[0], cell[1], seed_type.code))
                break

    used_ant: set[tuple[int, int]] = set()
    ant_cells: list[tuple[int, int]] = []
    for _ in range(config.n_ants):
        while True:
            cell = (rng.randrange(cols), rng.randrange(rows))
            if cell not in used_ant:
                used_ant.add(cell)
                ant_cells.append(cell)
                break
    ant_dirs = tuple(rng.randrange(_DIRECTIONS) for _ in range(config.n_ants))

    return InitialState(
        rows=rows,
        cols=cols,
        seed_types=config.seed_types,
        n_seeds=config.n_seeds,
        n_ants=config.n_ants,
        master_seed=master_seed,
        seed_cells=tuple(seed_cells),
        ant_cells=tuple(ant_cells),
        ant_dirs=ant_dirs,
    )
