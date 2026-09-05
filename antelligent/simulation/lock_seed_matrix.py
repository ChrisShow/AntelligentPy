"""Matrice "normale" (array 2D) con un lock per cella.

Il lock di ogni cella funge da lock di *occupazione*: la formica che si trova su
una cella ne tiene il lock finche' non si sposta, per cui al massimo una formica
per cella. Lo spostamento usa :meth:`try_acquire_cell` (acquisizione non
bloccante): la formica che non ottiene la cella non si blocca e non perde la
mossa, ne sceglie un'altra.

Si usa ``threading.Lock`` (non ``RLock``): il lock di una cella viene acquisito e
poi rilasciato potenzialmente da thread diversi del pool tra un'iterazione e la
successiva, cosa che ``RLock`` (con proprieta' per-thread) non permetterebbe.
``threading.Lock`` non espone un'acquisizione equa (fair).
"""

from __future__ import annotations

import threading
from typing import Optional

from ..seeds.seed import Seed
from .seed_matrix import SeedMatrix


class LockSeedMatrix(SeedMatrix):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__(rows, cols)
        # Matrice normale riga-per-riga: _seeds[y][x] (indicizzata [riga][colonna]).
        self._seeds: list[list[Optional[Seed]]] = [[None] * cols for _ in range(rows)]
        self._locks: list[list[threading.Lock]] = [
            [threading.Lock() for _ in range(cols)] for _ in range(rows)
        ]

    def get_seed_at(self, pos_x: int, pos_y: int) -> Optional[Seed]:
        return self._seeds[pos_y][pos_x]

    def set_seed_at(self, pos_x: int, pos_y: int, seed: Seed) -> None:
        self._seeds[pos_y][pos_x] = seed

    def clear_seed_at(self, pos_x: int, pos_y: int) -> None:
        self._seeds[pos_y][pos_x] = None

    def try_acquire_cell(self, pos_x: int, pos_y: int) -> bool:
        return self._locks[pos_y][pos_x].acquire(blocking=False)

    def acquire_cell(self, pos_x: int, pos_y: int) -> None:
        self._locks[pos_y][pos_x].acquire()

    def release_cell(self, pos_x: int, pos_y: int) -> None:
        self._locks[pos_y][pos_x].release()
