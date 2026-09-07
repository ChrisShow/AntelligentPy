"""Politica euristica pre-impostata (baseline A0).

Applica l'algoritmo probabilistico pre-impostato: probabilita' di pick/drop di
Lumer-Faieta `(kp/(kp+f))^2`, `(f/(kd+f))^2` (via
:meth:`SeedMatrix.pick_probability` / :meth:`~SeedMatrix.drop_probability`) e
random walk con momentum (via :func:`check_move.random_move_index`). Nessun
apprendimento.
"""

from __future__ import annotations

import random
from typing import Optional

from ..actions import Action, Manipulation, Observation
from ..simulation.ant import Ant
from ..simulation.seed_matrix import SeedMatrix
from ..utilities import check_move
from .base import AntPolicyBase


class HeuristicPolicy(AntPolicyBase):
    def __init__(self, matrix: SeedMatrix, seed_types: int, rng: Optional[random.Random] = None) -> None:
        self._matrix = matrix
        self._seed_types = seed_types
        self._rng = rng or random.Random()

    def select_action(self, obs: Observation, ant: Ant) -> Action:
        x, y = ant.position.x, ant.position.y
        manipulation = Manipulation.NOOP

        if obs.can_pick:
            seed = self._matrix.get_seed_at(x, y)
            if seed is not None:
                p = self._matrix.pick_probability(x, y, seed.code, self._seed_types)
                if p > 0 and (self._rng.random() * 100 + 1) <= p:
                    manipulation = Manipulation.PICK
        elif obs.can_drop and ant.carried_seed is not None:
            p = self._matrix.drop_probability(x, y, ant.carried_seed.code, self._seed_types)
            if p > 0 and (self._rng.random() * 100 + 1) <= p:
                manipulation = Manipulation.DROP

        move_index = check_move.random_move_index(
            x, y, self._matrix.rows, self._matrix.cols, ant.direction, self._rng
        )
        return Action(manipulation, move_index)
