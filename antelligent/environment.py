"""Ambiente di simulazione, comune a euristica e RL (``docs/rl-design.md`` §3.4, §4.2).

Un :class:`Environment` possiede una matrice, le formiche e i semi di **una**
copia del campo. Espone:

- :meth:`observe` — osservazione locale e parziale di una formica;
- :meth:`step` — applica la manipolazione + la mossa di una formica (fisica di
  occupazione con ``try_acquire`` e ripiego sulla contesa, identici per euristica
  e RL), aggiorna i contatori, calcola la ricompensa;
- :meth:`tick` — un tick completo: tutte le formiche agiscono una volta in ordine
  casuale, delegando la decisione a una politica.

La politica NON e' un attributo dell'ambiente: viene passata a :meth:`tick`, cosi'
lo stesso ambiente serve sia l'euristica sia l'RL.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .actions import STAY, Action, Manipulation, Observation, StepResult, Transition
from .seeds.seed import Seed
from .seeds.seed_type import SeedType
from .simulation.ant import Ant
from .simulation.lock_seed_matrix import LockSeedMatrix
from .utilities import check_move
from .utilities.position import Position
from .world import InitialState

if TYPE_CHECKING:
    from .policies.base import AntPolicy

_LOGGER = logging.getLogger(__name__)

# Codici cella per la finestra categorica dell'osservazione.
_CELL_EMPTY = 0
_CELL_ANT = -1
_CELL_OUT = -2


@dataclass
class RewardConfig:
    """Pesi della ricompensa locale (``docs/rl-design.md`` §3.3). L'euristica li ignora.

    La manipolazione e' premiata *rispetto a* ``pivot``: raccogliere un seme
    isolato dai suoi simili (``f_own < pivot``) e posare un seme tra i suoi simili
    (``f_own > pivot``) danno ricompensa positiva; il contrario e' penalizzato,
    cosi' il "churn" (raccogli-e-riposa a caso) non conviene.
    """

    step_penalty: float = 0.01
    contested_penalty: float = 0.05
    invalid_penalty: float = 0.05
    manip_scale: float = 1.0
    pivot: float = 0.5
    final_scale: float = 0.0  # bonus terminale alpha*(H0 - H_final): lo aggiunge il runner


class Environment:
    STAY = STAY

    def __init__(
        self,
        matrix: LockSeedMatrix,
        ants: list[Ant],
        seeds: list[Seed],
        seed_types: int,
        *,
        reward: Optional[RewardConfig] = None,
        rng: Optional[random.Random] = None,
        window_radius: int = 2,
        entropy_every: int = 1,
    ) -> None:
        self.matrix = matrix
        self.ants = ants
        self.seeds = seeds
        self.seed_types = seed_types
        self.reward_cfg = reward or RewardConfig()
        self.rng = rng or random.Random()
        self.window_radius = window_radius
        self.entropy_every = max(1, entropy_every)

        self.total_moves = 0
        self.seeds_collected = 0
        self.iterations = 0

        self._ant_cells: set[tuple[int, int]] = {(a.position.x, a.position.y) for a in ants}
        self.initial_entropy = matrix.check_entropy_placed(seed_types)
        self._entropy = self.initial_entropy

    # ------------------------------------------------------------------ #
    # Costruzione
    # ------------------------------------------------------------------ #

    @classmethod
    def from_initial_state(
        cls,
        init: InitialState,
        *,
        reward: Optional[RewardConfig] = None,
        rng: Optional[random.Random] = None,
        window_radius: int = 2,
        entropy_every: int = 1,
    ) -> "Environment":
        matrix = LockSeedMatrix(init.rows, init.cols)
        seeds: list[Seed] = []
        for (x, y, code) in init.seed_cells:
            seed = Seed(SeedType.from_code(code))
            matrix.set_seed_at(x, y, seed)
            seed.set_position(x, y)
            seeds.append(seed)
        ants: list[Ant] = []
        for code, ((x, y), direction) in enumerate(zip(init.ant_cells, init.ant_dirs)):
            matrix.acquire_cell(x, y)  # cella distinta su matrice fresca: sempre ottenuta
            ants.append(Ant(code, Position(x, y), direction))
        return cls(
            matrix, ants, seeds, init.seed_types,
            reward=reward, rng=rng, window_radius=window_radius, entropy_every=entropy_every,
        )

    # ------------------------------------------------------------------ #
    # Osservazione
    # ------------------------------------------------------------------ #

    def observe(self, ant: Ant) -> Observation:
        x, y = ant.position.x, ant.position.y
        carrying = ant.carried_seed is not None
        carried_type = ant.carried_seed.code if carrying else 0
        seed_here = self.matrix.has_seed(x, y)
        counts = self.matrix.count_types_around(x, y, self.seed_types)
        total = sum(counts) or 1
        f_here = tuple(c / total for c in counts)

        neigh = check_move.check_around(x, y, self.matrix.rows, self.matrix.cols, ant.direction)
        blocked = tuple(
            (p is None) or ((p.x, p.y) in self._ant_cells) for p in neigh
        )

        best_dir = STAY
        if carrying:
            # proxy a basso costo: prima cella adiacente che contiene gia' un seme
            # del tipo trasportato (8 letture, non 8 scansioni di vicinato).
            for idx, p in enumerate(neigh):
                if p is None:
                    continue
                seed = self.matrix.get_seed_at(p.x, p.y)
                if seed is not None and seed.code == carried_type:
                    best_dir = idx
                    break

        return Observation(
            carrying=carrying,
            carried_type=carried_type,
            seed_here=seed_here,
            can_pick=(not carrying) and seed_here,
            can_drop=carrying and not seed_here,
            f_here=f_here,
            blocked=blocked,
            best_dir=best_dir,
            direction=ant.direction,
            window=self._window(x, y),
        )

    def _window(self, cx: int, cy: int) -> tuple[tuple[int, ...], ...]:
        r = self.window_radius
        if r <= 0:  # disabilitato (Fase 1 tabellare non usa la finestra)
            return ()
        rows: list[tuple[int, ...]] = []
        for dy in range(-r, r + 1):
            line: list[int] = []
            for dx in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if not (0 <= x < self.matrix.cols and 0 <= y < self.matrix.rows):
                    line.append(_CELL_OUT)
                elif (x, y) in self._ant_cells and (dx != 0 or dy != 0):
                    line.append(_CELL_ANT)
                elif self.matrix.has_seed(x, y):
                    seed = self.matrix.get_seed_at(x, y)
                    line.append(seed.code + 1 if seed is not None else _CELL_EMPTY)
                else:
                    line.append(_CELL_EMPTY)
            rows.append(tuple(line))
        return tuple(rows)

    # ------------------------------------------------------------------ #
    # Passo
    # ------------------------------------------------------------------ #

    def step(self, ant: Ant, action: Action) -> StepResult:
        x0, y0 = ant.position.x, ant.position.y
        info: dict[str, object] = {}

        # --- relevanza per la ricompensa (calcolata PRIMA di mutare la cella) ---
        f_relevant = 0.0
        if action.manipulation == Manipulation.PICK and ant.carried_seed is None and self.matrix.has_seed(x0, y0):
            code = self.matrix.get_seed_at(x0, y0).code
            counts = self.matrix.count_types_around(x0, y0, self.seed_types)
            f_relevant = counts[code] / (sum(counts) or 1)
        elif action.manipulation == Manipulation.DROP and ant.carried_seed is not None and not self.matrix.has_seed(x0, y0):
            code = ant.carried_seed.code
            counts = self.matrix.count_types_around(x0, y0, self.seed_types)
            f_relevant = counts[code] / (sum(counts) or 1)

        # --- manipolazione ---
        manip_attempted = action.manipulation != Manipulation.NOOP
        manip_success = False
        if action.manipulation == Manipulation.PICK and ant.carried_seed is None and self.matrix.has_seed(x0, y0):
            seed = self.matrix.commit_pick(x0, y0)
            if seed is not None:
                ant.carried_seed = seed
                self.seeds_collected += 1
                manip_success = True
        elif action.manipulation == Manipulation.DROP and ant.carried_seed is not None and not self.matrix.has_seed(x0, y0):
            if self.matrix.commit_drop(x0, y0, ant.carried_seed):
                ant.carried_seed = None
                manip_success = True

        # --- movimento ---
        moved, contested = self._apply_move(ant, action.move_index)

        info.update(
            manip_attempted=manip_attempted,
            manip_success=manip_success,
            moved=moved,
            contested=contested,
            f_relevant=f_relevant,
        )
        reward = self._reward(action, info)
        return StepResult(reward=reward, done=False, info=info)

    def _apply_move(self, ant: Ant, move_index: int) -> tuple[bool, bool]:
        """Applica la mossa proposta. Ritorna ``(spostata, contesa_sul_primo_bersaglio)``.

        ``try_acquire`` non bloccante sulla cella scelta; se occupata, ripiego su
        un'altra adiacente libera; se tutte occupate, resta ferma (senza
        "perdere" il tick).
        """
        x, y = ant.position.x, ant.position.y
        d = ant.direction
        rows, cols = self.matrix.rows, self.matrix.cols

        if move_index == STAY:
            return (False, False)

        neigh = check_move.check_around(x, y, rows, cols, d)
        target = neigh[move_index]
        if target is None:  # scelta fuori griglia (possibile solo per l'RL) -> resta ferma
            return (False, False)
        new_dir = check_move.resulting_direction(move_index, d)

        if self._occupy(target.x, target.y, x, y):
            ant.position = target
            ant.direction = new_dir
            self.total_moves += 1
            return (True, False)

        candidates = check_move.neighbours_with_directions(x, y, rows, cols, d)
        self.rng.shuffle(candidates)
        for pos, direction in candidates:
            if pos == target:
                continue
            if self._occupy(pos.x, pos.y, x, y):
                ant.position = pos
                ant.direction = direction
                self.total_moves += 1
                return (True, True)
        return (False, True)

    def _occupy(self, nx: int, ny: int, ox: int, oy: int) -> bool:
        if not self.matrix.try_acquire_cell(nx, ny):
            return False
        self.matrix.release_cell(ox, oy)
        self._ant_cells.discard((ox, oy))
        self._ant_cells.add((nx, ny))
        return True

    def _reward(self, action: Action, info: dict[str, object]) -> float:
        cfg = self.reward_cfg
        reward = -cfg.step_penalty
        if info["contested"]:
            reward -= cfg.contested_penalty
        if info["manip_attempted"] and not info["manip_success"]:
            reward -= cfg.invalid_penalty
        if info["manip_success"]:
            f_rel = float(info["f_relevant"])
            if action.manipulation == Manipulation.PICK:
                # bene se il seme era isolato dai suoi simili (f_own basso)
                reward += cfg.manip_scale * (cfg.pivot - f_rel)
            else:  # DROP: bene se posato tra i suoi simili (f_own alto)
                reward += cfg.manip_scale * (f_rel - cfg.pivot)
        return reward

    # ------------------------------------------------------------------ #
    # Tick e arresto
    # ------------------------------------------------------------------ #

    def tick(self, policy: "AntPolicy") -> None:
        order = list(self.ants)
        self.rng.shuffle(order)
        for ant in order:
            obs = self.observe(ant)
            action = policy.select_action(obs, ant)
            result = self.step(ant, action)
            next_obs = self.observe(ant)
            policy.record(Transition(obs, action, result.reward, next_obs, result.done, result.info))
        self.iterations += 1
        if self.iterations % self.entropy_every == 0:
            self._entropy = self.matrix.check_entropy_placed(self.seed_types)

    @property
    def entropy(self) -> float:
        """Ultimo valore di entropia media calcolato (sui semi posati, sempre definito)."""
        return self._entropy

    def refresh_entropy(self) -> float:
        self._entropy = self.matrix.check_entropy_placed(self.seed_types)
        return self._entropy

    @property
    def carried_count(self) -> int:
        return sum(1 for a in self.ants if a.carried_seed is not None)

    def is_done(self, stop_criterion: int, max_iterations: int, entropy_threshold: float) -> bool:
        if stop_criterion == 0:
            return self.iterations >= max_iterations
        return self.carried_count == 0 and self._entropy <= entropy_threshold
