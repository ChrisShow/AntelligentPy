"""Q-learning tabellare a parametri condivisi (Fase 1, ``docs/rl-design.md`` §5.1).

Una sola politica per tutte le formiche. Q fattorizzata in due tabelle:

- ``q_move``  : stato_movimento    -> valori delle 9 mosse (8 direzioni + fermo);
- ``q_manip`` : stato_manipolazione -> valori di {no-op, pick/drop}.

Ogni componente puo' essere *appresa* o delegata all'euristica, per realizzare le
ablazioni A1 (solo movimento RL) / A2 (solo manipolazione RL) / A3 (entrambe).
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from typing import Optional

from ..actions import STAY, Action, Manipulation, Observation, Transition
from ..simulation.ant import Ant
from .base import AntPolicyBase
from .heuristic import HeuristicPolicy

_N_MOVES = 9  # 8 direzioni relative + STAY


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 200
    learn_move: bool = True
    learn_manip: bool = True
    f_bins: int = 4


class TabularQPolicy(AntPolicyBase):
    def __init__(
        self,
        seed_types: int,
        config: Optional[QConfig] = None,
        *,
        heuristic: Optional[HeuristicPolicy] = None,
    ) -> None:
        self.seed_types = seed_types
        self.config = config or QConfig()
        self.q_move: dict[tuple, list[float]] = {}
        self.q_manip: dict[tuple, list[float]] = {}
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        self.frozen = False
        self.rng = random.Random()
        self._heuristic = heuristic
        # (stato_move, indice_move, stato_manip | None, indice_manip)
        self._pending: Optional[tuple] = None

    @property
    def _needs_heuristic(self) -> bool:
        return not (self.config.learn_move and self.config.learn_manip)

    def bind_environment(self, env: object) -> None:
        """Lega un fallback euristico all'ambiente corrente (serve solo per A1/A2)."""
        self._heuristic = HeuristicPolicy(env.matrix, self.seed_types)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Discretizzazione dello stato
    # ------------------------------------------------------------------ #

    def _move_state(self, obs: Observation) -> tuple:
        return (
            int(obs.carrying),
            obs.carried_type if obs.carrying else 0,
            obs.best_dir,
            obs.direction,
            tuple(int(b) for b in obs.blocked),
        )

    def _manip_state(self, obs: Observation) -> tuple:
        relevant = obs.carried_type if obs.carrying else 0
        bins = self.config.f_bins
        f_bin = min(bins - 1, int(obs.f_here[relevant] * bins))
        return (int(obs.carrying), relevant, f_bin)

    # ------------------------------------------------------------------ #
    # Selezione dell'azione
    # ------------------------------------------------------------------ #

    def _row(self, table: dict[tuple, list[float]], state: tuple, n: int) -> list[float]:
        row = table.get(state)
        if row is None:
            row = [0.0] * n
            table[state] = row
        return row

    def _argmax(self, values: list[float]) -> int:
        best_index, best_value = 0, float("-inf")
        for index, value in enumerate(values):
            if value > best_value:
                best_value, best_index = value, index
        return best_index

    def select_action(self, obs: Observation, ant: Ant) -> Action:
        if self._needs_heuristic and self._heuristic is None:
            raise RuntimeError(
                "ablazione A1/A2 senza euristica di fallback: passa heuristic= o chiama bind_environment()"
            )
        fallback = self._heuristic.select_action(obs, ant) if self._heuristic is not None else None
        exploring = (not self.frozen) and self.rng.random() < self.epsilon

        # --- movimento ---
        move_state: Optional[tuple] = None
        if self.config.learn_move:
            move_state = self._move_state(obs)
            row = self._row(self.q_move, move_state, _N_MOVES)
            move_index = self.rng.randrange(_N_MOVES) if exploring else self._argmax(row)
        else:
            move_index = fallback.move_index  # type: ignore[union-attr]

        # --- manipolazione ---
        manip_state: Optional[tuple] = None
        manip_index = 0
        manipulation = Manipulation.NOOP
        if obs.can_pick or obs.can_drop:
            if self.config.learn_manip:
                manip_state = self._manip_state(obs)
                row = self._row(self.q_manip, manip_state, 2)
                manip_index = self.rng.randrange(2) if exploring else self._argmax(row)
            else:
                manip_index = 1 if fallback.manipulation != Manipulation.NOOP else 0  # type: ignore[union-attr]
            if manip_index == 1:
                manipulation = Manipulation.PICK if obs.can_pick else Manipulation.DROP

        self._pending = (move_state, move_index, manip_state, manip_index)
        return Action(manipulation, move_index)

    # ------------------------------------------------------------------ #
    # Apprendimento
    # ------------------------------------------------------------------ #

    def record(self, transition: Transition) -> None:
        if self.frozen or self._pending is None:
            return
        move_state, move_index, manip_state, manip_index = self._pending
        self._pending = None
        reward = transition.reward
        gamma = self.config.gamma
        alpha = self.config.alpha
        keep = 0.0 if transition.done else 1.0

        if move_state is not None:
            next_row = self._row(self.q_move, self._move_state(transition.next_obs), _N_MOVES)
            target = reward + gamma * max(next_row) * keep
            row = self._row(self.q_move, move_state, _N_MOVES)
            row[move_index] += alpha * (target - row[move_index])

        if manip_state is not None:
            next_obs = transition.next_obs
            if next_obs.can_pick or next_obs.can_drop:
                next_value = max(self._row(self.q_manip, self._manip_state(next_obs), 2))
            else:
                next_value = 0.0
            target = reward + gamma * next_value * keep
            row = self._row(self.q_manip, manip_state, 2)
            row[manip_index] += alpha * (target - row[manip_index])

    def end_episode(self) -> None:
        self.episode += 1
        frac = min(1.0, self.episode / max(1, self.config.epsilon_decay_episodes))
        self.epsilon = self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def freeze(self) -> None:
        """Congela la politica per l'inferenza (GUI / benchmark): niente esplorazione, niente update."""
        self.frozen = True
        self.epsilon = 0.0

    # ------------------------------------------------------------------ #
    # Persistenza
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        payload = {
            "seed_types": self.seed_types,
            "config": self.config,
            "q_move": self.q_move,
            "q_manip": self.q_manip,
            "episode": self.episode,
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path, *, heuristic: Optional[HeuristicPolicy] = None) -> "TabularQPolicy":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        policy = cls(payload["seed_types"], payload["config"], heuristic=heuristic)
        policy.q_move = payload["q_move"]
        policy.q_manip = payload["q_manip"]
        policy.episode = payload.get("episode", 0)
        policy.epsilon = payload.get("epsilon", policy.config.epsilon_end)
        return policy
