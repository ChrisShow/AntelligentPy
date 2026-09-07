"""Q-learning tabellare a parametri condivisi (Fase 1).

Una sola politica per tutte le formiche. Q fattorizzata in due tabelle:

- ``q_move``  : stato_movimento    -> valori delle 9 mosse (8 direzioni + fermo);
- ``q_manip`` : stato_manipolazione -> valori di {no-op, pick/drop}.

Ogni componente puo' essere *appresa* o delegata all'euristica, per realizzare le
ablazioni A1 (solo movimento RL) / A2 (solo manipolazione RL) / A3 (entrambe).
"""

from __future__ import annotations

import logging
import pickle
import random
from dataclasses import dataclass
from typing import Optional

from ..actions import STAY, Action, Manipulation, Observation, Transition
from ..simulation.ant import Ant
from .base import AntPolicyBase
from .heuristic import HeuristicPolicy

_LOGGER = logging.getLogger(__name__)

_N_MOVES = 9  # 8 direzioni relative + STAY

#: Versione della codifica dello stato. Va incrementata a ogni cambio di
#: ``_move_state`` / ``_manip_state``: una tabella salvata con una codifica diversa
#: non e' riutilizzabile (tutte le letture cadrebbero su righe vuote) e va
#: riaddestrata. :meth:`TabularQPolicy.load` lo segnala invece di degradare in
#: silenzio.
_STATE_VERSION = 3


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
    #: Se ``False`` (default) il "cervello" non puo' scegliere di **restare fermo**:
    #: sceglie fra le 8 direzioni relative, come fa l'euristica
    #: (:func:`check_move.random_move_index` non restituisce mai ``STAY`` su una
    #: griglia normale). Con ``STAY`` disponibile la Q-learning cade in un punto
    #: fisso assorbente: restare fermi non rischia mai ``contested_penalty``,
    #: quindi e' l'azione meno costosa ovunque non ci sia una ricompensa positiva
    #: raggiungibile, e ``Q(s,STAY) = -step_penalty/(1-gamma)`` si auto-sostiene.
    #: La fisica gestisce comunque il "non riesco a muovermi" (contesa -> ripiego
    #: -> resta ferma), quindi l'azione esplicita non serve.
    allow_stay: bool = False
    #: Rumore residuo in **inferenza** (politica congelata, GUI/benchmark).
    #: L'osservazione e' parziale e fortemente aliasata: stati diversi del mondo
    #: appaiono identici alla formica. Una politica *deterministica* su stati
    #: aliasati cade in cicli limite (avanti-e-indietro fra le stesse celle) da cui
    #: non puo' uscire, perche' la scelta dipende solo dall'osservazione. In un
    #: POMDP la politica ottima e' in generale **stocastica**; anche l'euristica di
    #: confronto lo e' (mossa estratta da una distribuzione, pick/drop
    #: probabilistici), quindi il paragone resta equo. Misurato su questo scenario:
    #: da ``0.00`` a ``0.05`` l'entropia finale passa da 44.7 a 36.6 e i ritorni
    #: sulla cella ``t-2`` da 30 % a 21 %. Mettere ``0.0`` per la greedy pura.
    inference_epsilon: float = 0.05


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

    @property
    def _n_move_actions(self) -> int:
        """Mosse selezionabili: 9 con ``STAY``, altrimenti solo le 8 direzioni reali.

        Le righe di ``q_move`` restano lunghe 9 (compatibilita' dei pickle): la
        colonna di ``STAY`` semplicemente non viene ne' scelta ne' usata per il
        bootstrap quando ``allow_stay`` e' ``False``.
        """
        return _N_MOVES if self.config.allow_stay else _N_MOVES - 1

    def bind_environment(self, env: object) -> None:
        """Lega un fallback euristico all'ambiente corrente (serve solo per A1/A2)."""
        self._heuristic = HeuristicPolicy(env.matrix, self.seed_types)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Discretizzazione dello stato
    # ------------------------------------------------------------------ #

    def _move_state(self, obs: Observation) -> tuple:
        """Stato per la testa del movimento: *dove* conviene andare e *cosa* e' bloccato.

        Volutamente **non** contiene ``direction`` ne' ``carried_type``:

        - le mosse sono gia' relative all'heading (avanti / dietro / dx / sx / ...),
          quindi ``direction`` replicherebbe la stessa situazione su 4 stati distinti,
          dividendo per 4 i dati per stato senza aggiungere informazione;
        - dove andare e' gia' riassunto da ``best_dir``, che tiene conto del tipo
          trasportato: il codice del tipo in se' non cambia la decisione di mossa.

        Cosi' la tabella passa da ~92 000 stati a ~4 600, molto piu' apprendibili.
        """
        return (
            int(obs.carrying),
            obs.best_dir,
            tuple(int(b) for b in obs.blocked),
        )

    def _manip_state(self, obs: Observation) -> tuple:
        # Per il DROP conta il tipo trasportato; per il PICK il tipo del seme
        # sotto la formica (non trasporta). Cosi' lo stato e' allineato al segnale
        # di ricompensa, che dipende dai simili di *quel* tipo (vedi env._reward).
        relevant = obs.carried_type if obs.carrying else obs.seed_here_type
        f_own = obs.f_here[relevant]
        bins = self.config.f_bins
        f_bin = min(bins - 1, int(f_own * bins))
        # Il segno della ricompensa di manipolazione si ribalta esattamente al
        # livello del caso (1/k, il ``pivot`` di ``RewardConfig``), che NON cade su
        # un confine dei bin: con k=5 il pivot vale 0.2 e finisce dentro il bin
        # [0, 0.25), dove il pick e' premiato a f=0.1 e punito a f=0.24. Senza
        # questo bit lo stato non puo' rappresentare il confine decisionale, la Q
        # media si appiattisce a ~0 e la testa resta indecisa (pochissimi pick).
        above_chance = int(f_own >= 1.0 / max(1, self.seed_types))
        return (int(obs.carrying), relevant, f_bin, above_chance)

    # ------------------------------------------------------------------ #
    # Selezione dell'azione
    # ------------------------------------------------------------------ #

    def _row(self, table: dict[tuple, list[float]], state: tuple, n: int) -> list[float]:
        row = table.get(state)
        if row is None:
            row = [0.0] * n
            table[state] = row
        return row

    def _argmax(self, values: list[float], limit: Optional[int] = None) -> int:
        head = values[:limit] if limit is not None else values
        best_value = max(head)
        winners = [index for index, value in enumerate(head) if value == best_value]
        # Parita' (tipico: stato mai visitato, riga ancora [0, 0, ...]): scelta
        # casuale invece di preferire sempre l'indice 0 (= no-op / prima direzione).
        return winners[0] if len(winners) == 1 else self.rng.choice(winners)

    def select_action(self, obs: Observation, ant: Ant) -> Action:
        if self._needs_heuristic and self._heuristic is None:
            raise RuntimeError(
                "ablazione A1/A2 senza euristica di fallback: passa heuristic= o chiama bind_environment()"
            )
        fallback = self._heuristic.select_action(obs, ant) if self._heuristic is not None else None
        # ``epsilon`` vale il decadimento in addestramento e ``inference_epsilon``
        # una volta congelata: ``frozen`` sospende l'apprendimento, non il rumore.
        exploring = self.rng.random() < self.epsilon

        # --- movimento ---
        move_state: Optional[tuple] = None
        if self.config.learn_move:
            n_moves = self._n_move_actions
            move_state = self._move_state(obs)
            row = self._row(self.q_move, move_state, _N_MOVES)
            move_index = self.rng.randrange(n_moves) if exploring else self._argmax(row, n_moves)
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
            # il bootstrap usa solo le azioni davvero selezionabili
            target = reward + gamma * max(next_row[: self._n_move_actions]) * keep
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
        """Congela la politica per l'inferenza (GUI / benchmark): niente piu' update.

        Resta il rumore residuo ``inference_epsilon``, indispensabile per non
        incastrarsi nei cicli limite indotti dagli stati aliasati (vedi
        :class:`QConfig`). Con ``inference_epsilon = 0`` si ottiene la greedy pura.
        """
        self.frozen = True
        self.epsilon = self.config.inference_epsilon

    # ------------------------------------------------------------------ #
    # Persistenza
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        payload = {
            "state_version": _STATE_VERSION,
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
        version = payload.get("state_version", 1)
        if version != _STATE_VERSION:
            _LOGGER.warning(
                "%s e' stata addestrata con la codifica di stato v%s, questa versione usa la v%s: "
                "la tabella non e' riutilizzabile e la politica si comporterebbe a caso. "
                "Riaddestrala con: python -m antelligent.train",
                path, version, _STATE_VERSION,
            )
        policy = cls(payload["seed_types"], payload["config"], heuristic=heuristic)
        policy.state_version = version
        policy.q_move = payload["q_move"]
        policy.q_manip = payload["q_manip"]
        policy.episode = payload.get("episode", 0)
        policy.epsilon = payload.get("epsilon", policy.config.epsilon_end)
        return policy
