"""Tipi condivisi tra ambiente e politiche.

Sia l'euristica sia l'RL producono la stessa :class:`Action`; l'ambiente
(``environment.py``) la applica in modo identico per entrambe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

#: ``move_index`` che significa "non spostarti".
STAY = 8


class Manipulation(IntEnum):
    NOOP = 0
    PICK = 1
    DROP = 2


@dataclass(frozen=True)
class Action:
    """Cosa fa la formica in un tick: una manipolazione + una mossa."""

    manipulation: Manipulation
    #: 0-7 = direzione relativa (indici di ``check_move.check_around``), 8 = fermo.
    move_index: int


@dataclass(frozen=True)
class Observation:
    """Vista locale e parziale della formica sull'ambiente."""

    carrying: bool
    carried_type: int          # codice del seme trasportato, 0 se non trasporta
    seed_here: bool            # c'e' un seme sulla cella della formica
    seed_here_type: int        # codice del seme sulla cella della formica, 0 se non c'e'
    can_pick: bool             # non trasporta ed e' su un seme
    can_drop: bool             # trasporta e la sua cella e' libera da semi
    f_here: tuple[float, ...]  # frazione di semi per tipo nelle 8 celle adiacenti
    blocked: tuple[bool, ...]  # per ognuna delle 8 direzioni relative: fuori griglia o cella occupata da una formica
    #: Gradiente locale, direzione relativa 0-7 (``STAY`` = nessun bersaglio in vista):
    #: trasportando, la cella *vuota* piu' circondata da semi del tipo in mano (dove
    #: posare bene); a mani libere, la cella col seme piu' fuori posto (da raccogliere).
    best_dir: int
    direction: int             # heading corrente (0-3)
    window: tuple[tuple[int, ...], ...]  # finestra categorica (2r+1)x(2r+1) per il deep RL


@dataclass(frozen=True)
class Transition:
    obs: Observation
    action: Action
    reward: float
    next_obs: Observation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    info: dict[str, Any]
