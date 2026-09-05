"""Logica di dominio della matrice condivisa.

La classe espone due livelli (vedi ``docs/rl-design.md`` §4.2):

- **primitive meccaniche** — accesso alla cella, lock di occupazione,
  ``count_types_around``, ``commit_pick`` / ``commit_drop`` (mutazioni
  incondizionate), calcolo di entropia;
- **regole euristiche pure** — ``pick_probability`` / ``drop_probability``
  (formule di Lumer-Faieta, senza RNG ne' mutazione), usate sia da
  ``pick_seed`` / ``drop_seed`` (compatibilita' col codice esistente) sia da
  ``HeuristicPolicy``.

L'ambiente RL (``environment.py``) applica manipolazione e movimento chiamando
``commit_pick`` / ``commit_drop`` + i lock di cella; la *decisione* se raccogliere
o posare spetta alla politica.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Optional

from ..seeds.seed import Seed
from ..utilities import check_move
from ..utilities.position import Position

_LOGGER = logging.getLogger(__name__)

_KP_PICK = 0.1
_KD_DROP = 0.3


class SeedMatrix(ABC):
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._random = random.Random()

    # --- accesso alla singola cella (implementato dalle sottoclassi) ---

    @abstractmethod
    def get_seed_at(self, pos_x: int, pos_y: int) -> Optional[Seed]:
        """Seme nella cella, o ``None`` se vuota."""

    @abstractmethod
    def set_seed_at(self, pos_x: int, pos_y: int, seed: Seed) -> None:
        """Inserisce un seme in una cella (il chiamante ha gia' verificato che fosse vuota)."""

    @abstractmethod
    def clear_seed_at(self, pos_x: int, pos_y: int) -> None:
        """Svuota la cella."""

    # --- lock di occupazione della cella (implementato dalle sottoclassi) ---

    @abstractmethod
    def try_acquire_cell(self, pos_x: int, pos_y: int) -> bool:
        """Prova ad acquisire, senza bloccare, il lock della cella. ``True`` se riuscito."""

    @abstractmethod
    def acquire_cell(self, pos_x: int, pos_y: int) -> None:
        """Acquisisce il lock della cella, bloccando (usato al piazzamento iniziale)."""

    @abstractmethod
    def release_cell(self, pos_x: int, pos_y: int) -> None:
        """Rilascia il lock della cella."""

    # --- primitive comuni ---

    def has_seed(self, pos_x: int, pos_y: int) -> bool:
        try:
            return self.get_seed_at(pos_x, pos_y) is not None
        except Exception:  # noqa: BLE001 - fuori dai limiti della matrice
            _LOGGER.debug("has_seed fuori dai limiti della matrice", exc_info=True)
            return False

    def place_seed(self, pos_x: int, pos_y: int, seed: Seed) -> bool:
        try:
            if self.has_seed(pos_x, pos_y):
                return False
            self.set_seed_at(pos_x, pos_y, seed)
            return True
        except Exception:  # noqa: BLE001
            _LOGGER.debug("place_seed fallito", exc_info=True)
            return False

    def count_types_around(self, pos_x: int, pos_y: int, seed_types: int) -> list[int]:
        """Conteggio dei semi per tipo nelle 8 celle adiacenti."""
        types_around = [0] * seed_types
        for neighbor in check_move.check_around(pos_x, pos_y, self.rows, self.cols, 0):
            if neighbor is None:
                continue
            if self.has_seed(neighbor.x, neighbor.y):
                neighbor_seed = self.get_seed_at(neighbor.x, neighbor.y)
                if neighbor_seed is not None:
                    types_around[neighbor_seed.code] += 1
        return types_around

    # --- regole euristiche pure (Lumer-Faieta) ---

    def pick_probability(self, pos_x: int, pos_y: int, seed_code: int, seed_types: int) -> float:
        """``P_pick = (kp/(kp+f))^2 * 100``. Ritorna ``0`` per un seme isolato.

        (Con un seme isolato ``f = 0/0`` non e' definita: il caso e' gestito qui in modo esplicito.)
        """
        types_around = self.count_types_around(pos_x, pos_y, seed_types)
        sum_all = sum(types_around)
        if sum_all == 0:
            return 0.0
        fc = types_around[seed_code] / sum_all
        return ((_KP_PICK / (_KP_PICK + fc)) ** 2) * 100

    def drop_probability(self, pos_x: int, pos_y: int, seed_code: int, seed_types: int) -> float:
        """``P_drop = (f/(kd+f))^2 * 100``. Ritorna ``0`` se non ci sono semi vicini."""
        types_around = self.count_types_around(pos_x, pos_y, seed_types)
        sum_all = sum(types_around)
        if sum_all == 0:
            return 0.0
        fc = types_around[seed_code] / sum_all
        return ((fc / (_KD_DROP + fc)) ** 2) * 100

    # --- mutazioni incondizionate (usate dall'ambiente RL) ---

    def commit_pick(self, pos_x: int, pos_y: int) -> Optional[Seed]:
        """Rimuove e restituisce il seme nella cella (``None`` se vuota). Nessuna probabilita'."""
        seed = self.get_seed_at(pos_x, pos_y)
        if seed is None:
            return None
        seed.seed_taken()
        self.clear_seed_at(pos_x, pos_y)
        return seed

    def commit_drop(self, pos_x: int, pos_y: int, seed: Seed) -> bool:
        """Posa il seme nella cella se libera. Nessuna probabilita'. ``True`` se posato."""
        if self.has_seed(pos_x, pos_y):
            return False
        seed.seed_dropped(Position(pos_x, pos_y))
        self.set_seed_at(pos_x, pos_y, seed)
        return True

    # --- wrapper probabilistici (compatibilita' col codice esistente / euristica) ---

    def pick_seed(self, pos_x: int, pos_y: int, seed_types: int) -> Optional[Seed]:
        try:
            seed = self.get_seed_at(pos_x, pos_y)
            if seed is None:
                return None
            probability = self.pick_probability(pos_x, pos_y, seed.code, seed_types)
            if probability <= 0:
                return None
            if (self._random.random() * 100 + 1) <= probability:
                return self.commit_pick(pos_x, pos_y)
            return None
        except Exception:  # noqa: BLE001
            _LOGGER.debug("pick_seed fallito", exc_info=True)
            return None

    def drop_seed(self, pos_x: int, pos_y: int, seed: Seed, seed_types: int) -> bool:
        try:
            if self.has_seed(pos_x, pos_y):
                return False
            probability = self.drop_probability(pos_x, pos_y, seed.code, seed_types)
            if probability <= 0:
                return False
            if (self._random.random() * 100 + 1) <= probability:
                return self.commit_drop(pos_x, pos_y, seed)
            return False
        except Exception:  # noqa: BLE001
            _LOGGER.debug("drop_seed fallito", exc_info=True)
            return False

    # --- entropia ---

    def _local_entropy(self, pos_x: int, pos_y: int, seed_types: int, denom: float) -> float:
        """Entropia locale (scala 0-100) della cella con seme ``(pos_x, pos_y)``."""
        seed = self.get_seed_at(pos_x, pos_y)
        neighbors = check_move.check_around(pos_x, pos_y, self.rows, self.cols, 0)
        num = 0.0
        for k in range(seed_types):
            sum_same = 1.0 if seed.code == k else 0.0
            sum_all = 1.0
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                if self.has_seed(neighbor.x, neighbor.y):
                    temp_seed = self.get_seed_at(neighbor.x, neighbor.y)
                    if temp_seed is not None and temp_seed.code == k:
                        sum_same += 1.0
                    sum_all += 1.0
            if sum_all == 0:
                sum_all = 1.0
            g = sum_same / sum_all
            if g == 0:
                g = 1.0
            num += g * math.log(1.0 / g)
        return (num / denom) * 100

    def check_entropy_placed(self, seed_types: int) -> float:
        """Entropia media sui **soli semi posati**: sempre definita (0-100), ``0`` se nessuno.

        E' il valore usato dall'ambiente RL come costo / segnale di ricompensa e per
        la statistica "entropia totale" della GUI. A differenza di
        :meth:`check_entropy` non restituisce mai ``-1``.
        """
        denom = math.log(seed_types) if seed_types > 1 else 1.0
        total = 0.0
        count = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.has_seed(j, i):
                    total += self._local_entropy(j, i, seed_types, denom)
                    count += 1
        return total / count if count else 0.0

    def check_entropy(self, total_seeds: int, seed_types: int, full_computation: bool) -> float:
        """Entropia media del sistema (scala 0-100), oppure ``-1.0``.

        Restituisce ``-1`` se non tutti i semi sono posati (una formica ne
        trasporta uno) o se la scansione stocastica esce in anticipo. Usata per
        il criterio di arresto "stretto".
        """
        denom = math.log(seed_types) if seed_types > 1 else 1.0
        local_entropies: list[float] = []
        num_seeds = 0
        ok = True
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.has_seed(j, i):
                    continue
                local_entropy = self._local_entropy(j, i, seed_types, denom)
                local_entropies.append(local_entropy)
                rand = 101.0 if full_computation else (self._random.random() * 100 + 1)
                if not (rand >= local_entropy):
                    ok = False
                    break
                num_seeds += 1
            if not ok:
                break

        if not ok:
            return -1.0
        if num_seeds != total_seeds:
            return -1.0
        if not local_entropies:
            return 0.0
        return sum(local_entropies) / len(local_entropies)
