"""Ambiente con ricompensa "alla Lumer-Faieta" (variante sperimentale).

:class:`LumerFaietaEnvironment` e' una **sottoclasse** di
:class:`~antelligent.environment.Environment`: fisica, osservazione, gradiente
locale ``best_dir``, lock di cella, contatori ed entropia restano *identici*
(quindi tutte le correzioni gia' fatte — formiche che non restano ferme,
osservazione compatta, bersaglio locale in entrambe le fasi — valgono anche qui).
Cambia **solo** il segnale di ricompensa:

===================  ==================================================================
``Environment``      manipolazione misurata rispetto al livello del caso ``1/k``
``LumerFaieta...``   manipolazione misurata con ``P_pick`` / ``P_drop`` di Lumer-Faieta
===================  ==================================================================

Quattro contributi, tutti pesabili da riga di comando:

1. **azione** — se la manipolazione riesce, ricompensa ``scale * valore(P)``;
2. **rifiuto** — se poteva manipolare con vantaggio e non l'ha fatto, paga quel
   vantaggio (``decline_scale``, unilaterale);
3. **navigazione** — shaping potenziale ``F = gamma*Phi(s') - Phi(s)`` con
   ``Phi`` = probabilita' di Lumer-Faieta della manipolazione desiderata nella
   cella (``shaping_scale``). E' il termine che risponde alla domanda "quanto
   conta il *punto* in cui lascio il seme": avvicinarsi a una cella con ``P_drop``
   alta viene premiato tick per tick, non solo nell'istante del deposito;
4. **muro** — scegliere una direzione fuori griglia costa ``wall_penalty``:
   altrimenti sarebbe l'unica azione a costo zero del problema, e riaprirebbe il
   punto fisso assorbente che ``QConfig.allow_stay = False`` aveva chiuso.

Vedi :mod:`antelligent.lumer_reward` per le formule e il punto di indifferenza.
"""

from __future__ import annotations

import random
from typing import Optional

from .actions import STAY, Action, StepResult
from .environment import Environment, RewardConfig
from .lumer_reward import (
    DROP,
    NONE,
    PICK,
    LumerRewardConfig,
    decline_value,
    manip_value,
    potential,
)
from .seeds.seed import Seed
from .simulation.ant import Ant
from .simulation.lock_seed_matrix import LockSeedMatrix
from .utilities import check_move
from .world import InitialState

#: Contesto della cella su cui si trova la formica: quale manipolazione e'
#: *possibile* li', con quale densita' di simili e se il vicinato e' vuoto.
_Context = tuple[str, float, bool]
_NO_CONTEXT: _Context = (NONE, 0.0, False)


class LumerFaietaEnvironment(Environment):
    def __init__(
        self,
        matrix: LockSeedMatrix,
        ants: list[Ant],
        seeds: list[Seed],
        seed_types: int,
        *,
        reward: Optional[RewardConfig] = None,
        lumer: Optional[LumerRewardConfig] = None,
        rng: Optional[random.Random] = None,
        window_radius: int = 2,
        entropy_every: int = 1,
    ) -> None:
        super().__init__(
            matrix, ants, seeds, seed_types,
            reward=reward, rng=rng, window_radius=window_radius, entropy_every=entropy_every,
        )
        self.lumer_cfg = lumer or LumerRewardConfig()
        self._context: _Context = _NO_CONTEXT
        self._into_wall = False

    @classmethod
    def from_initial_state(  # type: ignore[override]
        cls,
        init: InitialState,
        *,
        lumer: Optional[LumerRewardConfig] = None,
        **kwargs,
    ) -> "LumerFaietaEnvironment":
        env = super().from_initial_state(init, **kwargs)
        env.lumer_cfg = lumer or LumerRewardConfig()
        return env

    # ------------------------------------------------------------------ #
    # Contesto della cella
    # ------------------------------------------------------------------ #

    def cell_context(self, ant: Ant) -> _Context:
        """``(manipolazione possibile, f dei simili, vicinato vuoto)`` nella cella della formica.

        ``f`` e' la stessa quantita' che l'euristica passa alle sue formule: la
        frazione, fra gli **8 vicini**, dei semi dello stesso tipo di quello
        manipolato (quello trasportato per il drop, quello sotto la formica per il
        pick). La cella centrale non entra nel conteggio, esattamente come in
        :meth:`SeedMatrix.pick_probability`.
        """
        x, y = ant.position.x, ant.position.y
        if ant.carried_seed is not None:
            if self.matrix.has_seed(x, y):
                return _NO_CONTEXT  # cella occupata: qui non si puo' posare
            kind, code = DROP, ant.carried_seed.code
        else:
            seed = self.matrix.get_seed_at(x, y)
            if seed is None:
                return _NO_CONTEXT  # niente da raccogliere
            kind, code = PICK, seed.code
        counts = self.matrix.count_types_around(x, y, self.seed_types)
        total = sum(counts)
        return (kind, (counts[code] / total) if total else 0.0, total == 0)

    def _potential(self, ant: Ant) -> float:
        kind, f, isolated = self.cell_context(ant)
        return potential(kind, f, self.lumer_cfg, isolated=isolated)

    # ------------------------------------------------------------------ #
    # Passo
    # ------------------------------------------------------------------ #

    def step(self, ant: Ant, action: Action) -> StepResult:
        # Il contesto va letto PRIMA che ``super().step`` muti la matrice: e' lo
        # stato che l'euristica avrebbe visto al momento della decisione.
        self._context = self.cell_context(ant)
        self._into_wall = self._points_outside(ant, action.move_index)
        cfg = self.lumer_cfg
        phi_before = (
            potential(self._context[0], self._context[1], cfg, isolated=self._context[2])
            if cfg.shaping_scale
            else 0.0
        )

        result = super().step(ant, action)  # chiama il nostro _reward()

        if not cfg.shaping_scale:
            return result
        # Shaping potenziale: dipende dallo stato di arrivo, quindi non puo' stare
        # dentro _reward() (che l'ambiente base invoca prima di restituire).
        shaping = cfg.shaping_scale * (cfg.gamma * self._potential(ant) - phi_before)
        result.info["lf_shaping"] = shaping
        return StepResult(reward=result.reward + shaping, done=result.done, info=result.info)

    def _points_outside(self, ant: Ant, move_index: int) -> bool:
        """La mossa scelta punta fuori dalla griglia? (``blocked`` lo dice gia' alla politica.)"""
        if move_index == STAY:
            return False
        neigh = check_move.check_around(
            ant.position.x, ant.position.y, self.matrix.rows, self.matrix.cols, ant.direction
        )
        return neigh[move_index] is None

    def _reward(self, info: dict[str, object]) -> float:
        cfg, lf = self.reward_cfg, self.lumer_cfg
        kind, f, isolated = self._context

        reward = -cfg.step_penalty
        if info["contested"]:
            reward -= cfg.contested_penalty
        if info["manip_attempted"] and not info["manip_success"]:
            reward -= cfg.invalid_penalty
        if self._into_wall:
            # Puntare fuori griglia e' l'unica azione a costo zero del problema:
            # ``_apply_move`` la tratta come "resta ferma" senza contesa. Non
            # addebitarla riapre dalla porta di servizio il punto fisso assorbente
            # che ``QConfig.allow_stay = False`` aveva chiuso.
            reward -= lf.wall_penalty

        value = 0.0
        if kind != NONE:
            scale = lf.scale_for(kind)
            value = manip_value(kind, f, lf, isolated=isolated) * scale
            if info["manip_success"]:
                reward += value
            elif not info["manip_attempted"]:
                # ha rifiutato una manipolazione possibile: paga il vantaggio
                # buttato via (unilaterale, vedi decline_value)
                reward += decline_value(kind, f, lf, isolated=isolated) * scale

        info.update(lf_kind=kind, lf_f=f, lf_value=value, lf_wall=self._into_wall)
        return reward
