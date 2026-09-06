"""Politica tabellare per la ricompensa "alla Lumer-Faieta" (variante sperimentale).

:class:`LumerQPolicy` **eredita** da :class:`~antelligent.policies.tabular.TabularQPolicy`:
stessa Q-learning fattorizzata, stesso ``_move_state`` compatto con il gradiente
locale ``best_dir``, stesso ``allow_stay = False`` e stesso ``inference_epsilon``.
Tutte le correzioni che hanno sbloccato le formiche restano quindi valide senza
essere riscritte: se cambiano li', cambiano anche qui.

L'unica differenza e' la **discretizzazione della testa di manipolazione**, che
va riallineata al nuovo segnale (vedi :meth:`LumerQPolicy._manip_state`).
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional

from ..actions import Observation
from ..lumer_reward import LumerRewardConfig
from .heuristic import HeuristicPolicy
from .tabular import QConfig, TabularQPolicy

_LOGGER = logging.getLogger(__name__)

#: Chiave scritta nel pickle per riconoscere la famiglia di politica: la codifica
#: della testa di manipolazione e' diversa da quella di ``TabularQPolicy``, quindi
#: caricare l'una come l'altra darebbe letture sistematicamente a vuoto (politica
#: che si comporta a caso). Vedi :func:`load_policy`.
KIND = "lumer"

#: Versione della codifica di stato di *questa* famiglia (lineage separato da
#: ``tabular._STATE_VERSION``). Da incrementare a ogni cambio di ``_manip_state``.
_STATE_VERSION = 1


class LumerQPolicy(TabularQPolicy):
    def __init__(
        self,
        seed_types: int,
        config: Optional[QConfig] = None,
        *,
        lumer: Optional[LumerRewardConfig] = None,
        heuristic: Optional[HeuristicPolicy] = None,
    ) -> None:
        super().__init__(seed_types, config, heuristic=heuristic)
        self.lumer_cfg = lumer or LumerRewardConfig()

    # ------------------------------------------------------------------ #
    # Discretizzazione dello stato
    # ------------------------------------------------------------------ #

    def _manip_state(self, obs: Observation) -> tuple:
        """Stato della testa pick/drop: ``(trasporta, bin di f, sopra l'indifferenza)``.

        Due differenze rispetto alla versione base, entrambe conseguenza diretta
        della nuova ricompensa:

        1. **il codice del tipo sparisce.** Con la ricompensa alla Lumer-Faieta il
           valore di una manipolazione e' ``g(azione, f)``: una funzione della sola
           frazione di simili. Il *quale* tipo sia non cambia nulla, entra gia'
           tutto in ``f``. Tenerlo replicherebbe la stessa decisione su ``k`` righe
           distinte dividendo per ``k`` i dati per stato. Si passa da
           ``2*k*bins*2`` a ``2*bins*2`` stati (16 con i default): la testa di
           manipolazione converge in pochi episodi.
        2. **il confine e' ``f* = sqrt(kp*kd)``**, non piu' il livello del caso
           ``1/k``. E' li' che la ricompensa cambia segno (indifferenza fra le due
           regole di Lumer-Faieta), e ``f*`` non cade su un confine dei bin: senza
           un bit dedicato lo stato non saprebbe distinguere "conviene" da "e'
           punito" dentro lo stesso bin.
        """
        relevant = obs.carried_type if obs.carrying else obs.seed_here_type
        f_own = obs.f_here[relevant]
        bins = self.config.f_bins
        f_bin = min(bins - 1, int(f_own * bins))
        above = int(f_own >= self.lumer_cfg.indifference_f)
        return (int(obs.carrying), f_bin, above)

    # ------------------------------------------------------------------ #
    # Persistenza (formato proprio: vedi KIND)
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        payload = {
            "policy_kind": KIND,
            "state_version": _STATE_VERSION,
            "seed_types": self.seed_types,
            "config": self.config,
            "lumer": self.lumer_cfg,
            "q_move": self.q_move,
            "q_manip": self.q_manip,
            "episode": self.episode,
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path, *, heuristic: Optional[HeuristicPolicy] = None) -> "LumerQPolicy":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        kind = payload.get("policy_kind", "tabular")
        version = payload.get("state_version", 0)
        if kind != KIND:
            _LOGGER.warning(
                "%s e' una politica '%s', non '%s': la codifica della testa di manipolazione "
                "e' diversa e la politica si comporterebbe a caso. "
                "Riaddestrala con: python -m antelligent.train_lumer",
                path, kind, KIND,
            )
        elif version != _STATE_VERSION:
            _LOGGER.warning(
                "%s usa la codifica di stato v%s, questa versione usa la v%s: la tabella non e' "
                "riutilizzabile. Riaddestrala con: python -m antelligent.train_lumer",
                path, version, _STATE_VERSION,
            )
        policy = cls(
            payload["seed_types"], payload["config"],
            lumer=payload.get("lumer") or LumerRewardConfig(), heuristic=heuristic,
        )
        policy.state_version = version
        policy.q_move = payload["q_move"]
        policy.q_manip = payload["q_manip"]
        policy.episode = payload.get("episode", 0)
        policy.epsilon = payload.get("epsilon", policy.config.epsilon_end)
        return policy


def policy_kind(path) -> str:
    """Famiglia della politica salvata in ``path`` (``"tabular"`` per i vecchi pickle)."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload.get("policy_kind", "tabular")


def load_policy(path, *, heuristic: Optional[HeuristicPolicy] = None) -> TabularQPolicy:
    """Carica una politica scegliendo la classe dalla famiglia scritta nel pickle.

    Serve alla GUI: le due varianti scrivono nello stesso formato ma discretizzano
    la testa di manipolazione in modo diverso, e caricare l'una come l'altra
    fallirebbe *in silenzio* (tutte le letture su righe mai viste -> azioni a caso).
    """
    cls = LumerQPolicy if policy_kind(path) == KIND else TabularQPolicy
    return cls.load(path, heuristic=heuristic)
