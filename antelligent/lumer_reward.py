"""Ricompense derivate dalle **probabilita' di Lumer-Faieta** (variante sperimentale).

Questo modulo e' *puro*: solo formule, nessuna matrice e nessun RNG. Serve alla
variante di ricompensa che, invece di misurare la manipolazione rispetto al
livello del caso (``environment.RewardConfig``), la misura con le **stesse
funzioni che governano l'euristica**:

.. math::

    P_{pick}(f) = \\left(\\frac{k_p}{k_p + f}\\right)^2 \\qquad
    P_{drop}(f) = \\left(\\frac{f}{k_d + f}\\right)^2

dove ``f`` e' la frazione di semi *dello stesso tipo* fra gli 8 vicini della cella.
Sono le identiche formule di :meth:`SeedMatrix.pick_probability` /
:meth:`~SeedMatrix.drop_probability` (li' riscalate a 0-100 per il confronto col
tiro di dado; qui restano normalizzate in ``[0, 1]``, perche' servono come
*valore*, non come probabilita' da estrarre).

L'euristica **campiona** da queste probabilita'; qui invece le usiamo come
**segnale di ricompensa denso**: l'agente non estrae piu' a sorte, ma riceve un
rinforzo tanto piu' alto quanto piu' l'azione scelta e' quella che l'euristica
giudicherebbe conveniente *in quel punto*. E' il senso della richiesta
"avvicinarci all'algoritmo euristico": stessa conoscenza di dominio, ma appresa
invece che cablata — e quindi migliorabile dall'RL, che puo' scoprire *dove
andare* per trovare i punti buoni, cosa che l'euristica (random walk) non fa.

Punto di indifferenza
---------------------

Il segnale deve essere **con segno**, altrimenti "manipolare" domina sempre
"non manipolare" e la colonia entra in churn (raccogli-e-riposa a caso). Il punto
naturale in cui il segno si ribalta non e' arbitrario: e' il ``f`` in cui le due
regole di Lumer-Faieta si equivalgono,

.. math::

    P_{pick}(f^*) = P_{drop}(f^*) \\iff \\frac{k_p}{k_p+f} = \\frac{f}{k_d+f}
    \\iff f^{*} = \\sqrt{k_p k_d}

Con i valori standard ``k_p = 0.1``, ``k_d = 0.3`` si ha ``f* ~ 0.1732``: sotto,
l'euristica preferisce raccogliere; sopra, preferisce posare. E' lo stesso ruolo
che nel modello precedente aveva il "livello del caso" ``1/k``, ma qui deriva
dalle costanti dell'euristica invece che dal numero di tipi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Costanti di Lumer-Faieta, le stesse usate dall'euristica
#: (``simulation/seed_matrix.py``). Un test di guardia verifica che non divergano.
KP_PICK = 0.1
KD_DROP = 0.3

#: Manipolazione *possibile* nella cella in cui si trova la formica.
PICK = "pick"
DROP = "drop"
NONE = ""

#: Modalita' di trasformazione delle probabilita' in ricompensa (vedi
#: :func:`manip_value`). ``centered`` e' il default *misurato*; ``advantage`` ha
#: lo zero nel punto di indifferenza di Lumer-Faieta;
#: ``raw`` non e' centrata e serve come ablazione (porta al churn).
MODES = ("advantage", "centered", "raw")


@dataclass(frozen=True)
class LumerRewardConfig:
    """Pesi della ricompensa "alla Lumer-Faieta".

    ``pick_scale`` / ``drop_scale`` sono separati apposta: la domanda di ricerca
    e' *quanto conta il punto in cui si posa il seme*, quindi il peso del drop
    deve poter essere alzato indipendentemente da quello del pick.

    Il default ``drop_scale = 2.0`` non e' arbitrario. In modalita' ``centered`` i
    due segnali hanno **escursione molto diversa**: il pick vale al massimo
    ``2*P_pick(0) - 1 = +1.0`` (seme isolato), il drop al massimo
    ``2*P_drop(1) - 1 = +0.18`` (seme circondato solo da simili) — un quinto. Senza
    riequilibrio la testa di manipolazione impara molto meglio *quando raccogliere*
    che *dove posare*. Misurato su 10 ``master_seed`` x 3 seed di addestramento:
    ``drop_scale = 2`` porta l'entropia finale da ~21 a ~15, battendo l'euristica su
    10 seed su 10. Lo sweep 1.0/1.5/2/3/4 non e' monotono e la varianza fra seed di
    addestramento e' dello stesso ordine dello sweep: ``2`` e' il punto migliore
    misurato, non un ottimo dimostrato.

    ``decline_scale`` punisce il **rifiuto** di una manipolazione conveniente: se
    la formica poteva agire con vantaggio positivo e ha scelto il no-op, paga
    quel vantaggio. Senza, rifiutare e' sempre gratis e la testa di manipolazione
    riceve segnale solo sul ramo "agisci".

    Il rifiuto e' **unilaterale** di proposito: rifiutare una manipolazione
    *sconveniente* vale ``0``, non un premio. ``decline_two_sided = True``
    ripristina la versione simmetrica (premio ``-valore`` anche quando il valore
    e' negativo) — che e' un'ablazione istruttiva ma **patologica**: pagare una
    formica ogni tick perche' "sta correttamente ferma su un buon grappolo" crea
    una rendita di posizione. Misurato: con il rifiuto simmetrico la politica
    impara a scegliere una direzione **fuori griglia** nel 39-44 % dei passi —
    ``_apply_move`` la tratta come "resta ferma" senza addebitare
    ``contested_penalty``, quindi e' un ``STAY`` gratuito dalla porta di servizio
    (stessa patologia di ``QConfig.allow_stay``, vedi ``wall_penalty``).

    ``wall_penalty`` chiude proprio quella porta: addebita la scelta di una
    direzione fuori dalla griglia, che l'osservazione segnala gia' in
    ``blocked``. Senza, restare fermi contro un bordo e' l'unica azione a costo
    zero del problema.

    ``shaping_scale`` attiva lo **shaping potenziale** ``F = gamma*Phi(s') -
    Phi(s)`` con ``Phi`` = probabilita' di Lumer-Faieta della manipolazione
    desiderata nella cella corrente (Ng, Harada & Russell 1999). E' il termine che
    insegna *dove andare*: una formica che trasporta guadagna avvicinandosi alle
    celle dove ``P_drop`` e' alta, cioe' ai punti in cui posare quel seme e'
    giusto. Essendo potenziale, non cambia la politica ottima: cambia solo la
    velocita' con cui la si trova.

    ``isolated_guard`` riproduce alla lettera la guardia dell'euristica
    (``sum_all == 0 -> P = 0`` per *entrambe* le regole). Di default e' ``False``
    e il vicinato vuoto viene trattato come ``f = 0``, cioe' estendendo per
    continuita' le formule: raccogliere un seme completamente isolato vale ``+1``
    (e' il seme piu' fuori posto che esista) e posarne uno nel deserto vale
    ``-1``. La guardia dell'euristica e' un artificio numerico sullo ``0/0``, non
    una scelta di modello, e riprodurla renderebbe il deserto un posto *neutro*
    dove scaricare i semi.
    """

    kp: float = KP_PICK
    kd: float = KD_DROP
    pick_scale: float = 1.0
    drop_scale: float = 2.0
    decline_scale: float = 1.0
    decline_two_sided: bool = False
    shaping_scale: float = 1.0
    wall_penalty: float = 0.05
    #: fattore di sconto usato dal termine di shaping: deve coincidere con il
    #: ``gamma`` della Q-learning, altrimenti lo shaping non e' piu' potenziale.
    gamma: float = 0.95
    isolated_guard: bool = False
    mode: str = "centered"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode deve essere uno di {MODES}, non {self.mode!r}")

    @property
    def indifference_f(self) -> float:
        """``f* = sqrt(kp*kd)``: la frazione di simili in cui l'euristica e' indifferente."""
        return math.sqrt(self.kp * self.kd)

    def scale_for(self, kind: str) -> float:
        return self.pick_scale if kind == PICK else self.drop_scale


def pick_probability(f: float, kp: float = KP_PICK) -> float:
    """``(kp/(kp+f))^2`` normalizzata in ``[0, 1]``. Decrescente in ``f``."""
    return (kp / (kp + f)) ** 2


def drop_probability(f: float, kd: float = KD_DROP) -> float:
    """``(f/(kd+f))^2`` normalizzata in ``[0, 1]``. Crescente in ``f``."""
    return (f / (kd + f)) ** 2


def probabilities(f: float, cfg: LumerRewardConfig, *, isolated: bool = False) -> tuple[float, float]:
    """``(P_pick, P_drop)`` nella cella, gestendo il vicinato vuoto (vedi la config)."""
    if isolated and cfg.isolated_guard:
        return (0.0, 0.0)
    return (pick_probability(f, cfg.kp), drop_probability(f, cfg.kd))


def manip_value(kind: str, f: float, cfg: LumerRewardConfig, *, isolated: bool = False) -> float:
    """Valore (con segno, in ``[-1, 1]``) della manipolazione ``kind`` con densita' ``f``.

    - ``centered``  — ``2*P(azione) - 1``: "quanto spesso l'euristica lo farebbe",
      centrato su una probabilita' del 50 %. E' la soglia piu' **selettiva**: con
      ``kd = 0.3`` il drop diventa conveniente solo da ``f > 0.72``, cioe' solo in
      un punto davvero buono. E' il default, ed e' quello che misura meglio;
    - ``advantage`` — ``P(azione) - P(azione opposta)``: positivo finche' l'euristica
      preferisce quella mossa, zero esattamente in ``f* = sqrt(kp*kd)``;
    - ``raw``       — ``P(azione)``, mai negativo: ablazione che mostra il churn.
    """
    if kind == NONE:
        return 0.0
    p_pick, p_drop = probabilities(f, cfg, isolated=isolated)
    mine, other = (p_pick, p_drop) if kind == PICK else (p_drop, p_pick)
    if cfg.mode == "advantage":
        return mine - other
    if cfg.mode == "centered":
        return 2.0 * mine - 1.0
    return mine  # "raw"


def potential(kind: str, f: float, cfg: LumerRewardConfig, *, isolated: bool = False) -> float:
    """``Phi(s)``: probabilita' di Lumer-Faieta della manipolazione *desiderata* qui.

    Vale ``0`` quando nella cella non c'e' niente da fare (formica libera su cella
    vuota), cosi' il potenziale e' definito su tutto lo spazio degli stati e lo
    shaping resta invariante rispetto alla politica ottima.
    """
    if kind == NONE:
        return 0.0
    p_pick, p_drop = probabilities(f, cfg, isolated=isolated)
    return p_pick if kind == PICK else p_drop


def decline_value(kind: str, f: float, cfg: LumerRewardConfig, *, isolated: bool = False) -> float:
    """Ricompensa (<= 0 di default) per aver **rifiutato** la manipolazione possibile.

    Unilaterale: rifiutare una manipolazione conveniente costa il suo valore,
    rifiutarne una sconveniente non vale nulla. Vedi :class:`LumerRewardConfig`
    per la ragione (la versione simmetrica crea una rendita di posizione).
    """
    value = manip_value(kind, f, cfg, isolated=isolated)
    if cfg.decline_two_sided:
        return -cfg.decline_scale * value
    return -cfg.decline_scale * max(0.0, value)
