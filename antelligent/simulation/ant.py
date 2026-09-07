"""La formica come "corpo".

Dopo la rifattorizzazione per l'RL la formica non
contiene piu' logica di decisione: posizione, heading e seme trasportato sono
stato puro. Le decisioni sono della politica (``policies/``), l'applicazione
(mosse, pick/drop, lock di cella, contesa) e' dell'ambiente (``environment.py``).
"""

from __future__ import annotations

from typing import Optional

from ..seeds.seed import Seed
from ..utilities.position import Position

BLACK_ANT_IMAGE = "ants/blackAnt.png"


class Ant:
    def __init__(self, code: int, position: Position, direction: int) -> None:
        self.code = code
        self.position = position
        self.direction = direction
        self.carried_seed: Optional[Seed] = None

    @property
    def image_path(self) -> str:
        return BLACK_ANT_IMAGE if self.carried_seed is None else self.carried_seed.type.ant_image_path

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ant) and self.code == other.code

    def __hash__(self) -> int:
        return hash(self.code)
