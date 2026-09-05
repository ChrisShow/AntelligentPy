"""I cinque tipi di seme.

Ogni tipo porta con se' il codice numerico storico (usato dalle formule di
pick/drop/entropia) e i percorsi degli sprite, relativi a
``antelligent/resources/images/``.
"""

from __future__ import annotations

import random
from enum import Enum


class SeedType(Enum):
    BLUE = (0, "seeds/blueSeed.png", "ants/blueAnt.png")
    PURPLE = (1, "seeds/purpleSeed.png", "ants/purpleAnt.png")
    GRAY = (2, "seeds/graySeed.png", "ants/grayAnt.png")
    ORANGE = (3, "seeds/orangeSeed.png", "ants/orangeAnt.png")
    YELLOW = (4, "seeds/yellowSeed.png", "ants/yellowAnt.png")

    def __init__(self, code: int, seed_image_path: str, ant_image_path: str) -> None:
        self.code = code
        self.seed_image_path = seed_image_path
        self.ant_image_path = ant_image_path

    @classmethod
    def from_code(cls, code: int) -> "SeedType":
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"codice seme sconosciuto: {code}")

    @classmethod
    def random(cls, rng: random.Random, seed_types_count: int) -> "SeedType":
        """Tipo casuale tra i primi ``seed_types_count`` valori dell'enum.

        Riproduce la distribuzione canonica (0=Blue, 1=Purple, 2=Gray, 3=Orange,
        4 o oltre=Yellow) ed e' robusta a valori fuori range.
        """
        members = list(cls)
        bound = max(1, min(seed_types_count, len(members)))
        index = min(rng.randrange(bound), len(members) - 1)
        return members[index]
