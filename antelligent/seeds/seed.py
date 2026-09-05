"""Un seme sulla matrice.

Il caricamento/scalatura dell'immagine non e' qui: la GUI (single thread) mappa
:attr:`Seed.image_path` sullo sprite gia' pronto. Cosi' la classe resta pura e
usabile dai thread lavoratori senza toccare Tk/Pillow.
"""

from __future__ import annotations

from ..utilities.position import Position
from .seed_type import SeedType

BLANK_SEED_IMAGE = "seeds/blankSeed.png"


class Seed:
    def __init__(self, seed_type: SeedType) -> None:
        self.type = seed_type
        self.position = Position(0, 0)
        self.taken = False

    @property
    def code(self) -> int:
        return self.type.code

    @property
    def image_path(self) -> str:
        """Sprite corrente: seme "vuoto" mentre e' trasportato, altrimenti colorato."""
        return BLANK_SEED_IMAGE if self.taken else self.type.seed_image_path

    def set_position(self, x: int, y: int) -> None:
        self.position.set(x, y)

    def seed_taken(self) -> None:
        self.taken = True

    def seed_dropped(self, position: Position) -> None:
        self.taken = False
        self.position.set(position.x, position.y)
