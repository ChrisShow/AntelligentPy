"""Coordinata (x, y) su griglia."""

from __future__ import annotations


class Position:
    """Una cella della griglia. ``x`` = colonna, ``y`` = riga."""

    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    @classmethod
    def copy_of(cls, other: "Position") -> "Position":
        return cls(other.x, other.y)

    def set(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Position) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"
