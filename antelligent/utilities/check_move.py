"""Regole di movimento delle formiche.

Direzioni: 0 = UP, 1 = DOWN, 2 = DX (destra), 3 = SX (sinistra).

:func:`neighbours_with_directions` e la firma con ``rng`` opzionale servono al
meccanismo di risoluzione della contesa sulle celle (vedi ``environment.py``):
quando una formica non riesce ad acquisire la cella scelta ripiega su un'altra
cella adiacente libera invece di perdere la mossa.
"""

from __future__ import annotations

import random
from typing import Optional

from .position import Position

# Soglie cumulative (0-100) della distribuzione delle mosse:
# avanti 25%, dietro 5%, dx 15%, sx 15%, avanti-dx 15%, avanti-sx 15%,
# dietro-dx 5%, dietro-sx 5%.
FORWARD = 25
BACK = 30
DX = 45
SX = 60
FORWARD_DX = 75
FORWARD_SX = 90
BACK_DX = 95
BACK_SX = 100

#: ``move_index`` restituito quando nessuna cella adiacente e' in griglia (es. 1x1).
STAY = 8

_default_rng = random.Random()


def check_forward(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y > 0:
            return Position(pos_x, pos_y - 1)
    elif direction == 1:
        if pos_y < rows - 1:
            return Position(pos_x, pos_y + 1)
    elif direction == 2:
        if pos_x < cols - 1:
            return Position(pos_x + 1, pos_y)
    elif direction == 3:
        if pos_x > 0:
            return Position(pos_x - 1, pos_y)
    return None


def check_back(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y < rows - 1:
            return Position(pos_x, pos_y + 1)
    elif direction == 1:
        if pos_y > 0:
            return Position(pos_x, pos_y - 1)
    elif direction == 2:
        if pos_x > 0:
            return Position(pos_x - 1, pos_y)
    elif direction == 3:
        if pos_x < cols - 1:
            return Position(pos_x + 1, pos_y)
    return None


def check_dx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_x < cols - 1:
            return Position(pos_x + 1, pos_y)
    elif direction == 1:
        if pos_x > 0:
            return Position(pos_x - 1, pos_y)
    elif direction == 2:
        if pos_y < rows - 1:
            return Position(pos_x, pos_y + 1)
    elif direction == 3:
        if pos_y > 0:
            return Position(pos_x, pos_y - 1)
    return None


def check_sx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_x > 0:
            return Position(pos_x - 1, pos_y)
    elif direction == 1:
        if pos_x < cols - 1:
            return Position(pos_x + 1, pos_y)
    elif direction == 2:
        if pos_y > 0:
            return Position(pos_x, pos_y - 1)
    elif direction == 3:
        if pos_y < rows - 1:
            return Position(pos_x, pos_y + 1)
    return None


def check_forward_dx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y > 0 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y - 1)
    elif direction == 1:
        if pos_y < rows - 1 and pos_x > 0:
            return Position(pos_x - 1, pos_y + 1)
    elif direction == 2:
        if pos_y < rows - 1 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y + 1)
    elif direction == 3:
        if pos_y > 0 and pos_x > 0:
            return Position(pos_x - 1, pos_y - 1)
    return None


def check_forward_sx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y > 0 and pos_x > 0:
            return Position(pos_x - 1, pos_y - 1)
    elif direction == 1:
        if pos_y < rows - 1 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y + 1)
    elif direction == 2:
        if pos_y > 0 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y - 1)
    elif direction == 3:
        if pos_y < rows - 1 and pos_x > 0:
            return Position(pos_x - 1, pos_y + 1)
    return None


def check_back_dx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y < rows - 1 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y + 1)
    elif direction == 1:
        if pos_y > 0 and pos_x > 0:
            return Position(pos_x - 1, pos_y - 1)
    elif direction == 2:
        if pos_x > 0 and pos_y < rows - 1:
            return Position(pos_x - 1, pos_y + 1)
    elif direction == 3:
        if pos_x < cols - 1 and pos_y > 0:
            return Position(pos_x + 1, pos_y - 1)
    return None


def check_back_sx(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> Optional[Position]:
    if direction == 0:
        if pos_y < rows - 1 and pos_x > 0:
            return Position(pos_x - 1, pos_y + 1)
    elif direction == 1:
        if pos_y > 0 and pos_x < cols - 1:
            return Position(pos_x + 1, pos_y - 1)
    elif direction == 2:
        if pos_x > 0 and pos_y > 0:
            return Position(pos_x - 1, pos_y - 1)
    elif direction == 3:
        if pos_x < cols - 1 and pos_y < rows - 1:
            return Position(pos_x + 1, pos_y + 1)
    return None


def check_around(pos_x: int, pos_y: int, rows: int, cols: int, direction: int) -> list[Optional[Position]]:
    """Le 8 celle adiacenti (indici 0-7); ``None`` dove la cella e' fuori dai bordi."""
    return [
        check_forward(pos_x, pos_y, rows, cols, direction),
        check_back(pos_x, pos_y, rows, cols, direction),
        check_dx(pos_x, pos_y, rows, cols, direction),
        check_sx(pos_x, pos_y, rows, cols, direction),
        check_forward_dx(pos_x, pos_y, rows, cols, direction),
        check_forward_sx(pos_x, pos_y, rows, cols, direction),
        check_back_dx(pos_x, pos_y, rows, cols, direction),
        check_back_sx(pos_x, pos_y, rows, cols, direction),
    ]


# Mappa indice-mossa -> nuova direzione (per le mosse relative alla heading).
_TURN_AROUND = {0: 1, 1: 0, 2: 3, 3: 2}
_TURN_DX = {0: 2, 1: 3, 2: 1, 3: 0}
_TURN_SX = {0: 3, 1: 2, 2: 0, 3: 1}


def resulting_direction(index: int, direction: int) -> int:
    """Direzione risultante dopo aver eseguito la mossa relativa ``index`` (0-7)."""
    if index == 1 or index == 6 or index == 7:  # dietro, dietro-dx, dietro-sx
        return _TURN_AROUND[direction]
    if index == 2:  # destra
        return _TURN_DX[direction]
    if index == 3:  # sinistra
        return _TURN_SX[direction]
    # avanti (0), avanti-dx (4), avanti-sx (5): la direzione non cambia
    return direction


def neighbours_with_directions(
    pos_x: int, pos_y: int, rows: int, cols: int, direction: int
) -> list[tuple[Position, int]]:
    """Celle adiacenti valide con la direzione risultante, per il fallback di contesa."""
    result: list[tuple[Position, int]] = []
    for index, pos in enumerate(check_around(pos_x, pos_y, rows, cols, direction)):
        if pos is not None:
            result.append((pos, resulting_direction(index, direction)))
    return result


def _pick_move_index(rng: random.Random) -> int:
    move = rng.randint(1, 100)
    if move <= FORWARD:
        return 0
    if move <= BACK:
        return 1
    if move <= DX:
        return 2
    if move <= SX:
        return 3
    if move <= FORWARD_DX:
        return 4
    if move <= FORWARD_SX:
        return 5
    if move <= BACK_DX:
        return 6
    return 7


def random_move_index(
    pos_x: int,
    pos_y: int,
    rows: int,
    cols: int,
    direction: int,
    rng: Optional[random.Random] = None,
) -> int:
    """Indice di mossa relativo (0-7) estratto secondo la distribuzione.

    Ripete l'estrazione finche' la cella corrispondente non e' dentro la griglia.
    Usato sia dall'euristica sia dall'ambiente per risolvere una mossa in modo
    uniforme.
    """
    rng = rng or _default_rng
    neighbours = check_around(pos_x, pos_y, rows, cols, direction)
    if all(n is None for n in neighbours):  # griglia troppo piccola: nessuna mossa possibile
        return STAY
    while True:
        index = _pick_move_index(rng)
        if neighbours[index] is not None:
            return index


def random_move(
    pos_x: int,
    pos_y: int,
    rows: int,
    cols: int,
    direction: int,
    rng: Optional[random.Random] = None,
) -> tuple[Optional[Position], int]:
    """Estrae una mossa secondo la distribuzione; ripete finche' non e' in griglia.

    Ritorna ``(nuova_posizione, nuova_direzione)``; ``(None, direction)`` se non
    esiste alcuna cella adiacente valida.
    """
    index = random_move_index(pos_x, pos_y, rows, cols, direction, rng)
    if index == STAY:
        return None, direction
    neighbours = check_around(pos_x, pos_y, rows, cols, direction)
    return neighbours[index], resulting_direction(index, direction)
