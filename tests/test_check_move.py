"""Test delle regole di movimento (``antelligent.utilities.check_move``)."""

from __future__ import annotations

from antelligent.utilities import check_move
from antelligent.utilities.position import Position


def test_check_forward_at_grid_edge_is_none() -> None:
    # direzione 0 = UP: al bordo superiore (pos_y = 0) non esiste una cella "avanti".
    assert check_move.check_forward(1, 0, 3, 3, 0) is None


def test_check_forward_away_from_edge_returns_cell_above() -> None:
    forward = check_move.check_forward(1, 1, 3, 3, 0)
    assert forward is not None
    assert forward.x == 1
    assert forward.y == 0


def test_check_around_corner_matches_expected_neighbors() -> None:
    neighbors = check_move.check_around(0, 0, 3, 3, 0)

    assert len(neighbors) == 8
    assert neighbors[0] is None            # forward: bordo superiore
    assert neighbors[1] == Position(0, 1)  # back
    assert neighbors[2] == Position(1, 0)  # dx
    assert neighbors[3] is None            # sx: bordo sinistro
    assert neighbors[4] is None            # forward-dx: bordo superiore
    assert neighbors[5] is None            # forward-sx: entrambi i bordi
    assert neighbors[6] == Position(1, 1)  # back-dx
    assert neighbors[7] is None            # back-sx: bordo sinistro


def test_random_move_from_center_always_returns_in_bounds_position() -> None:
    for _ in range(500):
        new_position, new_direction = check_move.random_move(1, 1, 3, 3, 0)
        assert new_position is not None
        assert 0 <= new_position.x < 3
        assert 0 <= new_position.y < 3
        assert 0 <= new_direction < 4


def test_random_move_index_is_a_valid_in_bounds_neighbour() -> None:
    for _ in range(500):
        index = check_move.random_move_index(1, 1, 3, 3, 0)
        assert 0 <= index < 8
        assert check_move.check_around(1, 1, 3, 3, 0)[index] is not None


def test_random_move_index_on_1x1_grid_is_stay() -> None:
    assert check_move.random_move_index(0, 0, 1, 1, 0) == check_move.STAY
    assert check_move.random_move(0, 0, 1, 1, 0)[0] is None
