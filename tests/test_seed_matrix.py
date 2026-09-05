"""Test della logica condivisa di :class:`SeedMatrix`, esercitata attraverso
l'unica implementazione concreta, :class:`LockSeedMatrix`.
"""

from __future__ import annotations

from antelligent.seeds.seed import Seed
from antelligent.seeds.seed_type import SeedType
from antelligent.simulation.lock_seed_matrix import LockSeedMatrix


def test_place_seed_on_empty_cell_succeeds_and_is_visible() -> None:
    mat = LockSeedMatrix(2, 2)
    seed = Seed(SeedType.BLUE)

    assert not mat.has_seed(0, 0)
    assert mat.place_seed(0, 0, seed)
    assert mat.has_seed(0, 0)


def test_place_seed_on_occupied_cell_fails() -> None:
    mat = LockSeedMatrix(2, 2)
    mat.place_seed(0, 0, Seed(SeedType.BLUE))

    assert not mat.place_seed(0, 0, Seed(SeedType.YELLOW))


def test_pick_seed_on_empty_cell_returns_none() -> None:
    mat = LockSeedMatrix(2, 2)
    assert mat.pick_seed(0, 0, 5) is None


def test_pick_seed_isolated_seed_with_no_neighbors_is_never_picked() -> None:
    # Su una griglia 1x1 non esistono celle vicine: f(c) = 0/0 non e' definita e
    # il caso e' gestito esplicitamente (probabilita' di pick = 0).
    mat = LockSeedMatrix(1, 1)
    mat.place_seed(0, 0, Seed(SeedType.BLUE))

    for _ in range(100):
        assert mat.pick_seed(0, 0, 2) is None
        assert mat.has_seed(0, 0)


def test_drop_seed_into_isolated_empty_cell_never_succeeds() -> None:
    mat = LockSeedMatrix(1, 1)
    seed = Seed(SeedType.BLUE)

    for _ in range(100):
        assert not mat.drop_seed(0, 0, seed, 2)
        assert not mat.has_seed(0, 0)


def test_check_entropy_single_isolated_seed_is_zero() -> None:
    mat = LockSeedMatrix(1, 1)
    mat.place_seed(0, 0, Seed(SeedType.BLUE))

    assert mat.check_entropy(1, 2, True) == 0.0


def test_check_entropy_two_adjacent_seeds_of_same_type_is_zero() -> None:
    mat = LockSeedMatrix(1, 2)
    mat.place_seed(0, 0, Seed(SeedType.BLUE))
    mat.place_seed(1, 0, Seed(SeedType.BLUE))

    assert mat.check_entropy(2, 2, True) == 0.0


def test_check_entropy_two_adjacent_seeds_of_different_type_is_maximal() -> None:
    mat = LockSeedMatrix(1, 2)
    mat.place_seed(0, 0, Seed(SeedType.BLUE))
    mat.place_seed(1, 0, Seed(SeedType.PURPLE))

    assert mat.check_entropy(2, 2, True) == 100.0
