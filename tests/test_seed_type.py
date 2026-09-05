"""Test dei tipi di seme (``antelligent.seeds.seed_type``)."""

from __future__ import annotations

import random

from antelligent.seeds.seed_type import SeedType


def test_codes_match_original_ordering() -> None:
    assert SeedType.BLUE.code == 0
    assert SeedType.PURPLE.code == 1
    assert SeedType.GRAY.code == 2
    assert SeedType.ORANGE.code == 3
    assert SeedType.YELLOW.code == 4


def test_from_code_round_trip() -> None:
    for member in SeedType:
        assert SeedType.from_code(member.code) is member


def test_random_with_count_one_always_returns_blue() -> None:
    rng = random.Random()
    for _ in range(100):
        assert SeedType.random(rng, 1) is SeedType.BLUE


def test_random_with_count_three_only_returns_first_three_types() -> None:
    rng = random.Random()
    allowed = {SeedType.BLUE, SeedType.PURPLE, SeedType.GRAY}
    for _ in range(200):
        assert SeedType.random(rng, 3) in allowed


def test_random_with_out_of_range_count_never_raises() -> None:
    rng = random.Random()
    for _ in range(50):
        SeedType.random(rng, 0)
        SeedType.random(rng, 999)
