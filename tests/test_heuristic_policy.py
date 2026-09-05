"""Politica euristica: probabilita' di Lumer-Faieta e casi limite."""

from __future__ import annotations

import random

from antelligent.actions import Manipulation
from antelligent.environment import Environment
from antelligent.policies.heuristic import HeuristicPolicy
from antelligent.seeds.seed import Seed
from antelligent.seeds.seed_type import SeedType
from antelligent.simulation.lock_seed_matrix import LockSeedMatrix
from antelligent.world import InitialState


def _single_ant_env(rows, cols, seed_cells, seed_types=2) -> Environment:
    init = InitialState(
        rows=rows, cols=cols, seed_types=seed_types, n_seeds=len(seed_cells), n_ants=1,
        master_seed=0, seed_cells=tuple(seed_cells), ant_cells=((0, 0),), ant_dirs=(0,),
    )
    return Environment.from_initial_state(init)


def test_isolated_seed_is_never_picked_through_policy() -> None:
    env = _single_ant_env(1, 1, [(0, 0, 0)])
    policy = HeuristicPolicy(env.matrix, 2, rng=random.Random(1))
    ant = env.ants[0]
    for _ in range(200):
        action = policy.select_action(env.observe(ant), ant)
        assert action.manipulation == Manipulation.NOOP


def test_pick_probability_is_maximal_when_neighbours_are_all_other_types() -> None:
    matrix = LockSeedMatrix(1, 2)
    matrix.set_seed_at(0, 0, Seed(SeedType.BLUE))
    matrix.set_seed_at(1, 0, Seed(SeedType.PURPLE))
    # BLUE (code 0) circondato solo da PURPLE -> f = 0 -> P_pick = (0.1/0.1)^2*100 = 100
    assert matrix.pick_probability(0, 0, 0, 2) == 100.0
    # nessun vicino -> 0
    assert LockSeedMatrix(1, 1).pick_probability(0, 0, 0, 2) == 0.0


def test_drop_probability_grows_with_same_type_neighbours() -> None:
    matrix = LockSeedMatrix(1, 3)
    matrix.set_seed_at(0, 0, Seed(SeedType.BLUE))
    matrix.set_seed_at(2, 0, Seed(SeedType.PURPLE))
    p_low = matrix.drop_probability(1, 0, 0, 2)   # 1 vicino BLUE, 1 PURPLE -> f = 0.5
    matrix.set_seed_at(2, 0, Seed(SeedType.BLUE))
    p_high = matrix.drop_probability(1, 0, 0, 2)  # 2 vicini BLUE -> f = 1.0
    assert 0 < p_low < p_high
