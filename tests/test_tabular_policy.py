"""Politica tabellare: aggiornamento Q, greedy congelato, persistenza."""

from __future__ import annotations

import random

from antelligent.actions import Action, Manipulation, Observation, Transition
from antelligent.policies.tabular import QConfig, TabularQPolicy
from antelligent.simulation.ant import Ant
from antelligent.utilities.position import Position


def _obs(carrying=False, carried_type=0, seed_here=False, best_dir=8, direction=0) -> Observation:
    return Observation(
        carrying=carrying,
        carried_type=carried_type,
        seed_here=seed_here,
        can_pick=(not carrying) and seed_here,
        can_drop=carrying and not seed_here,
        f_here=(0.0, 0.0, 0.0),
        blocked=(False,) * 8,
        best_dir=best_dir,
        direction=direction,
        window=((0,),),
    )


def _ant() -> Ant:
    return Ant(0, Position(0, 0), 0)


def test_positive_reward_raises_the_chosen_move_value() -> None:
    policy = TabularQPolicy(3, QConfig(epsilon_start=0.0, epsilon_end=0.0))
    obs = _obs()
    action = policy.select_action(obs, _ant())
    policy.record(Transition(obs, action, reward=1.0, next_obs=obs, done=True))
    state = policy._move_state(obs)  # noqa: SLF001
    assert policy.q_move[state][action.move_index] > 0.0


def test_frozen_policy_is_greedy_and_does_not_learn() -> None:
    policy = TabularQPolicy(3, QConfig())
    obs = _obs()
    state = policy._move_state(obs)  # noqa: SLF001
    policy.q_move[state] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    policy.freeze()
    for _ in range(20):
        assert policy.select_action(obs, _ant()).move_index == 1
    policy.record(Transition(obs, Action(Manipulation.NOOP, 1), 99.0, obs, True))
    assert policy.q_move[state][1] == 5.0  # nessun aggiornamento da congelata


def test_save_and_load_round_trip(tmp_path) -> None:
    policy = TabularQPolicy(3, QConfig())
    obs = _obs()
    policy.record  # touch
    action = policy.select_action(obs, _ant())
    policy.record(Transition(obs, action, 1.0, obs, True))
    path = tmp_path / "policy.pkl"
    policy.save(path)

    loaded = TabularQPolicy.load(path)
    assert loaded.seed_types == 3
    assert loaded.q_move == policy.q_move
    assert loaded.q_manip == policy.q_manip


def test_manip_learning_toggles_pick(monkeypatch) -> None:
    policy = TabularQPolicy(2, QConfig(epsilon_start=0.0, epsilon_end=0.0))
    obs = _obs(seed_here=True)  # can_pick True
    m_state = policy._manip_state(obs)  # noqa: SLF001
    policy.q_manip[m_state] = [0.0, 10.0]  # "act" molto meglio di "no-op"
    action = policy.select_action(obs, _ant())
    assert action.manipulation == Manipulation.PICK
