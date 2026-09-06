"""Politica tabellare: aggiornamento Q, greedy congelato, persistenza."""

from __future__ import annotations

import random

from antelligent.actions import STAY, Action, Manipulation, Observation, Transition
from antelligent.policies.tabular import QConfig, TabularQPolicy
from antelligent.simulation.ant import Ant
from antelligent.utilities.position import Position


def _obs(carrying=False, carried_type=0, seed_here=False, seed_here_type=0, best_dir=STAY,
         direction=0, f_here=(0.0, 0.0, 0.0)) -> Observation:
    return Observation(
        carrying=carrying,
        carried_type=carried_type,
        seed_here=seed_here,
        seed_here_type=seed_here_type if seed_here else 0,
        can_pick=(not carrying) and seed_here,
        can_drop=carrying and not seed_here,
        f_here=f_here,
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
    policy = TabularQPolicy(3, QConfig(inference_epsilon=0.0))  # greedy pura
    obs = _obs()
    state = policy._move_state(obs)  # noqa: SLF001
    policy.q_move[state] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    policy.freeze()
    for _ in range(20):
        assert policy.select_action(obs, _ant()).move_index == 1
    policy.record(Transition(obs, Action(Manipulation.NOOP, 1), 99.0, obs, True))
    assert policy.q_move[state][1] == 5.0  # nessun aggiornamento da congelata


def test_frozen_policy_keeps_a_little_noise_by_default_but_never_learns() -> None:
    """Su osservazioni aliasate una politica deterministica cade in cicli limite:
    la congelata mantiene ``inference_epsilon`` di rumore, ma non impara piu'."""
    policy = TabularQPolicy(3, QConfig())
    assert policy.config.inference_epsilon > 0.0
    obs = _obs()
    state = policy._move_state(obs)  # noqa: SLF001
    policy.q_move[state] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    policy.freeze()
    assert policy.epsilon == policy.config.inference_epsilon
    chosen = {policy.select_action(obs, _ant()).move_index for _ in range(400)}
    assert chosen != {1}, "nessuna variabilita': la politica non puo' uscire dai cicli"
    policy.record(Transition(obs, Action(Manipulation.NOOP, 1), 99.0, obs, True))
    assert policy.q_move[state][1] == 5.0  # congelata = nessun apprendimento


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


def test_pick_state_depends_on_the_seed_under_the_ant() -> None:
    # Stessa formica (non trasporta) su due semi di tipo diverso, con densita'
    # locale diversa per quei tipi: gli stati di manipolazione devono differire.
    policy = TabularQPolicy(3, QConfig())
    base = dict(carrying=False, carried_type=0, seed_here=True, best_dir=8, direction=0)
    obs_a = Observation(**base, seed_here_type=0, f_here=(0.9, 0.0, 0.1),
                        can_pick=True, can_drop=False, blocked=(False,) * 8, window=((0,),))
    obs_b = Observation(**base, seed_here_type=1, f_here=(0.9, 0.0, 0.1),
                        can_pick=True, can_drop=False, blocked=(False,) * 8, window=((0,),))
    assert policy._manip_state(obs_a) != policy._manip_state(obs_b)  # noqa: SLF001


def test_policy_never_chooses_to_stand_still_by_default() -> None:
    """Regressione: con ``STAY`` selezionabile la Q-learning ci cade dentro
    (nessun rischio di ``contested_penalty`` -> azione piu' economica ovunque),
    e le formiche restavano ferme, soprattutto mentre trasportavano un seme.
    """
    policy = TabularQPolicy(3, QConfig())
    assert policy.config.allow_stay is False
    obs_free, obs_carrying = _obs(), _obs(carrying=True, carried_type=1)
    for obs in (obs_free, obs_carrying):
        state = policy._move_state(obs)  # noqa: SLF001
        policy.q_move[state] = [-1.0] * 8 + [10.0]  # STAY apparentemente ottimo
        for _ in range(50):
            assert policy.select_action(obs, _ant()).move_index != STAY


def test_stay_can_be_re_enabled_for_ablations() -> None:
    policy = TabularQPolicy(3, QConfig(allow_stay=True, epsilon_start=0.0, epsilon_end=0.0))
    obs = _obs()
    policy.q_move[policy._move_state(obs)] = [-1.0] * 8 + [10.0]  # noqa: SLF001
    assert policy.select_action(obs, _ant()).move_index == STAY


def test_bootstrap_ignores_the_stay_column_when_it_is_disabled() -> None:
    # Q(STAY) gonfiato nello stato successivo non deve entrare nel target.
    policy = TabularQPolicy(3, QConfig(alpha=1.0, gamma=1.0, epsilon_start=0.0, epsilon_end=0.0))
    obs = _obs()
    next_state = policy._move_state(obs)  # noqa: SLF001
    policy.q_move[next_state] = [0.0] * 8 + [100.0]
    action = policy.select_action(obs, _ant())
    policy.record(Transition(obs, action, reward=0.0, next_obs=obs, done=False))
    assert policy.q_move[next_state][action.move_index] == 0.0  # non contaminato dai 100


def test_manip_state_can_express_the_decision_boundary() -> None:
    """Il segno della ricompensa si ribalta al livello del caso ``1/k``, che non cade
    su un confine dei bin: senza un bit dedicato lo stato non distingue "pick
    conviene" da "pick e' punito" dentro lo stesso bin."""
    policy = TabularQPolicy(5, QConfig(f_bins=4))  # pivot 0.2, bin 0 = [0, 0.25)
    below = _obs(seed_here=True, f_here=(0.10, 0.0, 0.0, 0.0, 0.0))  # pick premiato
    above = _obs(seed_here=True, f_here=(0.24, 0.0, 0.0, 0.0, 0.0))  # pick punito
    assert policy._manip_state(below) != policy._manip_state(above)  # noqa: SLF001


def test_manip_state_groups_values_on_the_same_side_of_chance() -> None:
    policy = TabularQPolicy(5, QConfig(f_bins=4))
    a = _obs(seed_here=True, f_here=(0.02, 0.0, 0.0, 0.0, 0.0))
    b = _obs(seed_here=True, f_here=(0.10, 0.0, 0.0, 0.0, 0.0))
    assert policy._manip_state(a) == policy._manip_state(b)  # noqa: SLF001


def test_move_state_ignores_heading_and_carried_type() -> None:
    """Le mosse sono gia' relative all'heading: metterlo nello stato frammenterebbe
    la stessa situazione su 4 tabelle distinte (e il tipo trasportato e' gia'
    riassunto da ``best_dir``)."""
    policy = TabularQPolicy(5, QConfig())
    base = dict(carrying=True, seed_here=False, seed_here_type=0, best_dir=3)
    states = {
        policy._move_state(_obs(direction=d, carried_type=t, **base))  # noqa: SLF001
        for d in range(4) for t in range(5)
    }
    assert len(states) == 1, f"stato frammentato su heading/tipo: {states}"


def test_move_state_distinguishes_the_target_direction() -> None:
    policy = TabularQPolicy(3, QConfig())
    a = policy._move_state(_obs(best_dir=0))  # noqa: SLF001
    b = policy._move_state(_obs(best_dir=5))  # noqa: SLF001
    assert a != b, "la direzione del bersaglio deve entrare nello stato"


def test_loading_a_policy_with_an_old_state_encoding_warns(tmp_path, caplog) -> None:
    import pickle

    from antelligent.policies import tabular as tab

    stale = tmp_path / "old.pkl"
    with stale.open("wb") as handle:
        pickle.dump({"seed_types": 3, "config": QConfig(), "q_move": {}, "q_manip": {},
                     "episode": 1, "epsilon": 0.02}, handle)  # senza state_version -> v1
    with caplog.at_level("WARNING", logger=tab.__name__):
        policy = TabularQPolicy.load(stale)
    assert policy.state_version == 1
    assert "riaddestrala" in caplog.text.lower()


def test_saved_policy_records_the_current_state_encoding(tmp_path) -> None:
    from antelligent.policies import tabular as tab

    path = tmp_path / "p.pkl"
    TabularQPolicy(3, QConfig()).save(path)
    loaded = TabularQPolicy.load(path)
    assert loaded.state_version == tab._STATE_VERSION


def test_manip_learning_toggles_pick(monkeypatch) -> None:
    policy = TabularQPolicy(2, QConfig(epsilon_start=0.0, epsilon_end=0.0))
    obs = _obs(seed_here=True)  # can_pick True
    m_state = policy._manip_state(obs)  # noqa: SLF001
    policy.q_manip[m_state] = [0.0, 10.0]  # "act" molto meglio di "no-op"
    action = policy.select_action(obs, _ant())
    assert action.manipulation == Manipulation.PICK
