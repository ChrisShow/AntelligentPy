"""Politica tabellare della variante Lumer-Faieta: stato, persistenza, ereditarieta'."""

from __future__ import annotations


from antelligent.actions import STAY, Observation
from antelligent.lumer_reward import LumerRewardConfig
from antelligent.policies import lumer_tabular as lt
from antelligent.policies.lumer_tabular import KIND, LumerQPolicy, load_policy, policy_kind
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


# ------------------------------------------------------------------ stato --- #


def test_manip_state_ignores_the_seed_type() -> None:
    """Con la ricompensa alla Lumer-Faieta il valore dipende solo da ``f``: tenere il
    codice del tipo replicherebbe la stessa decisione su ``k`` righe distinte."""
    policy = LumerQPolicy(3, QConfig())
    a = _obs(seed_here=True, seed_here_type=0, f_here=(0.6, 0.2, 0.2))
    b = _obs(seed_here=True, seed_here_type=1, f_here=(0.2, 0.6, 0.2))
    assert policy._manip_state(a) == policy._manip_state(b)  # noqa: SLF001


def test_manip_state_splits_at_the_lumer_faieta_indifference_point() -> None:
    policy = LumerQPolicy(5, QConfig(f_bins=4))
    f_star = policy.lumer_cfg.indifference_f  # ~0.173, dentro il bin [0, 0.25)
    below = _obs(seed_here=True, f_here=(f_star - 0.02, 0, 0, 0, 0))
    above = _obs(seed_here=True, f_here=(f_star + 0.02, 0, 0, 0, 0))
    assert policy._manip_state(below) != policy._manip_state(above)  # noqa: SLF001


def test_manip_state_space_stays_tiny() -> None:
    policy = LumerQPolicy(5, QConfig(f_bins=4))
    states = {
        policy._manip_state(_obs(carrying=c, carried_type=t, seed_here=not c,  # noqa: SLF001
                                 seed_here_type=t, f_here=tuple(i / 20 for _ in range(5))))
        for c in (False, True) for t in range(5) for i in range(21)
    }
    assert len(states) <= 2 * 4 * 2  # (trasporta) x (bin) x (sopra/sotto f*)


def test_manip_state_differs_from_the_base_encoding() -> None:
    """Guardia: se le due codifiche coincidessero, il marcatore di famiglia nel
    pickle sarebbe inutile e nessuno se ne accorgerebbe."""
    obs = _obs(seed_here=True, seed_here_type=2, f_here=(0.1, 0.2, 0.7))
    assert LumerQPolicy(3)._manip_state(obs) != TabularQPolicy(3)._manip_state(obs)  # noqa: SLF001


# --------------------------------------------------- correzioni ereditate --- #


def test_it_never_chooses_to_stand_still() -> None:
    """Regressione ereditata: con ``STAY`` selezionabile la Q-learning ci cade dentro
    (punto fisso assorbente) e le formiche restano ferme."""
    policy = LumerQPolicy(3, QConfig())
    assert policy.config.allow_stay is False
    for obs in (_obs(), _obs(carrying=True, carried_type=1)):
        policy.q_move[policy._move_state(obs)] = [-1.0] * 8 + [10.0]  # noqa: SLF001
        for _ in range(50):
            assert policy.select_action(obs, _ant()).move_index != STAY


def test_frozen_policy_keeps_the_residual_noise() -> None:
    policy = LumerQPolicy(3, QConfig())
    obs = _obs()
    state = policy._move_state(obs)  # noqa: SLF001
    policy.q_move[state] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    policy.freeze()
    assert policy.epsilon == policy.config.inference_epsilon > 0.0
    assert {policy.select_action(obs, _ant()).move_index for _ in range(400)} != {1}


def test_move_state_is_the_inherited_compact_one() -> None:
    policy = LumerQPolicy(5, QConfig())
    base = dict(carrying=True, seed_here=False, seed_here_type=0, best_dir=3)
    states = {
        policy._move_state(_obs(direction=d, carried_type=t, **base))  # noqa: SLF001
        for d in range(4) for t in range(5)
    }
    assert len(states) == 1


# ------------------------------------------------------------ persistenza --- #


def test_save_and_load_round_trip(tmp_path) -> None:
    policy = LumerQPolicy(3, QConfig(), lumer=LumerRewardConfig(drop_scale=2.5))
    obs = _obs(seed_here=True, f_here=(0.1, 0.0, 0.0))
    policy.select_action(obs, _ant())
    path = tmp_path / "policy_lumer.pkl"
    policy.save(path)

    loaded = LumerQPolicy.load(path)
    assert isinstance(loaded, LumerQPolicy)
    assert loaded.seed_types == 3
    assert loaded.lumer_cfg.drop_scale == 2.5
    assert loaded.q_move == policy.q_move
    assert loaded.q_manip == policy.q_manip
    assert loaded.state_version == lt._STATE_VERSION  # noqa: SLF001


def test_the_pickle_declares_its_family(tmp_path) -> None:
    base, lumer = tmp_path / "base.pkl", tmp_path / "lumer.pkl"
    TabularQPolicy(3, QConfig()).save(base)
    LumerQPolicy(3, QConfig()).save(lumer)
    assert policy_kind(base) == "tabular"
    assert policy_kind(lumer) == KIND


def test_load_policy_dispatches_on_the_family(tmp_path) -> None:
    base, lumer = tmp_path / "base.pkl", tmp_path / "lumer.pkl"
    TabularQPolicy(3, QConfig()).save(base)
    LumerQPolicy(3, QConfig()).save(lumer)
    assert type(load_policy(base)) is TabularQPolicy
    assert type(load_policy(lumer)) is LumerQPolicy


def test_loading_the_wrong_family_warns_loudly(tmp_path, caplog) -> None:
    path = tmp_path / "base.pkl"
    TabularQPolicy(3, QConfig()).save(path)
    with caplog.at_level("WARNING", logger=lt.__name__):
        LumerQPolicy.load(path)
    assert "riaddestrala" in caplog.text.lower()
    assert "train_lumer" in caplog.text


def test_loading_an_old_state_encoding_warns(tmp_path, caplog) -> None:
    import pickle

    path = tmp_path / "old.pkl"
    with path.open("wb") as handle:
        pickle.dump({"policy_kind": KIND, "state_version": 0, "seed_types": 3,
                     "config": QConfig(), "q_move": {}, "q_manip": {}}, handle)
    with caplog.at_level("WARNING", logger=lt.__name__):
        policy = LumerQPolicy.load(path)
    assert policy.state_version == 0
    assert "riaddestrala" in caplog.text.lower()


# ----------------------------------------------------------- integrazione --- #


def test_a_short_training_run_learns_something(tmp_path) -> None:
    from antelligent.train_lumer import build_parser, train

    out = tmp_path / "p.pkl"
    args = build_parser().parse_args([
        "--episodes", "2", "--max-ticks", "40", "--entropy-every", "20",
        "--out", str(out), "--log-every", "5",
    ])
    train(args)
    assert out.is_file()
    assert out.with_name("p_log.csv").is_file()
    loaded = load_policy(out)
    assert isinstance(loaded, LumerQPolicy)
    assert loaded.q_manip, "nessuno stato di manipolazione visitato"
    assert any(v != 0.0 for row in loaded.q_manip.values() for v in row)


def test_cli_defaults_come_from_the_reward_config() -> None:
    """Guardia: i default della CLI non devono poter divergere dalla dataclass
    (era gia' successo: ``drop_scale`` 2.0 nella config, 1.0 sulla riga di comando)."""
    from antelligent.train_lumer import build_parser

    args = build_parser().parse_args([])
    cfg = LumerRewardConfig()
    for field in ("mode", "kp", "kd", "pick_scale", "drop_scale",
                  "decline_scale", "shaping_scale", "wall_penalty"):
        assert getattr(args, field) == getattr(cfg, field), field
