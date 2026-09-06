"""Ricompensa "alla Lumer-Faieta": formule pure e ambiente che le applica."""

from __future__ import annotations

import pytest

from antelligent.actions import STAY, Action, Manipulation
from antelligent.environment import RewardConfig
from antelligent.environment_lumer import LumerFaietaEnvironment
from antelligent.lumer_reward import (
    DROP,
    KD_DROP,
    KP_PICK,
    NONE,
    PICK,
    LumerRewardConfig,
    decline_value,
    drop_probability,
    manip_value,
    pick_probability,
    potential,
    probabilities,
)
from antelligent.seeds.seed import Seed
from antelligent.seeds.seed_type import SeedType
from antelligent.simulation import seed_matrix as sm
from antelligent.utilities import check_move
from antelligent.world import InitialState

_NO_COST = RewardConfig(step_penalty=0.0, contested_penalty=0.0, invalid_penalty=0.0)


def _only_move_index(env, ant) -> int:
    """Unico indice di mossa valido (griglie a una riga / angoli)."""
    neigh = check_move.check_around(
        ant.position.x, ant.position.y, env.matrix.rows, env.matrix.cols, ant.direction)
    return next(i for i, p in enumerate(neigh) if p is not None)


def _env(rows, cols, seed_cells=(), ant_cells=(), seed_types=2, lumer=None, reward=_NO_COST):
    init = InitialState(
        rows=rows, cols=cols, seed_types=seed_types,
        n_seeds=len(seed_cells), n_ants=len(ant_cells), master_seed=0,
        seed_cells=tuple(seed_cells), ant_cells=tuple(ant_cells),
        ant_dirs=tuple(0 for _ in ant_cells),
    )
    return LumerFaietaEnvironment.from_initial_state(
        init, reward=reward, lumer=lumer or LumerRewardConfig(shaping_scale=0.0),
    )


# ---------------------------------------------------------------- formule --- #


def test_constants_match_the_heuristic_ones() -> None:
    """Guardia: se l'euristica cambia kp/kd, la ricompensa deve cambiare con lei."""
    assert (KP_PICK, KD_DROP) == (sm._KP_PICK, sm._KD_DROP)  # noqa: SLF001


@pytest.mark.parametrize("same,other", [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
def test_probabilities_are_the_heuristic_ones_rescaled(same, other) -> None:
    """Stesso campo, stesse formule: l'euristica lavora su scala 0-100, qui in [0, 1]."""
    neighbours = [(0, 0), (2, 0), (0, 2), (2, 2)]
    cells = [(1, 1, 0)] + [
        (x, y, 0 if i < same else 1) for i, (x, y) in enumerate(neighbours[: same + other])
    ]
    env = _env(3, 3, seed_cells=cells, ant_cells=[(1, 1)])
    f = same / (same + other)
    assert pick_probability(f) == pytest.approx(env.matrix.pick_probability(1, 1, 0, 2) / 100)
    assert drop_probability(f) == pytest.approx(env.matrix.drop_probability(1, 1, 0, 2) / 100)


def test_advantage_is_zero_exactly_at_the_indifference_point() -> None:
    cfg = LumerRewardConfig(mode="advantage")
    f_star = cfg.indifference_f
    assert f_star == pytest.approx((KP_PICK * KD_DROP) ** 0.5)
    assert manip_value(PICK, f_star, cfg) == pytest.approx(0.0, abs=1e-12)
    assert manip_value(DROP, f_star, cfg) == pytest.approx(0.0, abs=1e-12)


def test_advantage_changes_sign_around_the_indifference_point() -> None:
    cfg = LumerRewardConfig(mode="advantage")
    below, above = cfg.indifference_f - 0.05, cfg.indifference_f + 0.05
    assert manip_value(PICK, below, cfg) > 0 > manip_value(PICK, above, cfg)
    assert manip_value(DROP, above, cfg) > 0 > manip_value(DROP, below, cfg)


@pytest.mark.parametrize("mode", ["advantage", "centered"])
def test_value_is_monotone_in_the_local_density(mode) -> None:
    cfg = LumerRewardConfig(mode=mode)
    grid = [i / 20 for i in range(21)]
    picks = [manip_value(PICK, f, cfg) for f in grid]
    drops = [manip_value(DROP, f, cfg) for f in grid]
    assert picks == sorted(picks, reverse=True)  # piu' simili intorno -> raccogliere conviene meno
    assert drops == sorted(drops)                # piu' simili intorno -> posare conviene di piu'


def test_raw_mode_never_punishes_and_therefore_invites_churn() -> None:
    cfg = LumerRewardConfig(mode="raw")
    assert all(manip_value(PICK, f / 10, cfg) >= 0 for f in range(11))
    assert all(manip_value(DROP, f / 10, cfg) >= 0 for f in range(11))


def test_centered_mode_flips_at_a_fifty_percent_probability() -> None:
    cfg = LumerRewardConfig(mode="centered")
    f_half = KD_DROP * (0.5 ** 0.5) / (1 - 0.5 ** 0.5)  # P_drop(f) = 0.5
    assert manip_value(DROP, f_half, cfg) == pytest.approx(0.0, abs=1e-9)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        LumerRewardConfig(mode="qualcosa")


def test_empty_neighbourhood_extends_the_formulas_by_default() -> None:
    cfg = LumerRewardConfig()
    assert manip_value(PICK, 0.0, cfg, isolated=True) == pytest.approx(1.0)
    assert manip_value(DROP, 0.0, cfg, isolated=True) == pytest.approx(-1.0)


def test_isolated_guard_reproduces_the_heuristic_zero() -> None:
    cfg = LumerRewardConfig(isolated_guard=True, mode="advantage")
    assert probabilities(0.0, cfg, isolated=True) == (0.0, 0.0)
    assert manip_value(PICK, 0.0, cfg, isolated=True) == 0.0
    assert manip_value(DROP, 0.0, cfg, isolated=True) == 0.0


def test_potential_is_zero_where_there_is_nothing_to_do() -> None:
    assert potential(NONE, 0.7, LumerRewardConfig()) == 0.0


# ------------------------------------------------------------- ambiente --- #


def test_drop_that_joins_same_type_seeds_is_rewarded() -> None:
    env = _env(1, 4, seed_cells=[(0, 0, 1), (1, 0, 0), (3, 0, 0)], ant_cells=[(2, 0)])
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)  # codice 0, come i vicini
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is True
    assert result.info["lf_kind"] == DROP
    assert result.info["lf_f"] == 1.0
    cfg = env.lumer_cfg
    assert result.reward == pytest.approx(manip_value(DROP, 1.0, cfg) * cfg.drop_scale)


def test_drop_in_an_empty_desert_is_maximally_punished() -> None:
    env = _env(3, 3, ant_cells=[(1, 1)])
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is True
    assert result.reward == pytest.approx(-1.0 * env.lumer_cfg.drop_scale)


def test_the_drop_signal_has_a_much_smaller_range_than_the_pick_one() -> None:
    """Motivo strutturale del default ``drop_scale = 2``: in ``centered`` il pick
    arriva a +1.0 (seme isolato), il drop solo a +0.18 (seme fra soli simili)."""
    cfg = LumerRewardConfig(pick_scale=1.0, drop_scale=1.0)
    assert manip_value(PICK, 0.0, cfg) == pytest.approx(1.0)
    assert manip_value(DROP, 1.0, cfg) == pytest.approx(0.183, abs=0.005)


def test_pick_of_a_misplaced_seed_is_rewarded_and_of_a_clustered_one_punished() -> None:
    misplaced = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1)],
                     ant_cells=[(1, 1)])
    clustered = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 2, 0)],
                     ant_cells=[(1, 1)])
    assert misplaced.step(misplaced.ants[0], Action(Manipulation.PICK, STAY)).reward > 0
    assert clustered.step(clustered.ants[0], Action(Manipulation.PICK, STAY)).reward < 0


def test_declining_a_good_pick_is_punished_and_declining_a_bad_one_is_free() -> None:
    """Il rifiuto e' **unilaterale**: rifiutare quel che conviene costa, rifiutare
    quel che non conviene vale zero. La versione simmetrica pagherebbe una formica
    ogni tick per stare ferma sul posto giusto — una rendita di posizione."""
    good = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1)],
                ant_cells=[(1, 1)])
    bad = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 2, 0)],
               ant_cells=[(1, 1)])
    assert good.step(good.ants[0], Action(Manipulation.NOOP, STAY)).reward < 0
    assert bad.step(bad.ants[0], Action(Manipulation.NOOP, STAY)).reward == 0.0


def test_two_sided_decline_pays_for_standing_on_a_good_cluster() -> None:
    """Ablazione: e' la rendita di posizione che rende patologico il rifiuto simmetrico."""
    cfg = LumerRewardConfig(decline_two_sided=True, shaping_scale=0.0)
    env = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 2, 0)],
               ant_cells=[(1, 1)], lumer=cfg)
    assert env.step(env.ants[0], Action(Manipulation.NOOP, STAY)).reward > 0


def test_decline_value_is_never_positive_by_default() -> None:
    cfg = LumerRewardConfig()
    assert all(decline_value(PICK, f / 20, cfg) <= 0 for f in range(21))
    assert all(decline_value(DROP, f / 20, cfg) <= 0 for f in range(21))


def test_choosing_a_direction_outside_the_grid_is_charged() -> None:
    """Puntare fuori griglia e' l'unica azione a costo zero del problema: la fisica
    la tratta come "resta ferma" senza contesa. Non addebitarla riapre dalla porta di
    servizio il punto fisso assorbente chiuso da ``allow_stay = False``: misurato,
    la politica ci finiva dentro nel 39-44 % dei passi."""
    reward = RewardConfig(step_penalty=0.0, contested_penalty=0.0, invalid_penalty=0.0)
    env = _env(1, 2, ant_cells=[(0, 0)], reward=reward)
    ant = env.ants[0]
    neigh = check_move.check_around(0, 0, env.matrix.rows, env.matrix.cols, ant.direction)
    outside = next(i for i, p in enumerate(neigh) if p is None)
    result = env.step(ant, Action(Manipulation.NOOP, outside))
    assert result.info["moved"] is False and result.info["contested"] is False
    assert result.info["lf_wall"] is True
    assert result.reward == pytest.approx(-env.lumer_cfg.wall_penalty)


def test_a_legal_move_is_not_charged_as_a_wall() -> None:
    reward = RewardConfig(step_penalty=0.0, contested_penalty=0.0, invalid_penalty=0.0)
    env = _env(1, 2, ant_cells=[(0, 0)], reward=reward)
    ant = env.ants[0]
    result = env.step(ant, Action(Manipulation.NOOP, _only_move_index(env, ant)))
    assert result.info["lf_wall"] is False
    assert result.reward == 0.0


def test_declining_is_free_when_decline_scale_is_zero() -> None:
    cfg = LumerRewardConfig(decline_scale=0.0, shaping_scale=0.0)
    env = _env(3, 3, seed_cells=[(1, 1, 0), (1, 0, 1)], ant_cells=[(1, 1)], lumer=cfg)
    assert env.step(env.ants[0], Action(Manipulation.NOOP, STAY)).reward == 0.0


def test_base_costs_of_the_parent_environment_still_apply() -> None:
    env = _env(1, 2, ant_cells=[(0, 0), (1, 0)], reward=RewardConfig(
        step_penalty=0.01, contested_penalty=0.05, invalid_penalty=0.05))
    ant = env.ants[0]
    result = env.step(ant, Action(Manipulation.NOOP, _only_move_index(env, ant)))
    assert result.info["contested"] is True
    assert result.reward == pytest.approx(-0.06)


def test_an_impossible_manipulation_pays_only_the_invalid_penalty() -> None:
    # trasporta ma la cella e' occupata da un seme: il drop non e' possibile
    env = _env(3, 3, seed_cells=[(1, 1, 0)], ant_cells=[(1, 1)],
               reward=RewardConfig(step_penalty=0.0, contested_penalty=0.0, invalid_penalty=0.05))
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is False
    assert result.info["lf_kind"] == NONE
    assert result.reward == pytest.approx(-0.05)


def test_shaping_rewards_getting_closer_to_a_good_drop_site() -> None:
    """Il termine potenziale e' quello che insegna *dove* posare: muoversi verso una
    cella con P_drop piu' alta paga gia' prima di deporre il seme."""
    cfg = LumerRewardConfig(shaping_scale=1.0)
    # (0,0) non ha vicini con semi (P_drop = 0); (1,0) confina con un seme
    # dello stesso tipo del carico (P_drop massima): avvicinarsi deve pagare.
    env = _env(1, 5, seed_cells=[(2, 0, 0), (4, 0, 0)], ant_cells=[(0, 0)], lumer=cfg)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    assert env._potential(ant) == 0.0  # noqa: SLF001
    result = env.step(ant, Action(Manipulation.NOOP, _only_move_index(env, ant)))
    assert ant.position.x == 1
    assert result.info["lf_shaping"] > 0.0


def test_shaping_telescopes_along_any_path() -> None:
    """Con gamma=1 la somma dello shaping su un cammino qualunque vale
    ``Phi(fine) - Phi(inizio)``: dipende solo dagli estremi, non da come ci si
    arriva. E' la proprieta' che rende lo shaping *potenziale* e quindi
    invariante rispetto alla politica ottima (Ng-Harada-Russell 1999): nessuna
    formica puo' guadagnare girando in tondo."""
    cfg = LumerRewardConfig(shaping_scale=1.0, gamma=1.0)
    env = _env(4, 4, seed_cells=[(0, 0, 0), (3, 3, 1), (1, 3, 0)], ant_cells=[(1, 1)], lumer=cfg)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    phi_start = env._potential(ant)  # noqa: SLF001
    total = sum(
        env.step(ant, Action(Manipulation.NOOP, env.rng.randrange(8))).info["lf_shaping"]
        for _ in range(40)
    )
    assert total == pytest.approx(env._potential(ant) - phi_start, abs=1e-9)  # noqa: SLF001


def test_from_initial_state_carries_the_lumer_config() -> None:
    cfg = LumerRewardConfig(drop_scale=3.0)
    env = _env(2, 2, ant_cells=[(0, 0)], lumer=cfg)
    assert isinstance(env, LumerFaietaEnvironment)
    assert env.lumer_cfg is cfg


def test_drop_scale_weighs_the_placement_more_than_the_pickup() -> None:
    cfg = LumerRewardConfig(drop_scale=2.0, shaping_scale=0.0)
    assert LumerRewardConfig().drop_scale == 2.0  # default misurato (docs/uso.md §3bis-g)
    env = _env(1, 4, seed_cells=[(0, 0, 1), (1, 0, 0), (3, 0, 0)], ant_cells=[(2, 0)], lumer=cfg)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.reward == pytest.approx(2.0 * manip_value(DROP, 1.0, cfg))
