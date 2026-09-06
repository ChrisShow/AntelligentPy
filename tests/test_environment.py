"""Ambiente: fisica di occupazione/contesa, contatori, entropia, tick, copie gemelle."""

from __future__ import annotations

import pytest

from antelligent.actions import STAY, Action, Manipulation
from antelligent.environment import Environment, RewardConfig
from antelligent.policies.heuristic import HeuristicPolicy
from antelligent.seeds.seed import Seed
from antelligent.seeds.seed_type import SeedType
from antelligent.simulation.ant import Ant
from antelligent.utilities.position import Position
from antelligent.world import InitialState, build_initial_state


def _env(rows: int, cols: int, seed_cells=(), ant_cells=(), ant_dirs=None, seed_types=2) -> Environment:
    ant_dirs = ant_dirs or tuple(0 for _ in ant_cells)
    init = InitialState(
        rows=rows, cols=cols, seed_types=seed_types,
        n_seeds=len(seed_cells), n_ants=len(ant_cells), master_seed=0,
        seed_cells=tuple(seed_cells), ant_cells=tuple(ant_cells), ant_dirs=ant_dirs,
    )
    return Environment.from_initial_state(init)


def test_ant_moves_into_free_cell_and_releases_the_old_one() -> None:
    env = _env(1, 2, ant_cells=[(0, 0)])
    ant = env.ants[0]
    env.step(ant, Action(Manipulation.NOOP, _only_move_index(env, ant)))
    assert ant.position == Position(1, 0)
    assert env.total_moves == 1
    assert env.matrix.try_acquire_cell(0, 0)       # vecchia cella liberata
    assert not env.matrix.try_acquire_cell(1, 0)   # nuova cella tenuta


def test_ant_stays_put_when_every_neighbour_is_occupied() -> None:
    env = _env(1, 2, ant_cells=[(0, 0), (1, 0)])
    ant = env.ants[0]
    result = env.step(ant, Action(Manipulation.NOOP, _only_move_index(env, ant)))
    assert ant.position == Position(0, 0)
    assert env.total_moves == 0
    assert result.info["contested"] is True


def test_stay_action_does_not_move() -> None:
    env = _env(3, 3, ant_cells=[(1, 1)])
    ant = env.ants[0]
    env.step(ant, Action(Manipulation.NOOP, STAY))
    assert ant.position == Position(1, 1)
    assert env.total_moves == 0


def test_pick_and_drop_update_counters_and_carry() -> None:
    # seme su (0,0), tipo diverso accanto: pick "manuale" via azione
    env = _env(1, 3, seed_cells=[(0, 0, 0), (1, 0, 1)], ant_cells=[(0, 0)], seed_types=2)
    ant = env.ants[0]
    env.step(ant, Action(Manipulation.PICK, STAY))
    assert ant.carried_seed is not None
    assert env.seeds_collected == 1
    assert not env.matrix.has_seed(0, 0)


def test_entropy_placed_is_defined_while_carrying() -> None:
    env = _env(1, 3, seed_cells=[(0, 0, 0), (1, 0, 1)], ant_cells=[(0, 0)], seed_types=2)
    env.step(env.ants[0], Action(Manipulation.PICK, STAY))
    value = env.matrix.check_entropy_placed(2)
    assert value >= 0.0  # mai -1, anche con un seme in mano


def test_pick_of_misplaced_seed_is_rewarded() -> None:
    # BLUE (0) circondato solo da PURPLE (1): f_own = 0 < pivot -> pick premiato.
    seeds = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1)]
    env = _env(3, 3, seed_cells=seeds, ant_cells=[(1, 1)], seed_types=2)
    result = env.step(env.ants[0], Action(Manipulation.PICK, STAY))
    assert result.info["manip_success"] is True
    assert result.info["f_own"] == 0.0
    assert result.reward > 0.0


def test_pick_of_well_clustered_seed_is_penalised() -> None:
    # BLUE circondato solo da BLUE: f_own = 1 > pivot -> pick penalizzato.
    seeds = [(1, 1, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 2, 0)]
    env = _env(3, 3, seed_cells=seeds, ant_cells=[(1, 1)], seed_types=2)
    result = env.step(env.ants[0], Action(Manipulation.PICK, STAY))
    assert result.info["manip_success"] is True
    assert result.info["f_own"] == 1.0
    assert result.reward < 0.0


def test_drop_next_to_a_foreign_seed_is_penalised() -> None:
    # posare un BLUE con accanto solo un PURPLE: f_own = 0 < pivot -> drop penalizzato.
    env = _env(1, 3, seed_cells=[(0, 0, 1)], ant_cells=[(1, 0)], seed_types=2)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is True
    assert result.info["f_own"] == 0.0
    assert result.reward < 0.0


def test_drop_that_joins_same_type_seeds_is_rewarded() -> None:
    # posare un BLUE nel varco fra due BLUE: f_own = 1 > pivot -> drop premiato.
    env = _env(1, 4, seed_cells=[(0, 0, 1), (1, 0, 0), (3, 0, 0)], ant_cells=[(2, 0)], seed_types=2)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is True
    assert result.info["f_own"] == 1.0
    assert result.reward > 0.0


def test_pivot_defaults_to_chance_level_of_the_scenario() -> None:
    # Senza pivot esplicito il livello di riferimento e' 1/seed_types (il caso).
    for seed_types in (2, 3, 4, 5):
        env = _env(3, 3, ant_cells=[(1, 1)], seed_types=seed_types)
        assert env.pivot == pytest.approx(1.0 / seed_types)


def test_explicit_pivot_overrides_the_chance_level() -> None:
    init = InitialState(rows=3, cols=3, seed_types=5, n_seeds=0, n_ants=1, master_seed=0,
                        seed_cells=(), ant_cells=((1, 1),), ant_dirs=(0,))
    env = Environment.from_initial_state(init, reward=RewardConfig(pivot=0.5))
    assert env.pivot == 0.5


def test_drop_at_chance_level_is_not_punished_with_many_seed_types() -> None:
    """Regressione: con ``pivot`` fisso a 0.5 e 5 tipi, un drop "medio" (f_own = 1/5)
    valeva -0.30 e la politica greedy imparava a non posare mai, lasciando le
    formiche bloccate con il seme in mano. Al livello del caso deve valere 0.
    """
    # cella centrale libera, 5 vicini: 1 dello stesso tipo del seme trasportato -> f_own = 0.2 = 1/5
    seeds = [(1, 0, 0), (0, 1, 1), (2, 1, 2), (0, 0, 3), (2, 0, 4)]
    env = _env(3, 3, seed_cells=seeds, ant_cells=[(1, 1)], seed_types=5)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)  # code 0
    result = env.step(ant, Action(Manipulation.DROP, STAY))
    assert result.info["manip_success"] is True
    assert result.info["f_own"] == pytest.approx(0.2)
    # il contributo della manipolazione e' nullo: resta solo la penalita' di passo
    assert result.reward == pytest.approx(-env.reward_cfg.step_penalty)


def test_free_ant_is_pointed_at_the_most_misplaced_neighbouring_seed() -> None:
    """Regressione: prima ``best_dir`` era calcolato solo mentre si trasportava, quindi
    la formica libera non aveva **nessuna** informazione su dove fossero i semi e il
    suo movimento degenerava in un riflesso fisso su heading/celle bloccate.
    """
    # (1,1) libera. A sinistra un BLUE ben circondato da BLUE, a destra un BLUE
    # isolato fra PURPLE: il bersaglio giusto e' quello di destra.
    seeds = [
        (0, 1, 0), (0, 0, 0), (0, 2, 0),          # BLUE fra BLUE -> a posto
        (3, 1, 0), (3, 0, 1), (3, 2, 1), (4, 1, 1),  # BLUE fra PURPLE -> fuori posto
        (2, 1, 1),  # PURPLE adiacente alla formica, verso destra
    ]
    env = _env(3, 5, seed_cells=seeds, ant_cells=[(1, 1)], seed_types=2)
    obs = env.observe(env.ants[0])
    assert obs.carrying is False
    assert obs.best_dir != STAY, "la formica libera deve avere un bersaglio"
    neigh = _neighbours(env, env.ants[0])
    target = neigh[obs.best_dir]
    assert (target.x, target.y) == (2, 1)  # il PURPLE fuori posto, non il BLUE a posto


def test_carrying_ant_is_pointed_at_an_empty_cell_not_at_an_occupied_one() -> None:
    """Puntare a una cella che *contiene* gia' un seme simile e' inutile: li' non si posa."""
    # formica su (1,1) con un BLUE in mano. (2,1) contiene un BLUE (non si puo' posare),
    # (1,0) e' vuota e circondata da BLUE -> e' li' che va posato.
    seeds = [(2, 1, 0), (0, 0, 0), (2, 0, 0)]
    env = _env(3, 3, seed_cells=seeds, ant_cells=[(1, 1)], seed_types=2)
    ant = env.ants[0]
    ant.carried_seed = Seed(SeedType.BLUE)
    obs = env.observe(ant)
    assert obs.best_dir != STAY
    target = _neighbours(env, ant)[obs.best_dir]
    assert not env.matrix.has_seed(target.x, target.y), "bersaglio occupato: non ci si puo' posare"
    assert (target.x, target.y) == (1, 0)


def test_best_dir_is_stay_when_there_is_no_target() -> None:
    env = _env(3, 3, ant_cells=[(1, 1)], seed_types=2)  # griglia vuota
    assert env.observe(env.ants[0]).best_dir == STAY


def test_block_scan_matches_the_matrix_neighbour_counts(make_config) -> None:
    """``_counts_from_block`` deve coincidere con ``count_types_around`` della matrice."""
    cfg = make_config(cols=9, rows=7, n_ants=2, n_seeds=25, seed_types=4)
    init = build_initial_state(cfg, 4242)
    env = Environment.from_initial_state(init)
    for ant in env.ants:
        cx, cy = ant.position.x, ant.position.y
        block = env._seed_block(cx, cy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < cfg.cols and 0 <= y < cfg.rows:
                    assert env._counts_from_block(block, dx, dy) == \
                        env.matrix.count_types_around(x, y, cfg.seed_types)


def _neighbours(env: Environment, ant: Ant):
    from antelligent.utilities import check_move

    return check_move.check_around(
        ant.position.x, ant.position.y, env.matrix.rows, env.matrix.cols, ant.direction
    )


def test_tick_advances_and_twins_start_identical(make_config) -> None:
    cfg = make_config(n_ants=4, n_seeds=6, cols=6, rows=5, seed_types=3)
    init = build_initial_state(cfg, 999)
    env_a = Environment.from_initial_state(init)
    env_b = Environment.from_initial_state(init)
    assert env_a.initial_entropy == env_b.initial_entropy
    assert {(s.position.x, s.position.y, s.code) for s in env_a.seeds} == {
        (s.position.x, s.position.y, s.code) for s in env_b.seeds
    }
    policy = HeuristicPolicy(env_a.matrix, cfg.seed_types)
    env_a.tick(policy)
    assert env_a.iterations == 1


def _only_move_index(env: Environment, ant: Ant) -> int:
    """Indice dell'unica cella adiacente valida (per griglie strette usate nei test)."""
    from antelligent.utilities import check_move

    neigh = check_move.check_around(
        ant.position.x, ant.position.y, env.matrix.rows, env.matrix.cols, ant.direction
    )
    for i, p in enumerate(neigh):
        if p is not None:
            return i
    return STAY
