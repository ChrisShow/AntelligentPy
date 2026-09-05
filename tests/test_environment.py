"""Ambiente: fisica di occupazione/contesa, contatori, entropia, tick, copie gemelle."""

from __future__ import annotations

from antelligent.actions import STAY, Action, Manipulation
from antelligent.environment import Environment
from antelligent.policies.heuristic import HeuristicPolicy
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
