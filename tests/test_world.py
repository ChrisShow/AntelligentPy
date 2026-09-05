"""Stato iniziale condiviso (``world.build_initial_state``)."""

from __future__ import annotations

from antelligent.world import build_initial_state, resolve_master_seed


def test_same_master_seed_produces_identical_state(make_config) -> None:
    cfg = make_config()
    assert build_initial_state(cfg, 42) == build_initial_state(cfg, 42)


def test_different_master_seed_produces_different_state(make_config) -> None:
    cfg = make_config()
    a = build_initial_state(cfg, 1)
    b = build_initial_state(cfg, 2)
    assert a.seed_cells != b.seed_cells or a.ant_cells != b.ant_cells


def test_counts_and_distinct_cells(make_config) -> None:
    cfg = make_config(n_seeds=10, n_ants=5, cols=8, rows=5)
    state = build_initial_state(cfg, 7)
    assert len(state.seed_cells) == 10
    assert len(state.ant_cells) == 5
    assert len(state.ant_dirs) == 5
    assert len({(x, y) for x, y, _ in state.seed_cells}) == 10  # semi su celle distinte
    assert len(set(state.ant_cells)) == 5                        # formiche su celle distinte
    for x, y in state.ant_cells:
        assert 0 <= x < cfg.cols and 0 <= y < cfg.rows


def test_resolve_master_seed() -> None:
    assert resolve_master_seed(99) == 99
    assert resolve_master_seed(0) > 0
