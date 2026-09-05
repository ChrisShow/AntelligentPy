"""Fixture condivise dai test."""

from __future__ import annotations

import pytest

from antelligent.config import SimulationConfig

_DEFAULTS = dict(
    cols=6,
    rows=4,
    n_thread=2,
    n_ants=3,
    n_seeds=8,
    seed_types=3,
    matrix_type=1,
    refresh_rate=10,
    stop_criterion=0,
    max_iterations=50,
    entropy_threshold=5.0,
    master_seed=123,
)


@pytest.fixture
def make_config():
    def _make(**overrides) -> SimulationConfig:
        params = dict(_DEFAULTS)
        params.update(overrides)
        return SimulationConfig(**params)

    return _make
