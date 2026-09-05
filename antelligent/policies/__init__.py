"""Politiche delle formiche: euristica pre-impostata vs reinforcement learning.

Vedi ``docs/rl-design.md`` §4.1 (il "seam" ``AntPolicy``) e §5 (algoritmi).
"""

from .base import AntPolicy, AntPolicyBase
from .heuristic import HeuristicPolicy
from .tabular import QConfig, TabularQPolicy

__all__ = [
    "AntPolicy",
    "AntPolicyBase",
    "HeuristicPolicy",
    "QConfig",
    "TabularQPolicy",
]
