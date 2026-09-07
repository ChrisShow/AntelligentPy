"""Interfaccia comune delle politiche."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..actions import Action, Observation, Transition

if TYPE_CHECKING:
    from ..simulation.ant import Ant


@runtime_checkable
class AntPolicy(Protocol):
    def select_action(self, obs: Observation, ant: "Ant") -> Action: ...

    def record(self, transition: Transition) -> None: ...

    def end_episode(self) -> None: ...


class AntPolicyBase:
    """Implementazione di default: ``record`` / ``end_episode`` sono no-op."""

    def select_action(self, obs: Observation, ant: "Ant") -> Action:  # pragma: no cover - astratto
        raise NotImplementedError

    def record(self, transition: Transition) -> None:
        return None

    def end_episode(self) -> None:
        return None
