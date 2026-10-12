"""Capability-oriented role selection for complex-task graph expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RoleSpec:
    name: str
    capabilities: frozenset[str]
    instruction: str


def select_role(pool: Sequence[RoleSpec], gaps: Mapping[str, float], used: Sequence[str]) -> RoleSpec | None:
    used_names = set(used)
    ranked = sorted(
        (role for role in pool if role.name not in used_names),
        key=lambda role: (
            -max((float(gaps.get(capability, 0.0)) for capability in role.capabilities), default=0.0),
            -sum(float(gaps.get(capability, 0.0)) for capability in role.capabilities),
            role.name,
        ),
    )
    if not ranked:
        return None
    best = ranked[0]
    return best if any(float(gaps.get(capability, 0.0)) > 0 for capability in best.capabilities) else None
