"""Global, baseline-neutral budget accounting with final-stage reserves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Dict


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    total_tokens: int | None
    wall_clock_seconds: float | None
    model_calls: int | None
    tool_calls: int | None
    shell_calls: int | None
    test_runs: int | None
    environment_runs: int | None
    max_nodes: int | None
    max_graph_edits: int | None
    reserve_final_tokens: int = 0
    reserve_validation_tokens: int = 0
    soft_fraction: float = 0.8
    environment_steps: int = 1000000
    interaction_rounds: int = 1000000

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if value is None:
                continue
            if key == "soft_fraction":
                if not 0 < value <= 1:
                    raise ValueError("soft_fraction must be in (0, 1]")
            elif value < 0:
                raise ValueError(f"{key} must be non-negative")
        if self.total_tokens is not None and self.reserve_final_tokens + self.reserve_validation_tokens > self.total_tokens:
            raise ValueError("token reserves exceed total token budget")


class GlobalBudgetLedger:
    """Thread-safe accounting shared by every node, controller, and tool."""

    TOKEN_FIELDS = {
        "input_tokens", "output_tokens", "cached_tokens", "controller_tokens",
        "agent_tokens", "tool_result_tokens", "communication_tokens",
    }

    def __init__(self, limits: BudgetLimits) -> None:
        limits.validate()
        self.limits = limits
        self._lock = RLock()
        self._usage: Dict[str, float] = {
            "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
            "controller_tokens": 0, "agent_tokens": 0, "tool_result_tokens": 0,
            "communication_tokens": 0, "model_calls": 0, "tool_calls": 0,
            "shell_calls": 0, "test_runs": 0, "environment_runs": 0,
            "environment_steps": 0, "interaction_rounds": 0,
            "node_additions": 0, "node_removals": 0, "edge_rewires": 0,
            "wall_clock_seconds": 0.0, "gpu_seconds": 0.0,
            "estimated_cost_usd": 0.0,
        }

    @property
    def total_tokens(self) -> int:
        provider_total = self._usage["input_tokens"] + self._usage["output_tokens"] + self._usage["cached_tokens"]
        attributed_model = self._usage["controller_tokens"] + self._usage["agent_tokens"]
        return int(max(provider_total, attributed_model) + self._usage["tool_result_tokens"])

    def available_tokens(self, *, phase: str = "work") -> int:
        if self.limits.total_tokens is None:
            return 2_147_483_647
        reserve = 0
        if phase == "work":
            reserve = self.limits.reserve_final_tokens + self.limits.reserve_validation_tokens
        elif phase == "final":
            reserve = self.limits.reserve_validation_tokens
        return max(0, self.limits.total_tokens - self.total_tokens - reserve)

    def charge(self, category: str, amount: float = 1, *, phase: str = "work") -> None:
        if category not in self._usage:
            raise KeyError(f"unknown budget category: {category}")
        if amount < 0:
            raise ValueError("budget charges must be non-negative")
        with self._lock:
            token_categories = self.TOKEN_FIELDS - {"communication_tokens"}
            if category in token_categories:
                before = self._usage[category]
                self._usage[category] += amount
                exceeds = self.limits.total_tokens is not None and self.total_tokens > self.limits.total_tokens - (
                    (self.limits.reserve_final_tokens + self.limits.reserve_validation_tokens) if phase == "work"
                    else self.limits.reserve_validation_tokens if phase == "final" else 0
                )
                self._usage[category] = before
                if exceeds:
                    raise BudgetExceeded(f"token budget exhausted during {phase}")
            self._usage[category] += amount
            self._enforce_non_token_limit(category)

    def _enforce_non_token_limit(self, category: str) -> None:
        limit_name = {
            "model_calls": "model_calls", "tool_calls": "tool_calls",
            "shell_calls": "shell_calls", "test_runs": "test_runs",
            "environment_runs": "environment_runs", "wall_clock_seconds": "wall_clock_seconds",
            "environment_steps": "environment_steps", "interaction_rounds": "interaction_rounds",
        }.get(category)
        limit = getattr(self.limits, limit_name) if limit_name else None
        if limit is not None and self._usage[category] > limit:
            raise BudgetExceeded(f"{category} hard limit exceeded")
        edits = self._usage["node_additions"] + self._usage["node_removals"] + self._usage["edge_rewires"]
        if self.limits.max_graph_edits is not None and edits > self.limits.max_graph_edits:
            raise BudgetExceeded("maximum graph edits exceeded")

    def warning_state(self) -> Dict[str, bool]:
        return {
            "tokens": self.limits.total_tokens is not None and self.total_tokens >= self.limits.total_tokens * self.limits.soft_fraction,
            "wall_clock": self.limits.wall_clock_seconds is not None and self._usage["wall_clock_seconds"] >= self.limits.wall_clock_seconds * self.limits.soft_fraction,
        }

    def snapshot(self) -> Dict[str, float | int | Dict[str, bool]]:
        with self._lock:
            return {**self._usage, "total_tokens": self.total_tokens, "warnings": self.warning_state()}
