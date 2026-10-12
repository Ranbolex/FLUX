"""Budget-, difficulty-, and collapse-aware structural control for FLUX.

The controller is training-free and consumes only visible task features,
current-episode diagnostics, graph state, and budget accounting. Dataset
profiles share one decision rule but use different cost horizons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Sequence


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean_pairwise_token_similarity(texts: Sequence[str]) -> float:
    def normalize_token(token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    token_sets = [
        {
            normalize_token(token)
            for token in re.findall(r"[a-z0-9_]+", str(text or "").lower())
        }
        for text in texts
        if str(text or "").strip()
    ]
    if len(token_sets) < 2:
        return 0.0
    similarities: list[float] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            similarities.append(
                len(token_sets[left] & token_sets[right]) / len(union)
                if union
                else 0.0
            )
    return sum(similarities) / len(similarities) if similarities else 0.0


@dataclass(frozen=True)
class HybridControlProfile:
    name: str
    benchmark: str
    difficulty_weight: float
    unresolved_weight: float
    budget_weight: float
    candidate_scale_weight: float
    node_scale_weight: float
    edit_scale_weight: float
    collapse_weight: float
    candidate_threshold: float
    node_threshold: float
    candidate_scale_reference: float
    node_scale_reference: float
    edit_scale_reference: float
    pressure_exponent: float
    collapse_repair_threshold: float
    repair_credit: float
    budget_horizon_rounds: int
    minimum_candidates: int = 1
    minimum_nodes: int = 3


PROFILES: dict[str, HybridControlProfile] = {
    "webshop-hybrid-budget-collapse-v1": HybridControlProfile(
        name="webshop-hybrid-budget-collapse-v1",
        benchmark="webshop",
        difficulty_weight=0.52,
        unresolved_weight=0.48,
        budget_weight=0.50,
        candidate_scale_weight=0.34,
        node_scale_weight=0.30,
        edit_scale_weight=0.18,
        collapse_weight=0.32,
        candidate_threshold=0.18,
        node_threshold=0.18,
        candidate_scale_reference=5.0,
        node_scale_reference=7.0,
        edit_scale_reference=8.0,
        pressure_exponent=2.0,
        collapse_repair_threshold=0.45,
        repair_credit=0.55,
        budget_horizon_rounds=8,
    ),
    "mmlu-pro-hybrid-budget-collapse-v1": HybridControlProfile(
        name="mmlu-pro-hybrid-budget-collapse-v1",
        benchmark="mmlu-pro",
        difficulty_weight=0.60,
        unresolved_weight=0.40,
        budget_weight=0.38,
        candidate_scale_weight=0.28,
        node_scale_weight=0.25,
        edit_scale_weight=0.14,
        collapse_weight=0.30,
        candidate_threshold=0.12,
        node_threshold=0.14,
        candidate_scale_reference=3.0,
        node_scale_reference=6.0,
        edit_scale_reference=6.0,
        pressure_exponent=2.0,
        collapse_repair_threshold=0.45,
        repair_credit=0.50,
        budget_horizon_rounds=1,
    ),
    "webshop-hybrid-budget-collapse-v2": HybridControlProfile(
        name="webshop-hybrid-budget-collapse-v2",
        benchmark="webshop",
        difficulty_weight=0.52,
        unresolved_weight=0.48,
        budget_weight=0.50,
        candidate_scale_weight=0.34,
        node_scale_weight=0.30,
        edit_scale_weight=0.18,
        collapse_weight=0.32,
        candidate_threshold=0.22,
        node_threshold=0.18,
        candidate_scale_reference=5.0,
        node_scale_reference=7.0,
        edit_scale_reference=8.0,
        pressure_exponent=2.0,
        collapse_repair_threshold=0.45,
        repair_credit=0.55,
        budget_horizon_rounds=8,
    ),
    "mmlu-pro-hybrid-budget-collapse-v2": HybridControlProfile(
        name="mmlu-pro-hybrid-budget-collapse-v2",
        benchmark="mmlu-pro",
        difficulty_weight=0.60,
        unresolved_weight=0.40,
        budget_weight=0.38,
        candidate_scale_weight=0.28,
        node_scale_weight=0.25,
        edit_scale_weight=0.14,
        collapse_weight=0.30,
        candidate_threshold=0.12,
        node_threshold=0.14,
        candidate_scale_reference=3.0,
        node_scale_reference=6.0,
        edit_scale_reference=6.0,
        pressure_exponent=2.0,
        collapse_repair_threshold=0.45,
        repair_credit=0.50,
        budget_horizon_rounds=1,
    ),
}


def get_hybrid_control_profile(name: str, benchmark: str) -> HybridControlProfile:
    try:
        profile = PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown adaptive_control_profile: {name}") from exc
    if profile.benchmark != benchmark:
        raise ValueError(
            f"adaptive_control_profile {name!r} is for {profile.benchmark}, not {benchmark}"
        )
    return profile


@dataclass(frozen=True)
class HybridControlSignals:
    difficulty: float
    unresolved: float
    candidate_count: int
    active_nodes: int
    graph_edits: int
    max_candidates: int
    max_nodes: int
    max_graph_edits: int
    tokens_used: int
    token_limit: int | None
    calls_used: int
    call_limit: int | None
    reserve_tokens: int
    reserve_calls: int
    projected_incremental_tokens: int
    projected_incremental_calls: int
    structural_collapse: float
    semantic_collapse: float
    budget_feasible: bool = True


@dataclass(frozen=True)
class HybridControlDecision:
    eligible: bool
    action: str
    reason: str
    net_value: float
    benefit: float
    penalty: float
    budget_pressure: float
    candidate_scale_pressure: float
    node_scale_pressure: float
    edit_scale_pressure: float
    collapse_pressure: float
    structural_collapse_pressure: float
    semantic_collapse_pressure: float
    threshold: float
    budget_feasible: bool
    capacity_feasible: bool
    recommends_repair: bool
    semantic_repair: bool
    profile: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _convex_pressure(proposed: int, minimum: int, reference: float, exponent: float) -> float:
    distance = max(0.0, float(proposed - minimum))
    return (distance / max(1e-9, float(reference))) ** float(exponent)


def _projected_budget_pressure(signals: HybridControlSignals) -> float:
    token_pressure = 0.0
    if signals.token_limit is not None and signals.token_limit > 0:
        projected = signals.tokens_used + signals.reserve_tokens + signals.projected_incremental_tokens
        token_pressure = projected / signals.token_limit
    call_pressure = 0.0
    if signals.call_limit is not None and signals.call_limit > 0:
        projected = signals.calls_used + signals.reserve_calls + signals.projected_incremental_calls
        call_pressure = projected / signals.call_limit
    return max(0.0, token_pressure, call_pressure)


def _decision(
    *,
    profile: HybridControlProfile,
    signals: HybridControlSignals,
    action: str,
    gap_strength: float,
    repair: bool,
) -> HybridControlDecision:
    is_candidate_action = action == "add_candidate"
    is_node_action = action in {"add_node", "add_repair_node"}
    is_edit_only_action = action == "rewire"
    is_removal_action = action == "remove_node"
    proposed_candidates = signals.candidate_count + int(action == "add_candidate")
    proposed_nodes = (
        max(profile.minimum_nodes, signals.active_nodes - 1)
        if is_removal_action
        else signals.active_nodes + int(is_candidate_action or is_node_action)
    )
    proposed_edits = signals.graph_edits + int(is_candidate_action or is_node_action or is_edit_only_action)
    candidate_scale = _convex_pressure(
        proposed_candidates,
        profile.minimum_candidates,
        profile.candidate_scale_reference,
        profile.pressure_exponent,
    )
    node_scale = _convex_pressure(
        proposed_nodes,
        profile.minimum_nodes,
        profile.node_scale_reference,
        profile.pressure_exponent,
    )
    edit_scale = _convex_pressure(
        proposed_edits,
        0,
        profile.edit_scale_reference,
        profile.pressure_exponent,
    )
    budget_pressure = _projected_budget_pressure(signals)
    structural_pressure = _clamp01(signals.structural_collapse)
    semantic_pressure = _clamp01(signals.semantic_collapse)
    collapse_pressure = max(structural_pressure, semantic_pressure)
    semantic_repair = (
        action == "add_candidate"
        and semantic_pressure >= profile.collapse_repair_threshold
    )
    difficulty = _clamp01(signals.difficulty)
    unresolved = max(_clamp01(signals.unresolved), _clamp01(gap_strength))
    benefit = profile.difficulty_weight * difficulty + profile.unresolved_weight * unresolved
    if repair:
        benefit += profile.repair_credit * collapse_pressure
    elif semantic_repair:
        benefit += profile.repair_credit * semantic_pressure
    scale_weight = (
        profile.candidate_scale_weight
        if is_candidate_action
        else profile.node_scale_weight
        if is_node_action
        else 0.0
    )
    scale_pressure = candidate_scale if is_candidate_action else node_scale if is_node_action else 0.0
    budget_penalty = (
        profile.budget_weight
        * budget_pressure ** profile.pressure_exponent
        * (1.0 + scale_pressure)
    )
    collapse_penalty_pressure = (
        0.0
        if repair
        else structural_pressure
        if semantic_repair
        else collapse_pressure
    )
    collapse_penalty = (
        profile.collapse_weight
        * collapse_penalty_pressure
        * (1.0 + 0.5 * scale_pressure)
    )
    penalty = (
        budget_penalty
        + scale_weight * scale_pressure
        + profile.edit_scale_weight * edit_scale
        + collapse_penalty
    )
    net_value = benefit - penalty
    threshold = profile.candidate_threshold if is_candidate_action else profile.node_threshold
    capacity_feasible = (
        proposed_candidates <= signals.max_candidates
        and proposed_nodes <= signals.max_nodes
        and proposed_edits <= signals.max_graph_edits
    )
    recommends_repair = (
        not repair
        and structural_pressure >= profile.collapse_repair_threshold
        and signals.active_nodes > profile.minimum_nodes
    )
    eligible = (
        signals.budget_feasible
        and capacity_feasible
        and not recommends_repair
        and net_value >= threshold
    )
    reason = (
        "repair_before_expand"
        if recommends_repair
        else "blocked_by_budget"
        if not signals.budget_feasible
        else "blocked_by_capacity"
        if not capacity_feasible
        else "semantic_repair_positive_net_value"
        if eligible and semantic_repair
        else "positive_net_value" if eligible
        else "below_net_value_threshold"
    )
    return HybridControlDecision(
        eligible=eligible,
        action=action,
        reason=reason,
        net_value=net_value,
        benefit=benefit,
        penalty=penalty,
        budget_pressure=budget_pressure,
        candidate_scale_pressure=candidate_scale,
        node_scale_pressure=node_scale,
        edit_scale_pressure=edit_scale,
        collapse_pressure=collapse_pressure,
        structural_collapse_pressure=structural_pressure,
        semantic_collapse_pressure=semantic_pressure,
        threshold=threshold,
        budget_feasible=signals.budget_feasible,
        capacity_feasible=capacity_feasible,
        recommends_repair=recommends_repair,
        semantic_repair=semantic_repair,
        profile=profile.name,
    )


def decide_candidate_widening(
    profile: HybridControlProfile,
    signals: HybridControlSignals,
) -> HybridControlDecision:
    return _decision(
        profile=profile,
        signals=signals,
        action="add_candidate",
        gap_strength=0.0,
        repair=False,
    )


def decide_graph_expansion(
    profile: HybridControlProfile,
    signals: HybridControlSignals,
    *,
    gap_strength: float,
    repair: bool = False,
) -> HybridControlDecision:
    return _decision(
        profile=profile,
        signals=signals,
        action="add_repair_node" if repair else "add_node",
        gap_strength=gap_strength,
        repair=repair,
    )


def decide_graph_rewire(
    profile: HybridControlProfile,
    signals: HybridControlSignals,
    *,
    gap_strength: float,
) -> HybridControlDecision:
    """Gate an edit-only rewire under the same budget and collapse accounting.

    Rewiring changes graph structure without increasing K or active node count;
    it still consumes one graph edit and the projected budget for the
    re-executed consolidator/control path.
    """
    return _decision(
        profile=profile,
        signals=signals,
        action="rewire",
        gap_strength=gap_strength,
        repair=True,
    )


def decide_graph_removal(
    profile: HybridControlProfile,
    signals: HybridControlSignals,
    *,
    gap_strength: float,
) -> HybridControlDecision:
    """Gate a redundant-node removal while preserving the minimum team elsewhere."""
    return _decision(
        profile=profile,
        signals=signals,
        action="remove_node",
        gap_strength=gap_strength,
        repair=True,
    )


def preserves_minimum_complete_team(
    *,
    nodes: Iterable[str],
    edges: Iterable[Sequence[str]],
    candidate_nodes: Iterable[str],
    remove_node: str,
    architect_node: str = "TaskArchitect",
    decision_node: str = "DecisionNode",
) -> bool:
    remaining = {str(node) for node in nodes if str(node) != remove_node}
    if architect_node not in remaining or decision_node not in remaining or len(remaining) < 3:
        return False
    candidates = {str(node) for node in candidate_nodes if str(node) in remaining}
    if not candidates:
        return False
    graph: dict[str, set[str]] = {node: set() for node in remaining}
    for edge in edges:
        if len(edge) != 2:
            continue
        source, target = str(edge[0]), str(edge[1])
        if source in remaining and target in remaining:
            graph.setdefault(source, set()).add(target)

    def reachable(source: str, target: str) -> bool:
        frontier = [source]
        seen = {source}
        while frontier:
            current = frontier.pop()
            if current == target:
                return True
            for child in graph.get(current, set()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return False

    return any(
        reachable(architect_node, candidate) and reachable(candidate, decision_node)
        for candidate in candidates
    )
