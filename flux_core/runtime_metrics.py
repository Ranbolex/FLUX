"""Protocol-v2 runtime capacity and collapse trajectory reconstruction."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .collapse import compute_cci_load, compute_cci_path, compute_semci


_DECISION_TARGETS = frozenset({"DecisionNode", "Integrator"})
_SOURCE_INFRASTRUCTURE = frozenset(
    {"TaskArchitect", "DecisionNode", "Integrator", "EvidenceConsolidator"}
)


def _clean_graph(graph: Mapping[str, Any] | None) -> tuple[list[str], list[tuple[str, str]]]:
    graph = graph or {}
    nodes = [str(item) for item in graph.get("nodes", [])]
    known = set(nodes)
    edges = [
        (str(edge[0]), str(edge[1]))
        for edge in graph.get("edges", [])
        if isinstance(edge, (list, tuple))
        and len(edge) == 2
        and str(edge[0]) in known
        and str(edge[1]) in known
    ]
    return nodes, edges


def _indexed_edges(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[tuple[int, int]]:
    index = {node: offset for offset, node in enumerate(nodes)}
    return [
        (index[source], index[target])
        for source, target in edges
        if source in index and target in index and source != target
    ]


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", str(text or "").lower()))


def _mean_pairwise_jaccard(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 0.0
    token_sets = [_token_set(text) for text in texts]
    values: list[float] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            values.append(len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else 0.0


def _normalized_hhi(loads: Sequence[int]) -> float:
    if len(loads) <= 1 or sum(loads) <= 0:
        return 0.0
    total = float(sum(loads))
    hhi = sum((value / total) ** 2 for value in loads)
    baseline = 1.0 / len(loads)
    return max(0.0, min(1.0, (hhi - baseline) / (1.0 - baseline)))


def _normalized_entropy_deficit(loads: Sequence[int]) -> float:
    """Return the normalized entropy deficit for non-negative message loads."""
    positive = [max(0, int(value)) for value in loads]
    total = sum(positive)
    if len(positive) <= 1 or total <= 0:
        return 0.0
    probabilities = [value / total for value in positive if value > 0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(positive))))


def _used_source_metrics(
    *,
    nodes: Sequence[str],
    window_messages: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, int]:
    """Measure concentration and coverage of provenance used by the decision."""
    eligible = [node for node in nodes if node not in _SOURCE_INFRASTRUCTURE]
    if len(eligible) < 2:
        return None, None, 0

    eligible_set = set(eligible)
    loads: dict[str, int] = defaultdict(int)
    terminal_packets = 0
    for message in window_messages:
        if message.get("delivered", True) is False:
            continue
        if str(message.get("target", "")) not in _DECISION_TARGETS:
            continue
        if not bool(message.get("consumed", message.get("used_by_decision", False))):
            continue
        terminal_packets += 1
        provenance = message.get("consolidated_sources")
        if not isinstance(provenance, (list, tuple)) or not provenance:
            provenance = [message.get("source")]
        for source in {str(item) for item in provenance if item is not None}:
            if source in eligible_set:
                loads[source] += 1

    load_vector = [loads.get(node, 0) for node in eligible]
    if sum(load_vector) <= 0:
        return None, 0.0, terminal_packets
    coverage = sum(value > 0 for value in load_vector) / len(load_vector)
    return _normalized_entropy_deficit(load_vector), coverage, terminal_packets


def _delivery_utilization(
    *,
    window_messages: Sequence[Mapping[str, Any]],
    consolidated_sources: Mapping[tuple[int, str], set[str]],
) -> float | None:
    delivered = [
        message
        for message in window_messages
        if message.get("delivered", True) is not False
    ]
    if not delivered:
        return None
    consumed = sum(
        _message_is_consumed(message, consolidated_sources=consolidated_sources)
        for message in delivered
    )
    return consumed / len(delivered)


def _message_is_consumed(
    message: Mapping[str, Any],
    *,
    consolidated_sources: Mapping[tuple[int, str], set[str]],
) -> bool:
    explicit = message.get("consumed")
    if explicit is not None:
        return bool(explicit)
    if bool(message.get("used_by_decision")):
        return True
    round_index = _round_of(message, 0)
    target = str(message.get("target", ""))
    source = str(message.get("source", ""))
    return source in consolidated_sources.get((round_index, target), set())


def _round_of(item: Mapping[str, Any], fallback: int) -> int:
    for key in ("round", "round_index", "generation", "step"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return max(0, fallback)


def build_runtime_trace(
    *,
    initial_nodes: Sequence[str],
    initial_edges: Sequence[tuple[str, str]],
    graph_events: Sequence[Mapping[str, Any]],
    node_records: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    n_max: int,
    round_count: int | None = None,
    n_eff: float | int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Build the per-round record required by the AAAI protocol.

    ``cci_load`` is the paper's primary normalized entropy deficit over active
    graph communication load, where ``F_i = in_degree_i + out_degree_i``.
    ``cci_msg`` is the normalized entropy deficit over delivered/consumed
    message counts in the frozen two-round window and never uses token counts.
    ``cci_used_source`` separately measures the concentration of provenance
    that reaches the terminal decision context, expanding lossless
    ``consolidated_sources`` rather than charging the consolidator as evidence.
    ``cci_load_token_hhi`` is a legacy sensitivity field and is not the primary
    CCI-load. ``semci`` keeps the repository's legacy diversity
    direction; ``semci_similarity`` and ``semci_jaccard_similarity`` expose the
    redundancy direction explicitly.
    """
    events_by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    records_by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    messages_by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for index, event in enumerate(graph_events):
        events_by_round[_round_of(event, index)].append(event)
    for index, record in enumerate(node_records):
        records_by_round[_round_of(record, index)].append(record)
    for index, message in enumerate(messages):
        messages_by_round[_round_of(message, index)].append(message)

    consolidated_sources: dict[tuple[int, str], set[str]] = defaultdict(set)
    for index, message in enumerate(messages):
        sources = message.get("consolidated_sources")
        if not isinstance(sources, (list, tuple)):
            continue
        round_index = _round_of(message, index)
        consolidator = str(message.get("source", ""))
        consolidated_sources[(round_index, consolidator)].update(str(item) for item in sources)

    observed_rounds = [*events_by_round, *records_by_round, *messages_by_round]
    inferred = max(observed_rounds, default=0) + 1
    total_rounds = max(1, inferred, int(round_count or 0))
    nodes = list(dict.fromkeys(str(node) for node in initial_nodes))
    edges = [
        (str(source), str(target))
        for source, target in initial_edges
        if str(source) in nodes and str(target) in nodes
    ]
    if graph_events:
        event_nodes, event_edges = _clean_graph(graph_events[0].get("graph_before"))
        if event_nodes:
            nodes, edges = event_nodes, event_edges
    n_start = len(nodes)
    active_exposure = 0
    trace: list[dict[str, Any]] = []
    latest_constraint: Mapping[str, Any] = {}
    latest_semantic_text: dict[str, str] = {}

    for round_index in range(total_rounds):
        interventions: list[dict[str, Any]] = []
        for event in events_by_round.get(round_index, []):
            after_nodes, after_edges = _clean_graph(event.get("graph_after"))
            if after_nodes:
                nodes, edges = after_nodes, after_edges
            constraint = event.get("constraint_vector_after")
            if isinstance(constraint, Mapping):
                latest_constraint = constraint
            interventions.append({
                "action": str(event.get("action", "")),
                "reason": str(event.get("reason", "")),
                "target_node": event.get("target_node"),
            })

        round_records = records_by_round.get(round_index, [])
        round_messages = messages_by_round.get(round_index, [])
        per_agent_tokens: dict[str, int] = defaultdict(int)
        for message in round_messages:
            source = str(message.get("source", ""))
            if source:
                per_agent_tokens[source] += int(message.get("tokens", 0) or 0)
        if not per_agent_tokens:
            for record in round_records:
                node_id = str(record.get("node_id", ""))
                if node_id:
                    per_agent_tokens[node_id] += int(record.get("tokens", 0) or 0)

        for record in round_records:
            node_id = str(record.get("node_id", ""))
            raw = str(record.get("raw_output") or record.get("output") or "").strip()
            diagnostics = record.get("diagnostics")
            if not raw and isinstance(diagnostics, Mapping):
                raw = str(diagnostics.get("summary") or "").strip()
            if node_id and raw:
                latest_semantic_text[node_id] = raw
        for message in round_messages:
            source = str(message.get("source", ""))
            content = str(message.get("content") or "").strip()
            if source and content and source not in latest_semantic_text:
                latest_semantic_text[source] = content

        semantic_nodes = [
            node
            for node in nodes
            if node not in {"TaskArchitect", "DecisionNode"}
            and node in latest_semantic_text
        ]
        texts = [latest_semantic_text[node] for node in semantic_nodes]

        indexed = _indexed_edges(nodes, edges)
        semci_diversity = float(compute_semci(texts)) if texts else 1.0
        token_loads = [int(per_agent_tokens.get(node, 0)) for node in nodes]
        message_loads: dict[str, int] = defaultdict(int)
        window_start = max(0, round_index - 1)
        window_messages = [
            message
            for window_round in range(window_start, round_index + 1)
            for message in messages_by_round.get(window_round, [])
        ]
        for message in window_messages:
            if message.get("delivered", True) is False:
                continue
            source = str(message.get("source", ""))
            target = str(message.get("target", ""))
            if source in nodes:
                message_loads[source] += 1
            if target in nodes and _message_is_consumed(
                message,
                consolidated_sources=consolidated_sources,
            ):
                message_loads[target] += 1
        message_count_loads = [int(message_loads.get(node, 0)) for node in nodes]
        cci_used_source, used_source_coverage, terminal_packet_count = _used_source_metrics(
            nodes=nodes,
            window_messages=window_messages,
        )
        delivery_utilization = _delivery_utilization(
            window_messages=window_messages,
            consolidated_sources=consolidated_sources,
        )
        reached_nodes = {node for node in nodes if node in {"DecisionNode", "Integrator"}}
        for window_round in range(window_start, round_index + 1):
            for message in messages_by_round.get(window_round, []):
                if message.get("delivered", True) is False or not _message_is_consumed(
                    message,
                    consolidated_sources=consolidated_sources,
                ):
                    continue
                source = str(message.get("source", ""))
                target = str(message.get("target", ""))
                if source in nodes:
                    reached_nodes.add(source)
                if target in nodes:
                    reached_nodes.add(target)
                for consolidated in message.get("consolidated_sources") or []:
                    if str(consolidated) in nodes:
                        reached_nodes.add(str(consolidated))
        participating = sum(value > 0 for value in token_loads)
        active_exposure += len(nodes)
        trace.append({
            "t": round_index,
            "active_nodes": list(nodes),
            "edges": [[source, target] for source, target in edges],
            "n_t": len(nodes),
            "active_exposure": active_exposure,
            "action": interventions[-1]["action"] if interventions else None,
            "action_reason": interventions[-1]["reason"] if interventions else None,
            "interventions": interventions,
            "trigger_signals": {
                "cci_load": float(compute_cci_load(indexed, len(nodes))),
                "cci_msg": _normalized_entropy_deficit(message_count_loads),
                "cci_used_source": cci_used_source,
                "used_source_coverage": used_source_coverage,
                "delivery_utilization": delivery_utilization,
                "decision_terminal_packet_count": terminal_packet_count,
                "cci_load_token_hhi": _normalized_hhi(token_loads),
                "cci_path": float(compute_cci_path(indexed, len(nodes))),
                "semci": semci_diversity,
                "semci_similarity": 1.0 - semci_diversity,
                "semci_jaccard_similarity": _mean_pairwise_jaccard(texts),
                "semci_observed_agents": len(texts),
                "semci_pair_count": len(texts) * max(0, len(texts) - 1) // 2,
                "evidence_gap": float(latest_constraint.get("evidence", latest_constraint.get("verification", 0.0)) or 0.0),
                "verification_gap": float(latest_constraint.get("verification", 0.0) or 0.0),
                "collapse_gap": float(latest_constraint.get("collapse", 0.0) or 0.0),
                "budget_gap": float(latest_constraint.get("budget", 0.0) or 0.0),
                "effective_participation": participating / max(1, len(nodes)),
                "n_reach": len(reached_nodes),
                "spr_reach": len(reached_nodes) / max(1, len(nodes)),
            },
            "per_agent_message_load": dict(sorted(message_loads.items())),
            "per_agent_tokens": dict(sorted(per_agent_tokens.items())),
        })

    n_values = [int(row["n_t"]) for row in trace]
    final_n_eff = float(n_eff if n_eff is not None else n_values[-1])
    n_peak = max(n_values, default=n_start)
    cci_msg_values = [float(row["trigger_signals"]["cci_msg"]) for row in trace]
    cci_used_source_values = [
        float(row["trigger_signals"]["cci_used_source"])
        for row in trace
        if row["trigger_signals"]["cci_used_source"] is not None
    ]
    used_source_coverage_values = [
        float(row["trigger_signals"]["used_source_coverage"])
        for row in trace
        if row["trigger_signals"]["used_source_coverage"] is not None
    ]
    delivery_utilization_values = [
        float(row["trigger_signals"]["delivery_utilization"])
        for row in trace
        if row["trigger_signals"]["delivery_utilization"] is not None
    ]
    final_n_reach = int(trace[-1]["trigger_signals"]["n_reach"]) if trace else 0
    summary: dict[str, float | int | None] = {
        "n_start": n_start,
        "n_peak": n_peak,
        "n_final": n_values[-1] if n_values else n_start,
        "active_exposure": active_exposure,
        "runtime_rounds_recorded": len(trace),
        "spr_peak": min(1.0, final_n_eff / max(1, n_peak)),
        "spr_cap": min(1.0, final_n_eff / max(1, n_max)),
        "cci_msg_mean": sum(cci_msg_values) / len(cci_msg_values) if cci_msg_values else 0.0,
        "cci_msg_max": max(cci_msg_values, default=0.0),
        "cci_used_source_mean": (
            sum(cci_used_source_values) / len(cci_used_source_values)
            if cci_used_source_values else None
        ),
        "cci_used_source_max": max(cci_used_source_values, default=None),
        "cci_used_source_observed_rounds": len(cci_used_source_values),
        "cci_used_source_observed_fraction": len(cci_used_source_values) / max(1, len(trace)),
        "used_source_coverage_mean": (
            sum(used_source_coverage_values) / len(used_source_coverage_values)
            if used_source_coverage_values else None
        ),
        "delivery_utilization_mean": (
            sum(delivery_utilization_values) / len(delivery_utilization_values)
            if delivery_utilization_values else None
        ),
        "delivery_utilization_min": min(delivery_utilization_values, default=None),
        "n_reach": final_n_reach,
        "spr_reach": final_n_reach / max(1, n_values[-1] if n_values else n_start),
    }
    return trace, summary
