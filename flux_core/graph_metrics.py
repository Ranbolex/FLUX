"""Defined communication-collapse and graph-participation metrics."""

from __future__ import annotations

from collections import Counter, deque
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .collapse import compute_collapse_metrics, compute_scale_preservation_metrics


def _concentration(values: Sequence[int]) -> float:
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    shares = [value / total for value in values]
    return sum(share * share for share in shares)


def _path_lengths(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[int]:
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
    lengths: list[int] = []
    for start in nodes:
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            current, distance = queue.popleft()
            for target in adjacency.get(current, []):
                if target in seen:
                    continue
                seen.add(target)
                queue.append((target, distance + 1))
                lengths.append(distance + 1)
    return lengths


def communication_metrics(
    *, nodes: Sequence[str], edges: Sequence[tuple[str, str]], decision_node: str | None,
    messages: Iterable[Mapping[str, Any]] = (), evidence: Iterable[Mapping[str, Any]] = (),
    total_tokens: int = 0,
) -> dict[str, float | int]:
    active = list(dict.fromkeys(nodes))
    clean_edges = [(source, target) for source, target in edges if source in active and target in active]
    indegree = Counter(target for _, target in clean_edges)
    outdegree = Counter(source for source, _ in clean_edges)
    degree = Counter(indegree)
    degree.update(outdegree)
    paths = _path_lengths(active, clean_edges)
    message_list = list(messages)
    evidence_list = list(evidence)
    communication_tokens = sum(int(item.get("tokens", 0) or 0) for item in message_list)
    incident = sum(
        item.get("source") == decision_node or item.get("target") == decision_node
        for item in message_list
    )
    generated = [item for item in evidence_list if item.get("valid", True)]
    absorbed = [item for item in generated if item.get("absorbed_by_final")]
    contributing_nodes = {str(item.get("node_id")) for item in absorbed if item.get("node_id") is not None}
    duplicate_messages = sum(bool(item.get("duplicate")) for item in message_list)
    possible_edges = len(active) * max(0, len(active) - 1)
    return {
        "active_node_count": len(active),
        "active_edge_count": len(clean_edges),
        "graph_density": len(clean_edges) / possible_edges if possible_edges else 0.0,
        "average_path_length": mean(paths) if paths else 0.0,
        "max_path_length": max(paths, default=0),
        "in_degree_concentration": _concentration([indegree[node] for node in active]),
        "out_degree_concentration": _concentration([outdegree[node] for node in active]),
        "hub_share": max(degree.values(), default=0) / max(1, 2 * len(clean_edges)),
        "integrator_message_share": incident / len(message_list) if message_list else 0.0,
        "effective_participation_rate": len(contributing_nodes) / len(active) if active else 0.0,
        "evidence_absorption_rate": len(absorbed) / len(generated) if generated else 0.0,
        "duplicate_message_ratio": duplicate_messages / len(message_list) if message_list else 0.0,
        "unused_node_ratio": 1.0 - (len(contributing_nodes) / len(active)) if active else 0.0,
        "communication_tokens": communication_tokens,
        "communication_token_ratio": communication_tokens / total_tokens if total_tokens else 0.0,
    }


def paper_collapse_metrics(
    *,
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    node_records: Iterable[Mapping[str, Any]] = (),
    requested_n: int | None = None,
) -> dict[str, float]:
    """Paper-facing CCI/HER/N_eff/SPR metrics for string-id FLUX graphs."""
    active = list(dict.fromkeys(nodes))
    index = {node: i for i, node in enumerate(active)}
    indexed_edges = [
        (index[source], index[target])
        for source, target in edges
        if source in index and target in index and source != target
    ]
    latest: dict[str, Mapping[str, Any]] = {}
    for record in node_records:
        node_id = str(record.get("node_id", ""))
        if node_id in index:
            latest[node_id] = record

    roles: list[str] = []
    outputs: list[str] = []
    for node in active:
        record = latest.get(node, {})
        roles.append(str(record.get("role") or node))
        diagnostics = record.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        checks = diagnostics.get("checks", [])
        if isinstance(checks, str):
            checks = [checks]
        summary = str(diagnostics.get("summary") or "")
        signature = str(record.get("signature") or "")
        text = " ".join([summary, *[str(item) for item in checks], signature]).strip()
        outputs.append(text or node)

    n_nodes = len(active)
    req = requested_n if requested_n is not None else n_nodes
    collapse = compute_collapse_metrics(indexed_edges, n_nodes, outputs=outputs)
    scale = compute_scale_preservation_metrics(
        indexed_edges,
        n_nodes,
        requested_n=req,
        roles=roles,
    )
    return {**collapse, **scale}
