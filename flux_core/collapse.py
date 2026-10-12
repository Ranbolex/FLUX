"""Structural and semantic collapse metrics.

The older SCI-* helpers are kept for backward compatibility with existing
experiment scripts. The CCI/HER helpers below match the Communication Collapse
paper framing: load concentration, path concentration, semantic redundancy, and
single-hub exposure.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

_encoder_model = None


def _get_sentence_encoder():
    global _encoder_model
    if _encoder_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if os.environ.get("FLUXGEN_ALLOW_CCI_SEM_MODEL_DOWNLOAD", "").lower() not in {"1", "true", "yes"}:
                kwargs["local_files_only"] = True
            _encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **kwargs)
        except Exception:
            _encoder_model = False
    return _encoder_model


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _mean_pairwise_jaccard(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 0.0
    sims: List[float] = []
    toks = [_token_set(t) for t in texts]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            union = toks[i] | toks[j]
            if not union:
                sims.append(0.0)
            else:
                sims.append(len(toks[i] & toks[j]) / len(union))
    return float(sum(sims) / max(1, len(sims)))


def compute_sci_hub(edges: Sequence[Tuple[int, int]], n_nodes: int) -> float:
    """
    Fraction of total in-degree mass captured by the top-3 in-degree nodes.

    Parameters
    ----------
    edges :
        Directed edges (u, v) meaning u -> v (in-degree counted at v).
    n_nodes :
        Number of nodes in the graph.
    """
    if n_nodes <= 0:
        return 0.0
    indeg = [0] * n_nodes
    for u, v in edges:
        if 0 <= v < n_nodes:
            indeg[v] += 1
    total = float(sum(indeg))
    if total <= 0:
        return 0.0
    top3 = sum(sorted(indeg, reverse=True)[:3])
    return float(top3 / total)


def _build_digraph(edges: Sequence[Tuple[int, int]], n_nodes: int) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(max(0, int(n_nodes))))
    for u, v in edges:
        if 0 <= int(u) < n_nodes and 0 <= int(v) < n_nodes and int(u) != int(v):
            g.add_edge(int(u), int(v))
    return g


def _entropy_deficit(values: Sequence[float]) -> float:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    total = float(vals.sum())
    if total <= 1e-15:
        return 0.0
    p = vals / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return float(max(0.0, min(1.0, 1.0 - entropy / math.log(vals.size))))


def compute_cci_load(
    edges: Sequence[Tuple[int, int]],
    n_nodes: int,
    *,
    mode: str = "total",
) -> float:
    """
    CCI-load: normalized entropy deficit of communication flow over nodes.

    The paper operationalizes each node's communication load as
    ``F_i = in_degree_i + out_degree_i`` on the active directed communication
    graph. With ``p_i = F_i / sum_j F_j``, this function returns
    ``1 - H(p) / log(N)``. Token-weighted HHI is a separate optional sensitivity
    diagnostic and must not be called the primary CCI-load.

    Parameters
    ----------
    edges:
        Directed edges ``u -> v``.
    n_nodes:
        Number of nodes.
    mode:
        ``in`` uses in-degree, ``out`` uses out-degree, and ``total`` uses their
        sum. ``total`` is the default because both hub consumers and hub
        broadcasters can induce collapse.
    """
    if n_nodes <= 1:
        return 0.0
    indeg = [0.0] * n_nodes
    outdeg = [0.0] * n_nodes
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            outdeg[u] += 1.0
            indeg[v] += 1.0
    if mode == "in":
        return _entropy_deficit(indeg)
    if mode == "out":
        return _entropy_deficit(outdeg)
    return _entropy_deficit([i + o for i, o in zip(indeg, outdeg)])


def compute_hub_exposure_ratio(
    edges: Sequence[Tuple[int, int]],
    n_nodes: int,
    *,
    mode: str = "total",
) -> float:
    """HER: fraction of communication load carried by the single busiest node."""
    if n_nodes <= 0:
        return 0.0
    indeg = [0.0] * n_nodes
    outdeg = [0.0] * n_nodes
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            outdeg[u] += 1.0
            indeg[v] += 1.0
    if mode == "in":
        vals = indeg
    elif mode == "out":
        vals = outdeg
    else:
        vals = [i + o for i, o in zip(indeg, outdeg)]
    total = float(sum(vals))
    if total <= 1e-15:
        return 0.0
    return float(max(vals) / total)


def compute_cci_path(edges: Sequence[Tuple[int, int]], n_nodes: int) -> float:
    """
    CCI-path: concentration of dependency paths on the busiest node.

    For DAGs up to moderate scale, this computes the fraction of all
    source-to-sink paths that include each intermediate node. If the graph is
    cyclic, or path counts explode, it falls back to betweenness concentration.
    """
    if n_nodes <= 2:
        return 0.0
    g = _build_digraph(edges, n_nodes)
    if g.number_of_edges() == 0:
        return 0.0

    if not nx.is_directed_acyclic_graph(g):
        bc = nx.betweenness_centrality(g)
        total = float(sum(bc.values()))
        return 0.0 if total <= 1e-15 else float(max(bc.values()) / total)

    sources = [v for v in g.nodes if g.in_degree(v) == 0]
    sinks = [v for v in g.nodes if g.out_degree(v) == 0]
    if not sources or not sinks:
        return 0.0

    topo = list(nx.topological_sort(g))
    source_paths = {v: 0.0 for v in g.nodes}
    sink_paths = {v: 0.0 for v in g.nodes}
    for s in sources:
        source_paths[s] = 1.0
    for v in topo:
        for _, child in g.out_edges(v):
            source_paths[child] += source_paths[v]
    for t in sinks:
        sink_paths[t] = 1.0
    for v in reversed(topo):
        for parent, _ in g.in_edges(v):
            sink_paths[parent] += sink_paths[v]
    total_paths = sum(source_paths[t] for t in sinks)
    if total_paths <= 1e-15:
        return 0.0
    exposures = []
    for v in g.nodes:
        if v in sources or v in sinks:
            continue
        exposures.append((source_paths[v] * sink_paths[v]) / total_paths)
    if not exposures:
        return 0.0
    return float(max(exposures))


def compute_sci_path(edges: Sequence[Tuple[int, int]], n_nodes: int) -> float:
    """
    Path / routing collapse: fraction of **betweenness centrality** mass on the
    top-3 nodes (parallel to SCI-hub on in-degree — «高速公路化» of shortest-path traffic).
    """
    if n_nodes <= 0:
        return 0.0
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            g.add_edge(u, v)
    if g.number_of_edges() == 0:
        return 0.0
    bc = nx.betweenness_centrality(g)
    vals = sorted(bc.values(), reverse=True)
    total = float(sum(bc.values()))
    if total <= 1e-15:
        return 0.0
    top3 = float(sum(vals[:3]))
    return top3 / total


def compute_cci_sem_from_embeddings(
    embeddings: Sequence[Sequence[float]],
    *,
    margin: float = 0.35,
) -> float:
    """
    CCI-sem: mean above-margin pairwise cosine similarity.

    Higher means stronger semantic redundancy. This is the paper-side
    complement of SemCI, whose higher value means diversity.
    """
    if len(embeddings) < 2:
        return 0.0
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.maximum(norm, 1e-12)
    sim = x @ x.T
    n = int(sim.shape[0])
    mask = ~np.eye(n, dtype=bool)
    pairs = sim[mask]
    red = np.maximum(pairs - float(margin), 0.0)
    return float(red.mean())


def compute_cci_sem(
    agent_outputs: List[str],
    *,
    margin: float = 0.35,
) -> float:
    """CCI-sem from text outputs via the sentence-transformer encoder."""
    if len(agent_outputs) < 2:
        return 0.0
    enc = _get_sentence_encoder()
    if enc is False:
        return float(max(0.0, _mean_pairwise_jaccard(agent_outputs) - margin))
    emb = enc.encode(agent_outputs, convert_to_numpy=True, show_progress_bar=False)
    return compute_cci_sem_from_embeddings(emb, margin=margin)


def compute_collapse_metrics(
    edges: Sequence[Tuple[int, int]],
    n_nodes: int,
    *,
    embeddings: Optional[Sequence[Sequence[float]]] = None,
    outputs: Optional[List[str]] = None,
    sem_margin: float = 0.35,
) -> Dict[str, float]:
    """Return the paper-facing collapse metric family in one dict."""
    cci_sem = 0.0
    if embeddings is not None:
        cci_sem = compute_cci_sem_from_embeddings(embeddings, margin=sem_margin)
    elif outputs is not None:
        cci_sem = compute_cci_sem(outputs, margin=sem_margin)
    return {
        "cci_load": compute_cci_load(edges, n_nodes),
        "cci_path": compute_cci_path(edges, n_nodes),
        "cci_sem": cci_sem,
        "her": compute_hub_exposure_ratio(edges, n_nodes),
        "sci_hub": compute_sci_hub(edges, n_nodes),
        "sci_path": compute_sci_path(edges, n_nodes),
    }


def _base_role_name(role: str) -> str:
    text = str(role)
    if "_" in text:
        head, tail = text.rsplit("_", 1)
        if tail.isdigit():
            return head
    return text


def compute_scale_preservation_metrics(
    edges: Sequence[Tuple[int, int]],
    n_nodes: int,
    *,
    requested_n: Optional[int] = None,
    roles: Optional[Sequence[str]] = None,
    skipped_nodes: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Scale-preservation diagnostics for reduction-based versus scale-preserving
    collaboration.

    ``N_eff`` is the number of non-skipped nodes that can structurally influence
    the final decision node under the current MacNet execution convention. For a
    DAG, the decision node is the last node in topological order; effective nodes
    are its ancestors plus itself. If the graph is cyclic or disconnected in a
    way that prevents a topological order, the fallback uses all non-skipped
    non-isolated nodes.

    ``SPR`` measures how much of the requested scale remains effective.
    ``RCR`` measures base-role coverage among effective nodes. ``FCE`` is the
    entropy of effective communication load, and ``CRC`` flags collapse relief
    achieved by compressing scale or role coverage.
    """
    n = max(0, int(n_nodes))
    req = max(1, int(requested_n if requested_n is not None else n))
    skipped = {int(x) for x in (skipped_nodes or []) if 0 <= int(x) < n}
    active = set(range(n)) - skipped

    if n == 0 or not active:
        return {
            "n_eff": 0.0,
            "spr": 0.0,
            "rcr": 0.0,
            "fce": 0.0,
            "crc": 1.0,
        }

    g = _build_digraph(edges, n)
    try:
        topo = list(nx.topological_sort(g))
        if topo:
            decision = topo[-1]
            effective = set(nx.ancestors(g, decision)) | {decision}
        else:
            effective = set(active)
    except Exception:
        effective = {v for v in active if g.degree(v) > 0}
        if not effective:
            effective = set(active)

    effective &= active
    n_eff = float(len(effective))
    spr = float(min(1.0, n_eff / req))

    role_list = [str(r) for r in (roles or [])]
    if role_list and len(role_list) >= n:
        required_roles = {_base_role_name(role_list[i]) for i in range(min(req, len(role_list)))}
        effective_roles = {_base_role_name(role_list[i]) for i in effective if i < len(role_list)}
        rcr = float(len(effective_roles) / max(1, len(required_roles)))
    else:
        rcr = spr
    rcr = float(max(0.0, min(1.0, rcr)))

    indeg = [0.0] * n
    outdeg = [0.0] * n
    for u, v in edges:
        if 0 <= int(u) < n and 0 <= int(v) < n:
            outdeg[int(u)] += 1.0
            indeg[int(v)] += 1.0
    loads = [indeg[i] + outdeg[i] for i in sorted(effective)]
    if not loads or sum(loads) <= 1e-15 or len(loads) <= 1:
        fce = 0.0 if n_eff <= 1 else 1.0
    else:
        fce = float(1.0 - _entropy_deficit(loads))

    crc = float(max(0.0, min(1.0, 1.0 - (spr * rcr))))
    return {
        "n_eff": n_eff,
        "spr": spr,
        "rcr": rcr,
        "fce": fce,
        "crc": crc,
    }


def compute_semci(agent_outputs: List[str]) -> float:
    """
    Semantic diversity: 1 - mean pairwise cosine similarity of utterances.

    Higher is more diverse (better).
    """
    if len(agent_outputs) < 2:
        return 1.0
    enc = _get_sentence_encoder()
    if enc is False:
        return float(1.0 - _mean_pairwise_jaccard(agent_outputs))
    emb = enc.encode(agent_outputs, convert_to_numpy=True, show_progress_bar=False)
    x = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.maximum(norm, 1e-12)
    sim = x @ x.T
    n = int(sim.shape[0])
    mask = ~np.eye(n, dtype=bool)
    pairs = sim[mask]
    mean_sim = float(pairs.mean())
    return float(1.0 - mean_sim)
