"""HRB metrics (paper Section 5.2) and paired bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hindsighttag.config import HindsightTagConfig
from hindsighttag.core import HindsightTagPlugin


@dataclass
class MetricResults:
    rescue_recall: float
    false_rescue_rate: float
    retention_lift: float
    coallocation_recall_at_k: float
    n_rescued_precursor: int
    n_false_rescue: int
    n_retention_a: int
    n_retention_c: int
    n_coalloc: int
    details: Dict[str, Any]


def _is_retained(host, memory_id: str, query: str, top_k: int = 5) -> Tuple[bool, float]:
    item = host.get_memory(memory_id)
    if item is None or getattr(item, "tombstoned", False):
        return False, 0.0
    retention = host.retention_at(memory_id) if hasattr(host, "retention_at") else item.retention
    retrievable = host.is_retrievable(memory_id, query, top_k=top_k)
    return retrievable, retention


def compute_hrb_metrics(
    host,
    manifest: List[Dict[str, Any]],
    config: HindsightTagConfig,
    plugin: Optional[HindsightTagPlugin] = None,
    top_k: int = 5,
) -> MetricResults:
    """Compute the four Section 5.2 metrics from the planted-scenario manifest."""
    rescued_hits = rescued_total = 0
    false_hits = false_total = 0
    retain_a = retain_a_total = 0
    retain_c = retain_c_total = 0
    coalloc_hits = coalloc_total = 0
    per_scenario: List[Dict[str, Any]] = []

    for s in manifest:
        cat, query = s["category"], s["query"]
        precursor_id, trigger_id = s.get("precursor_id"), s.get("trigger_id")

        if cat == "rescued_precursor" and precursor_id:
            rescued_total += 1
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            rescued_hits += int(retained)
            retain_a_total += 1
            retain_a += int(retained)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat, "retained": retained, "retention": ret})

        elif cat == "heterosynaptic_control" and precursor_id:
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat, "retained": retained, "retention": ret})

        elif cat == "negative_no_trigger" and precursor_id:
            false_total += 1
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            false_hits += int(retained)
            retain_c_total += 1
            retain_c += int(retained)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat, "retained": retained, "retention": ret})

        elif cat == "negative_no_precursor":
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat, "retained": None, "retention": None})

        if plugin is not None and precursor_id and trigger_id:
            prec, trig = host.get_memory(precursor_id), host.get_memory(trigger_id)
            if prec and trig and abs(host.get_timestamp(trig) - host.get_timestamp(prec)) <= config.E_window:
                coalloc_total += 1
                if trigger_id in plugin.get_temporal_coallocates(precursor_id, k=top_k):
                    coalloc_hits += 1

    def _safe_div(a, b):
        return a / b if b else float("nan")

    p_a = _safe_div(retain_a, retain_a_total)
    p_c = _safe_div(retain_c, retain_c_total)
    lift = (p_a - p_c) if not (np.isnan(p_a) or np.isnan(p_c)) else float("nan")

    return MetricResults(
        rescue_recall=_safe_div(rescued_hits, rescued_total),
        false_rescue_rate=_safe_div(false_hits, false_total),
        retention_lift=lift,
        coallocation_recall_at_k=_safe_div(coalloc_hits, coalloc_total),
        n_rescued_precursor=rescued_total,
        n_false_rescue=false_total,
        n_retention_a=retain_a_total,
        n_retention_c=retain_c_total,
        n_coalloc=coalloc_total,
        details={"per_scenario": per_scenario},
    )


def bootstrap_ci(
    values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Paired bootstrap CI -> (point_estimate, lower, upper)."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return point, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
