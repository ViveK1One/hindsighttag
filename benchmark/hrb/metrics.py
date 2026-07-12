"""HRB metrics (paper Section 5.2) and paired bootstrap confidence intervals.

Metric-correctness notes (2026 revision)
----------------------------------------
* **Co-Alloc Recall@k** is measured against a *scenario-defined* ground truth that
  is independent of ``E_window``: the true temporal partner of a planted precursor
  is its own scenario's trigger, and every precursor/trigger scenario is scored at
  *all four* delays. The earlier definition ("any pair within ``E_window``") was
  circular — Eq. 6 mechanically creates an edge for exactly those pairs, so recall
  was pinned at 1.00. Under the corrected definition the mechanism must surface the
  true trigger in the top-k against the full pool of temporally-registered memories,
  and out-of-window pairs (all delays > ``E_window``) are genuine misses.
* Every metric is also broken out **by delay bucket** (1 hour / 1 day / 1 week /
  1 month) so a blended headline number cannot hide a strong dependence on the
  shortest delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hindsighttag.config import HindsightTagConfig
from hindsighttag.core import HindsightTagPlugin

DELAY_ORDER = ["1_hour", "1_day", "1_week", "1_month"]


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
    # Per-delay breakdowns (key -> value); NaN when a bucket has no scenarios.
    rescue_recall_by_delay: Dict[str, float] = field(default_factory=dict)
    false_rescue_rate_by_delay: Dict[str, float] = field(default_factory=dict)
    retention_lift_by_delay: Dict[str, float] = field(default_factory=dict)
    coallocation_recall_by_delay: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


def _is_retained(host, memory_id: str, query: str, top_k: int = 5) -> Tuple[bool, float]:
    item = host.get_memory(memory_id)
    if item is None or getattr(item, "tombstoned", False):
        return False, 0.0
    retention = host.retention_at(memory_id) if hasattr(host, "retention_at") else item.retention
    retrievable = host.is_retrievable(memory_id, query, top_k=top_k)
    return retrievable, retention


def _new_buckets() -> Dict[str, List[int]]:
    return {d: [0, 0] for d in DELAY_ORDER}  # delay -> [hits, total]


def _recall_dict(buckets: Dict[str, List[int]]) -> Dict[str, float]:
    return {
        d: (hits / total if total else float("nan"))
        for d, (hits, total) in buckets.items()
    }


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

    rescue_by_delay = _new_buckets()
    false_by_delay = _new_buckets()
    coalloc_by_delay = _new_buckets()

    for s in manifest:
        cat, query = s["category"], s["query"]
        delay = s.get("delay")
        precursor_id, trigger_id = s.get("precursor_id"), s.get("trigger_id")

        if cat == "rescued_precursor" and precursor_id:
            rescued_total += 1
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            rescued_hits += int(retained)
            retain_a_total += 1
            retain_a += int(retained)
            if delay in rescue_by_delay:
                rescue_by_delay[delay][1] += 1
                rescue_by_delay[delay][0] += int(retained)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat,
                                 "delay": delay, "retained": retained, "retention": ret})

        elif cat == "heterosynaptic_control" and precursor_id:
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat,
                                 "delay": delay, "retained": retained, "retention": ret})

        elif cat == "negative_no_trigger" and precursor_id:
            false_total += 1
            retained, ret = _is_retained(host, precursor_id, query, top_k)
            false_hits += int(retained)
            retain_c_total += 1
            retain_c += int(retained)
            if delay in false_by_delay:
                false_by_delay[delay][1] += 1
                false_by_delay[delay][0] += int(retained)
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat,
                                 "delay": delay, "retained": retained, "retention": ret})

        elif cat == "negative_no_precursor":
            per_scenario.append({"scenario_id": s["scenario_id"], "category": cat,
                                 "delay": delay, "retained": None, "retention": None})

        # ---- Co-Alloc Recall@k (independent, non-circular ground truth) ----
        # The scenario's own trigger is the true temporal partner of the precursor.
        # We score EVERY precursor/trigger scenario at ALL delays and require the
        # mechanism to surface the true trigger in the top-k co-allocates against
        # the full pool of temporally registered memories. Out-of-window pairs
        # (delay > E_window) yield no edge and are therefore genuine misses --
        # this is what breaks the previous 1.00 tautology.
        if plugin is not None and precursor_id and trigger_id:
            coalloc_total += 1
            neigh = plugin.get_temporal_coallocates(precursor_id, k=top_k)
            hit = int(trigger_id in neigh)
            coalloc_hits += hit
            if delay in coalloc_by_delay:
                coalloc_by_delay[delay][1] += 1
                coalloc_by_delay[delay][0] += hit

    def _safe_div(a, b):
        return a / b if b else float("nan")

    p_a = _safe_div(retain_a, retain_a_total)
    p_c = _safe_div(retain_c, retain_c_total)
    lift = (p_a - p_c) if not (np.isnan(p_a) or np.isnan(p_c)) else float("nan")

    # Per-delay retention lift = P(retained | rescued precursor) - P(retained | no-trigger).
    rescue_recall_by_delay = _recall_dict(rescue_by_delay)
    false_rescue_rate_by_delay = _recall_dict(false_by_delay)
    lift_by_delay = {
        d: (rescue_recall_by_delay[d] - false_rescue_rate_by_delay[d])
        if not (np.isnan(rescue_recall_by_delay[d]) or np.isnan(false_rescue_rate_by_delay[d]))
        else float("nan")
        for d in DELAY_ORDER
    }

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
        rescue_recall_by_delay=rescue_recall_by_delay,
        false_rescue_rate_by_delay=false_rescue_rate_by_delay,
        retention_lift_by_delay=lift_by_delay,
        coallocation_recall_by_delay=_recall_dict(coalloc_by_delay),
        details={"per_scenario": per_scenario},
    )


def bootstrap_ci(
    values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap CI -> (point_estimate, lower, upper) over a list of per-run values."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return point, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
