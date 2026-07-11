"""Hindsight Rescue Benchmark (HRB) — paper Section 5."""

from .dataset import (
    DELAY_SECONDS,
    iter_conversation_stream,
    load_locomo_conversations,
    plant_hrb_scenarios,
    subsample_stream,
)
from .metrics import MetricResults, bootstrap_ci, compute_hrb_metrics
from .runner import replay_stream

__all__ = [
    "DELAY_SECONDS",
    "MetricResults",
    "bootstrap_ci",
    "compute_hrb_metrics",
    "iter_conversation_stream",
    "load_locomo_conversations",
    "plant_hrb_scenarios",
    "replay_stream",
    "subsample_stream",
]
