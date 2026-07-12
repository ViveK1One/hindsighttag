"""Run the Hindsight Rescue Benchmark with ablations (paper Section 5.3).

Outputs (per run, all hyperparameters logged):
  - hrb_results_<ts>.csv / .json   per-run metrics (incl. per-delay breakdowns)
  - hrb_summary_<ts>.csv           bootstrap 95% CIs (baseline vs omega=0), pooled
  - hrb_attributable_<ts>.csv      HindsightTag-attributable deltas (HT - baseline)
  - hrb_ablation_summary_<ts>.csv  bootstrap CIs over the full sweep
  - run_meta_<ts>.json             run metadata

Usage:
  python benchmark/run_hrb.py --seeds 0 1 2 3 4 --conversations 3
  python benchmark/run_hrb.py --full-locomo --seeds 0 1 2 3 4 5 6 7 8 9 --configs primary
  python benchmark/run_hrb.py --streams locomo longmemeval --full-locomo --configs primary
  python benchmark/run_hrb.py --full-ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark"))

from hindsighttag.config import HindsightTagConfig
from hindsighttag.embeddings import get_embedding_model
from hindsighttag.adapters import Mem0Host, VectorStoreHost
from hrb.dataset import (
    iter_conversation_stream,
    load_locomo_conversations,
    plant_hrb_scenarios,
    subsample_stream,
)
from hrb.longmemeval import load_longmemeval_streams
from hrb.metrics import DELAY_ORDER, bootstrap_ci, compute_hrb_metrics
from hrb.runner import replay_stream

DEFAULT_LOCOMO = ROOT / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_LONGMEMEVAL = ROOT / "data" / "longmemeval" / "longmemeval_s.json"

BASE_METRICS = ["rescue_recall", "false_rescue_rate", "retention_lift", "coallocation_recall_at_k"]
DELAY_METRIC_BASES = ["rescue_recall", "false_rescue_rate", "retention_lift", "coallocation_recall_at_k"]
DELAY_METRICS = [f"{m}__{d}" for m in DELAY_METRIC_BASES for d in DELAY_ORDER]
ALL_METRICS = BASE_METRICS + DELAY_METRICS
# Metrics for which an "attributable to HindsightTag" (HT - baseline) delta is meaningful.
ATTRIBUTABLE_METRICS = ["false_rescue_rate", "rescue_recall", "retention_lift"] + \
    [f"false_rescue_rate__{d}" for d in DELAY_ORDER] + \
    [f"rescue_recall__{d}" for d in DELAY_ORDER]


def default_config() -> HindsightTagConfig:
    return HindsightTagConfig()


def ablation_grid(kind: str = "omega") -> List[HindsightTagConfig]:
    """Return the list of +HindsightTag configs to evaluate.

    kind="primary": only omega_assoc=0 (the headline heterosynaptic setting).
    kind="omega":   the five omega_assoc values (0, .25, .5, .75, 1).
    kind="full":    the full Section 5.3 hyperparameter sweep.
    """
    base = default_config()
    if kind == "primary":
        return [HindsightTagConfig(**{**base.to_dict(), "omega_assoc": 0.0})]

    configs = [
        HindsightTagConfig(**{**base.to_dict(), "omega_assoc": omega})
        for omega in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    if kind != "full":
        return configs
    for w_cap in (base.W_capture * 0.25, base.W_capture, base.W_capture * 2):
        for e_win in (base.E_window * 0.5, base.E_window, base.E_window * 2):
            for theta_cap in (0.5, 0.7, 0.9):
                for theta_res in (0.05, 0.15, 0.3):
                    configs.append(
                        HindsightTagConfig(
                            T_tag=base.T_tag,
                            W_capture=w_cap,
                            theta_capture=theta_cap,
                            theta_rescue=theta_res,
                            omega_assoc=0.0,
                            E_window=e_win,
                            theta_link=base.theta_link,
                            delta_rescue=base.delta_rescue,
                        )
                    )
    return configs


def _stable_offset(sample_id: str) -> int:
    """Deterministic per-conversation seed offset (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def _repo_relative(path: Path) -> str:
    """Store paths relative to the repo root so metadata is safe to commit."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def make_host(name: str, seed: int, embedder=None):
    if name == "mem0":
        return Mem0Host(seed=seed, embedder=embedder)
    if name == "vector_store":
        return VectorStoreHost(seed=seed, embedder=embedder)
    raise ValueError(f"Unknown host: {name}")


def _flatten_by_delay(prefix: str, by_delay: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}__{d}": by_delay.get(d, float("nan")) for d in DELAY_ORDER}


def run_single(name, events, manifest, config, seed, use_plugin, embedder=None) -> Dict[str, Any]:
    host = make_host(name, seed=seed, embedder=embedder)
    query_end = max(m["query_timestamp"] for m in manifest) + 1.0
    host, plugin = replay_stream(
        host, events, config=config if use_plugin else None,
        use_hindsighttag=use_plugin, end_time=query_end,
    )
    metrics = compute_hrb_metrics(host, manifest, config, plugin=plugin if use_plugin else None)
    hetero = [
        s["retained"] for s in metrics.details["per_scenario"]
        if s["category"] == "heterosynaptic_control" and s["retained"] is not None
    ]
    row = {
        "host": name,
        "method": f"{name}+HindsightTag" if use_plugin else name,
        "use_hindsighttag": use_plugin,
        "seed": seed,
        **config.to_dict(),
        "rescue_recall": metrics.rescue_recall,
        "false_rescue_rate": metrics.false_rescue_rate,
        "retention_lift": metrics.retention_lift,
        "coallocation_recall_at_k": metrics.coallocation_recall_at_k,
        "heterosynaptic_recall": float(np.mean(hetero)) if hetero else float("nan"),
        "n_rescued_precursor": metrics.n_rescued_precursor,
        "n_false_rescue": metrics.n_false_rescue,
        "n_coalloc": metrics.n_coalloc,
        "mem0_library_available": getattr(host, "mem0_available", None) if name == "mem0" else None,
    }
    row.update(_flatten_by_delay("rescue_recall", metrics.rescue_recall_by_delay))
    row.update(_flatten_by_delay("false_rescue_rate", metrics.false_rescue_rate_by_delay))
    row.update(_flatten_by_delay("retention_lift", metrics.retention_lift_by_delay))
    row.update(_flatten_by_delay("coallocation_recall_at_k", metrics.coallocation_recall_by_delay))
    return row


def iter_streams(stream_names, locomo_path, longmemeval_path, n_conversations, full_locomo):
    """Yield (stream_name, sample_id, base_events) across the requested streams.

    ``full_locomo`` uses every LoCoMo conversation; LongMemEval always honours
    ``n_conversations`` (capped) because it ships 500 instances and would otherwise
    dominate runtime.
    """
    for stream in stream_names:
        if stream == "locomo":
            conversations = load_locomo_conversations(locomo_path)
            if not full_locomo and n_conversations and n_conversations > 0:
                conversations = conversations[:n_conversations]
            for conv_idx, conv in enumerate(conversations):
                sample_id = conv.get("sample_id", f"conv_{conv_idx}")
                base_events = subsample_stream(iter_conversation_stream(conv, sample_id))
                yield stream, sample_id, base_events
        elif stream == "longmemeval":
            if not Path(longmemeval_path).exists():
                print(f"[skip] LongMemEval not found at {longmemeval_path}. "
                      f"Run: python benchmark/download_longmemeval.py")
                continue
            cap = n_conversations if (n_conversations and n_conversations > 0) else 10
            streams = load_longmemeval_streams(longmemeval_path, cap)
            for sample_id, base_events in streams:
                yield stream, sample_id, subsample_stream(base_events)
        else:
            raise ValueError(f"Unknown stream: {stream}")


def run_experiments(args):
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = ablation_grid(args.configs)
    hosts = ["mem0", "vector_store"]
    embedder = get_embedding_model(seed=42)
    rows: List[Dict[str, Any]] = []

    stream_list = list(
        iter_streams(args.streams, args.locomo, args.longmemeval,
                     args.conversations, args.full_locomo)
    )
    if not stream_list:
        raise SystemExit("No conversation streams available. Check --streams and data paths.")

    for stream_name, sample_id, base_events in stream_list:
        for seed in args.seeds:
            rng = np.random.default_rng(seed + _stable_offset(sample_id))
            events, manifest = plant_hrb_scenarios(base_events, rng, args.scenarios_per_category)
            if not manifest:
                continue
            for host_name in hosts:
                base_row = {**run_single(host_name, events, manifest, default_config(),
                                         seed, False, embedder),
                            "conversation": sample_id, "stream": stream_name, "config_id": "baseline"}
                rows.append(base_row)
                for cfg_idx, config in enumerate(configs):
                    ht_row = {**run_single(host_name, events, manifest, config, seed, True, embedder),
                              "conversation": sample_id, "stream": stream_name,
                              "config_id": f"ablation_{cfg_idx}"}
                    rows.append(ht_row)

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"hrb_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    df.to_json(output_dir / f"hrb_results_{ts}.json", orient="records", indent=2)

    present_metrics = [m for m in ALL_METRICS if m in df.columns]

    def summarize(sub, label):
        out = []
        for method in sub["method"].unique():
            block = sub[sub["method"] == method]
            for metric in present_metrics:
                point, lo, hi = bootstrap_ci(block[metric].dropna().tolist(), seed=42)
                out.append({"comparison_group": label, "method": method, "metric": metric,
                            "point_estimate": point, "ci_lower": lo, "ci_upper": hi,
                            "n_runs": int(block[metric].notna().sum())})
        return pd.DataFrame(out)

    # ---- Full-ablation summary (all +HindsightTag configs) ----
    summarize(df[df["use_hindsighttag"]], "full_ablation").to_csv(
        output_dir / f"hrb_ablation_summary_{ts}.csv", index=False)

    # ---- Primary summary: baseline vs omega=0 (ablation_0), pooled across streams ----
    primary = pd.concat([
        df[df["config_id"] == "baseline"],
        df[(df["config_id"] == "ablation_0") & df["use_hindsighttag"]],
    ])
    summary_path = output_dir / f"hrb_summary_{ts}.csv"
    summarize(primary, "primary_omega0").to_csv(summary_path, index=False)

    # ---- Attributable deltas: (HT - baseline) paired on (stream, conversation, seed) ----
    attributable_rows = compute_attributable(primary, hosts)
    attr_path = output_dir / f"hrb_attributable_{ts}.csv"
    pd.DataFrame(attributable_rows).to_csv(attr_path, index=False)

    with (output_dir / f"run_meta_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": ts,
            "streams": args.streams,
            "locomo_path": _repo_relative(args.locomo),
            "longmemeval_path": _repo_relative(args.longmemeval),
            "full_locomo": args.full_locomo,
            "n_conversations_requested": args.conversations,
            "n_stream_conversations": len(stream_list),
            "seeds": args.seeds,
            "scenarios_per_category": args.scenarios_per_category,
            "configs": args.configs,
            "n_configs": len(configs),
            "summary_csv": _repo_relative(summary_path),
            "attributable_csv": _repo_relative(attr_path),
            "results_csv": _repo_relative(csv_path),
        }, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {attr_path}")
    return df, summary_path, attr_path


def compute_attributable(primary: pd.DataFrame, hosts: List[str]) -> List[Dict[str, Any]]:
    """Isolate HindsightTag's own contribution by pairing +HT vs baseline.

    For each host and metric, compute the per-(stream, conversation, seed) difference
    (WITH HindsightTag) - (baseline host alone), then bootstrap the differences. This
    separates the mechanism's effect from host-intrinsic behaviour (e.g. a non-decaying
    vector store retaining negative controls on its own).
    """
    keys = ["stream", "conversation", "seed"]
    out: List[Dict[str, Any]] = []
    metrics = [m for m in ATTRIBUTABLE_METRICS if m in primary.columns]
    for host in hosts:
        base = primary[(primary["host"] == host) & (~primary["use_hindsighttag"])]
        ht = primary[(primary["host"] == host) & (primary["use_hindsighttag"])]
        merged = base.merge(ht, on=keys, suffixes=("_base", "_ht"))
        for metric in metrics:
            diffs = (merged[f"{metric}_ht"] - merged[f"{metric}_base"]).dropna().tolist()
            point, lo, hi = bootstrap_ci(diffs, seed=42)
            out.append({
                "host": host,
                "metric": metric,
                "attributable_point": point,
                "attributable_ci_lower": lo,
                "attributable_ci_upper": hi,
                "n_pairs": len(diffs),
            })
    return out


def main():
    p = argparse.ArgumentParser(description="Run the Hindsight Rescue Benchmark")
    p.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    p.add_argument("--longmemeval", type=Path, default=DEFAULT_LONGMEMEVAL)
    p.add_argument("--streams", nargs="+", default=["locomo"],
                   choices=["locomo", "longmemeval"])
    p.add_argument("--output", type=Path, default=ROOT / "benchmark" / "results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--conversations", type=int, default=3,
                   help="Number of conversations per stream (ignored if --full-locomo).")
    p.add_argument("--full-locomo", action="store_true",
                   help="Use every conversation in each stream.")
    p.add_argument("--scenarios-per-category", type=int, default=2)
    p.add_argument("--configs", choices=["primary", "omega", "full"], default="omega",
                   help="primary=omega0 only; omega=5 omega values; full=Section 5.3 sweep.")
    p.add_argument("--full-ablation", action="store_true",
                   help="Alias for --configs full.")
    args = p.parse_args()
    if args.full_ablation:
        args.configs = "full"

    if "locomo" in args.streams and not args.locomo.exists():
        raise SystemExit(
            f"LoCoMo not found at {args.locomo}. Run: python benchmark/download_data.py")
    run_experiments(args)


if __name__ == "__main__":
    main()
