"""Run the Hindsight Rescue Benchmark with ablations (paper Section 5.3).

Outputs (per run, all hyperparameters logged):
  - hrb_results_<ts>.csv / .json   per-run metrics
  - hrb_summary_<ts>.csv           bootstrap 95% CIs (baseline vs omega=0)
  - hrb_ablation_summary_<ts>.csv  bootstrap CIs over the full sweep
  - run_meta_<ts>.json             run metadata

Usage:
  python benchmark/run_hrb.py --seeds 0 1 2 3 4 --conversations 3
  python benchmark/run_hrb.py --full-ablation
"""

from __future__ import annotations

import argparse
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
from hrb.metrics import bootstrap_ci, compute_hrb_metrics
from hrb.runner import replay_stream

DEFAULT_LOCOMO = ROOT / "data" / "locomo" / "data" / "locomo10.json"


def default_config() -> HindsightTagConfig:
    return HindsightTagConfig()


def ablation_grid(full: bool = False) -> List[HindsightTagConfig]:
    base = default_config()
    configs = [
        HindsightTagConfig(**{**base.to_dict(), "omega_assoc": omega})
        for omega in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    if not full:
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
    return {
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


def run_experiments(locomo_path, output_dir, seeds, n_conversations, scenarios_per_category, full_ablation):
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations = load_locomo_conversations(locomo_path)[:n_conversations]
    configs = ablation_grid(full=full_ablation)
    hosts = ["mem0", "vector_store"]
    embedder = get_embedding_model(seed=42)
    rows: List[Dict[str, Any]] = []

    for conv_idx, conv in enumerate(conversations):
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        base_events = subsample_stream(iter_conversation_stream(conv, sample_id))
        for seed in seeds:
            rng = np.random.default_rng(seed + conv_idx)
            events, manifest = plant_hrb_scenarios(base_events, rng, scenarios_per_category)
            with (output_dir / f"manifest_{sample_id}_seed{seed}.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            for host_name in hosts:
                rows.append({**run_single(host_name, events, manifest, default_config(), seed, False, embedder),
                             "conversation": sample_id, "config_id": "baseline"})
                for cfg_idx, config in enumerate(configs):
                    rows.append({**run_single(host_name, events, manifest, config, seed, True, embedder),
                                 "conversation": sample_id, "config_id": f"ablation_{cfg_idx}"})

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"hrb_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    df.to_json(output_dir / f"hrb_results_{ts}.json", orient="records", indent=2)

    metric_cols = ["rescue_recall", "false_rescue_rate", "retention_lift", "coallocation_recall_at_k"]

    def summarize(sub, label):
        out = []
        for method in sub["method"].unique():
            block = sub[sub["method"] == method]
            for metric in metric_cols:
                point, lo, hi = bootstrap_ci(block[metric].dropna().tolist(), seed=42)
                out.append({"comparison_group": label, "method": method, "metric": metric,
                            "point_estimate": point, "ci_lower": lo, "ci_upper": hi,
                            "n_runs": int(block[metric].notna().sum())})
        return pd.DataFrame(out)

    summarize(df[df["use_hindsighttag"]], "full_ablation").to_csv(
        output_dir / f"hrb_ablation_summary_{ts}.csv", index=False)
    primary = pd.concat([
        df[df["config_id"] == "baseline"],
        df[(df["config_id"] == "ablation_0") & df["use_hindsighttag"]],
    ])
    summary_path = output_dir / f"hrb_summary_{ts}.csv"
    summarize(primary, "primary_omega0").to_csv(summary_path, index=False)

    with (output_dir / f"run_meta_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": ts, "locomo_path": _repo_relative(locomo_path), "n_conversations": n_conversations,
            "seeds": seeds, "scenarios_per_category": scenarios_per_category,
            "full_ablation": full_ablation, "n_configs": len(configs),
            "summary_csv": _repo_relative(summary_path), "results_csv": _repo_relative(csv_path),
        }, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return df


def main():
    p = argparse.ArgumentParser(description="Run the Hindsight Rescue Benchmark")
    p.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    p.add_argument("--output", type=Path, default=ROOT / "benchmark" / "results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--conversations", type=int, default=3)
    p.add_argument("--scenarios-per-category", type=int, default=2)
    p.add_argument("--full-ablation", action="store_true")
    args = p.parse_args()
    if not args.locomo.exists():
        raise SystemExit(
            f"LoCoMo not found at {args.locomo}. Run: python benchmark/download_data.py")
    run_experiments(args.locomo, args.output, args.seeds, args.conversations,
                    args.scenarios_per_category, args.full_ablation)


if __name__ == "__main__":
    main()
