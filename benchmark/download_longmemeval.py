"""Fetch LongMemEval, the optional second HRB base conversation stream.

LongMemEval: https://github.com/xiaowu0162/LongMemEval
Paper: Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory" (2024/2025).

LongMemEval is distributed by its authors on Hugging Face
(https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) under the MIT
license — it is NOT bundled or redistributed by HindsightTag. This helper tries
the Hugging Face mirror, preferring the smaller ``oracle`` split (evidence
sessions only) so the download stays lightweight; if that is unavailable, it
prints manual instructions. Only session turns are read as chronological filler
for HRB (no QA questions/answers or needle labels are used).

The fetched file is normalised to:  data/longmemeval/longmemeval_s.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "data" / "longmemeval"
TARGET_FILE = TARGET_DIR / "longmemeval_s.json"

# (repo_id, filename) in preference order: oracle is smallest (evidence sessions only).
HF_CANDIDATES = [
    ("xiaowu0162/longmemeval-cleaned", "longmemeval_oracle.json"),
    ("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json"),
    ("xiaowu0162/longmemeval", "longmemeval_oracle.json"),
    ("xiaowu0162/longmemeval", "longmemeval_s.json"),
]

MANUAL_NOTICE = f"""
Could not auto-download LongMemEval.

Please download it from the official Hugging Face repository (MIT license):
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

and place it at:
  {TARGET_FILE}

Then re-run:
  python benchmark/run_hrb.py --streams locomo longmemeval --full-locomo --configs primary
"""


def main() -> None:
    if TARGET_FILE.exists():
        print(f"LongMemEval already present: {TARGET_FILE}")
        return
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print("huggingface_hub not installed (pip install huggingface_hub).")
        print(MANUAL_NOTICE)
        sys.exit(1)

    for repo_id, filename in HF_CANDIDATES:
        try:
            print(f"Trying Hugging Face: {repo_id}/{filename} ...")
            local = hf_hub_download(repo_id=repo_id, filename=filename,
                                    repo_type="dataset", local_dir=str(TARGET_DIR))
            Path(local).replace(TARGET_FILE)
            print(f"Ready: {TARGET_FILE}")
            return
        except Exception as exc:  # noqa: BLE001 - best-effort fetch
            print(f"  failed: {exc}")

    print(MANUAL_NOTICE)
    sys.exit(1)


if __name__ == "__main__":
    main()
