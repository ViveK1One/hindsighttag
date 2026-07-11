"""Fetch the public LoCoMo dataset used as the HRB base conversation stream.

LoCoMo: https://github.com/snap-research/locomo
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
         https://creativecommons.org/licenses/by-nc/4.0/

By running this script you agree to use LoCoMo under its license terms:
  - Attribute the LoCoMo authors (Maharana et al., ACL 2024)
  - Non-commercial use only (unless you obtain separate permission from Snap Research)
  - Do not redistribute the dataset except as permitted by CC BY-NC 4.0

LoCoMo is NOT bundled in the HindsightTag repository; each user downloads it
directly from the upstream source.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCOMO_DIR = ROOT / "data" / "locomo"
LOCOMO_FILE = LOCOMO_DIR / "data" / "locomo10.json"

LICENSE_NOTICE = """
LoCoMo license reminder (CC BY-NC 4.0):
  - Cite: Maharana et al., 2024 (arXiv:2402.17753)
  - Non-commercial use only unless you have explicit permission from Snap Research
  - See README.md section "Data & third-party attribution"
"""


def main() -> None:
    if LOCOMO_FILE.exists():
        print(f"LoCoMo already present: {LOCOMO_FILE}")
        print(LICENSE_NOTICE)
        return
    LOCOMO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning LoCoMo (snap-research/locomo)...")
    print(LICENSE_NOTICE)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/snap-research/locomo.git", str(LOCOMO_DIR)],
        check=True,
    )
    if LOCOMO_FILE.exists():
        print(f"Ready: {LOCOMO_FILE}")
    else:
        sys.exit("Clone completed but locomo10.json not found; check the repository layout.")


if __name__ == "__main__":
    main()
