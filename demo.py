"""Minimal HindsightTag demo — same as examples/quickstart.py.

Run from the repository root:

    python demo.py
"""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parent / "examples" / "quickstart.py"))
