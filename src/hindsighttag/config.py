"""Hyperparameter configuration for HindsightTag (paper Section 4.8).

The six mechanism hyperparameters have **no validated defaults** — the paper
reports them as literature-informed *starting ranges* for calibration, not
production values (Section 4.8, "Literature-informed starting ranges"). The
defaults below are calibration seeds only. Every field is logged per run so
that any result is fully reproducible (Section 4.8, reproducibility checklist).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class HindsightTagConfig:
    """The six HindsightTag hyperparameters plus a few operational scalars.

    Attributes
    ----------
    T_tag:
        Eq. 2 tag-decay time constant (seconds). The synaptic tag is a
        short-lived, minutes-to-hours phenomenon; the paper seeds this at the
        classical synaptic-tagging window of one to a few hours
        (Frey & Morris, 1997; Redondo & Morris, 2011).
    W_capture:
        Eq. 2-3 bounded lookback/lookahead window (seconds). Seeded somewhat
        longer than ``T_tag`` because the behaviorally relevant window can
        extend considerably further (Lin et al., 2025).
    theta_capture:
        Eq. 3 salience threshold above which an arriving event triggers a
        capture event. Free parameter; no neuroscience analogue.
    theta_rescue:
        Eq. 4-5 minimum capture strength C_i required for a rescue. Free
        parameter; no neuroscience analogue.
    omega_assoc:
        Eq. 4 heterosynaptic-to-associative mixing weight in [0, 1].
        ``0`` = pure heterosynaptic capture (content-independent, per
        Moncada & Viola, 2007); ``1`` = purely associative capture.
    E_window:
        Eq. 6 temporal co-allocation eligibility window (seconds). Free
        parameter; motivated by engram-allocation findings
        (Yiu et al., 2014; Rashid et al., 2016; Josselyn & Frankland, 2024).
    theta_link:
        Eq. 6 minimum co-allocation edge weight retained for retrieval.
    delta_rescue:
        Eq. 5 rescue increment scale. The paper leaves the exact form
        unspecified; logged per run.
    """

    # --- The six paper hyperparameters (Section 4.8) ---
    T_tag: float = 6 * 3600.0            # ~6 hours
    W_capture: float = 35 * 24 * 3600.0  # ~35 days (covers the 1-month HRB delay)
    theta_capture: float = 0.7
    theta_rescue: float = 0.10
    omega_assoc: float = 0.0             # heterosynaptic by default
    E_window: float = 4 * 3600.0         # ~4 hours

    # --- Operational scalars (implementation details, also logged) ---
    theta_link: float = 0.1
    delta_rescue: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Return all fields as a plain dict for per-run logging."""
        return asdict(self)

    def __post_init__(self) -> None:
        if not 0.0 <= self.omega_assoc <= 1.0:
            raise ValueError("omega_assoc must be in [0, 1]")
        for name in ("T_tag", "W_capture", "E_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
