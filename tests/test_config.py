"""Config validation and per-run logging (Section 4.8 reproducibility checklist item 1)."""

import pytest

from hindsighttag import HindsightTagConfig


def test_defaults_are_the_six_hyperparameters():
    cfg = HindsightTagConfig()
    d = cfg.to_dict()
    for key in ("T_tag", "W_capture", "theta_capture", "theta_rescue", "omega_assoc", "E_window"):
        assert key in d


def test_omega_assoc_out_of_range_rejected():
    with pytest.raises(ValueError):
        HindsightTagConfig(omega_assoc=1.5)


def test_negative_window_rejected():
    with pytest.raises(ValueError):
        HindsightTagConfig(W_capture=-1.0)
