"""Eq. 2 in isolation: synaptic tag value and decay (Section 4.8 checklist item 2)."""

import math

from hindsighttag.core import SynapticTag


def test_tag_value_at_write_time_equals_g0():
    tag = SynapticTag("m", tau_i=100.0, g0=0.3, T_tag=3600.0, expiry_time=100.0 + 7200.0)
    assert tag.value(100.0) == 0.3


def test_tag_decays_exponentially():
    tag = SynapticTag("m", tau_i=0.0, g0=0.5, T_tag=3600.0, expiry_time=7200.0)
    assert math.isclose(tag.value(3600.0), 0.5 * math.exp(-1.0), rel_tol=1e-9)


def test_tag_zero_before_write_and_after_window():
    tag = SynapticTag("m", tau_i=100.0, g0=0.5, T_tag=3600.0, expiry_time=100.0 + 7200.0)
    assert tag.value(50.0) == 0.0        # before write
    assert tag.value(100.0 + 7201.0) == 0.0  # past capture window


def test_is_active_tracks_consolidation():
    tag = SynapticTag("m", tau_i=0.0, g0=0.5, T_tag=3600.0, expiry_time=7200.0)
    assert tag.is_active(100.0)
    tag.consolidated = True
    assert not tag.is_active(100.0)
