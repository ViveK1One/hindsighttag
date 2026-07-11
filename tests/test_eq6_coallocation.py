"""Eq. 6 in isolation: temporal co-allocation link weight and graph (Section 4.5)."""

import math

from hindsighttag.core import TemporalGraph


def test_link_weight_decays_with_gap():
    e_window = 3600.0
    w = TemporalGraph.link_weight(0.0, 1800.0, e_window)
    assert abs(w - math.exp(-1800.0 / e_window)) < 1e-9


def test_link_weight_zero_outside_window():
    assert TemporalGraph.link_weight(0.0, 4000.0, 3600.0) == 0.0


def test_link_weight_one_at_zero_gap():
    assert TemporalGraph.link_weight(500.0, 500.0, 3600.0) == 1.0


def test_graph_creates_symmetric_edges_within_window():
    g = TemporalGraph(e_window=3600.0, theta_link=0.1)
    g.register("a", 0.0)
    g.register("b", 1800.0)
    g.register("c", 100000.0)  # far away -> no edge
    assert "b" in g.neighbours("a")
    assert "a" in g.neighbours("b")
    assert "c" not in g.neighbours("a")
