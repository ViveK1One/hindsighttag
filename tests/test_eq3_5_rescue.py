"""Eqs. 3-5 in isolation: capture detection and backward rescue (Section 4.3)."""

from hindsighttag import HindsightTagConfig, HindsightTagPlugin, MemoryItem
from hindsighttag.adapters import DecayMemoryHost


def _plugin(**overrides):
    params = dict(
        T_tag=3600.0, W_capture=7200.0, theta_capture=0.7,
        theta_rescue=0.1, omega_assoc=0.0, E_window=3600.0, delta_rescue=1.0,
    )
    params.update(overrides)
    return HindsightTagPlugin(DecayMemoryHost(consolidation_threshold=0.4), HindsightTagConfig(**params))


def test_high_salience_trigger_rescues_low_salience_precursor():
    plugin = _plugin()
    plugin.store(MemoryItem("prec", "pricing offhand", 1000.0, salience=0.2, retention=0.25))
    before = plugin.host.get_memory("prec").retention
    plugin.store(MemoryItem("trig", "pricing approved", 1000.0 + 1800.0, salience=0.95, retention=0.95))
    after = plugin.host.get_memory("prec").retention
    assert after > before
    assert len(plugin.rescue_log) == 1
    assert plugin.rescue_log[0]["mode"] == "heterosynaptic"


def test_low_salience_event_does_not_trigger_capture():
    plugin = _plugin()
    plugin.store(MemoryItem("prec", "low", 1000.0, salience=0.2, retention=0.25))
    plugin.store(MemoryItem("trig", "still low", 1000.0 + 100.0, salience=0.3, retention=0.3))
    assert plugin.rescue_log == []


def test_capture_strength_below_theta_rescue_is_skipped():
    # Very high theta_rescue: capture detected but rescue rejected.
    plugin = _plugin(theta_rescue=0.99)
    plugin.store(MemoryItem("prec", "x", 1000.0, salience=0.2, retention=0.25))
    plugin.store(MemoryItem("trig", "y", 1000.0 + 1800.0, salience=0.95, retention=0.95))
    assert plugin.rescue_log == []


def test_omega_one_requires_relatedness():
    # Pure associative: unrelated trigger should not rescue when Rel is ~0.
    plugin = _plugin(omega_assoc=1.0, theta_rescue=0.3)
    plugin.store(MemoryItem("prec", "quantum chromodynamics lattice gauge", 1000.0,
                            salience=0.2, retention=0.25))
    plugin.store(MemoryItem("trig", "pineapple smoothie recipe", 1000.0 + 600.0,
                            salience=0.98, retention=0.98))
    assert plugin.rescue_log == []


def test_non_interference_when_no_capture():
    plugin = _plugin()
    plugin.store(MemoryItem("prec", "z", 1000.0, salience=0.2, retention=0.25))
    v = plugin.host.get_memory("prec").retention
    plugin.tick(1000.0 + 10000.0)  # tag expires with no trigger
    # Retention only changed by the host's own decay, never increased by rescue.
    assert plugin.host.get_memory("prec") is None or plugin.host.get_memory("prec").retention <= v
    assert plugin.rescue_log == []
