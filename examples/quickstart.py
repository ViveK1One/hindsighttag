"""Minimal HindsightTag quickstart: attach to a host and rescue a precursor.

Run:  python examples/quickstart.py
"""

from hindsighttag import HindsightTagConfig, HindsightTagPlugin, MemoryItem
from hindsighttag.adapters import DecayMemoryHost

# 1. Any host implementing the Section 4.8 interface. Here: a decaying store.
host = DecayMemoryHost(consolidation_threshold=0.4)

# 2. Attach HindsightTag with calibration-seed hyperparameters.
plugin = HindsightTagPlugin(host, HindsightTagConfig(T_tag=6 * 3600, W_capture=7 * 86400))

# 3. Store a low-salience "precursor" — it would normally just decay away.
plugin.store(MemoryItem("m1", "Offhand: we might revisit the pricing plan.",
                        timestamp=0.0, salience=0.2, retention=0.25))

# 4. Hours later, a high-salience, related "trigger" arrives.
plugin.tick(3 * 3600)
plugin.store(MemoryItem("m2", "IMPORTANT: Leadership approved the pricing initiative.",
                        timestamp=3 * 3600, salience=0.95, retention=0.95))

# 5. The trigger retroactively rescued the precursor (Eqs. 3-5).
rescued = host.get_memory("m1")
print(f"precursor retention after rescue: {rescued.retention:.3f}")
print(f"rescued flag: {rescued.consolidated}")
for record in plugin.rescue_log:
    print("provenance:", record)

# 6. Temporal co-allocation (Eq. 6): "what else happened around this time".
print("co-allocates of m1:", plugin.get_temporal_coallocates("m1"))
