"""LoCoMo loader + Hindsight Rescue Benchmark scenario injection (paper Section 5.1).

Base conversation stream: the public LoCoMo dataset
(https://github.com/snap-research/locomo), used under CC BY-NC 4.0.
LoCoMo is NOT redistributed by this package — users must fetch it via
``benchmark/download_data.py``. Only text *turns* are read as chronological
filler; images and QA annotations are not used.

HRB planted precursor/trigger text is synthetic (see PRECURSOR_* / TRIGGER_*
templates below) and is original to this project.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATE_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
    re.IGNORECASE,
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_locomo_datetime(text: str) -> float:
    m = DATE_PATTERN.search(text.strip())
    if not m:
        return 0.0
    hour, minute, ampm, day, month_name, year = m.groups()
    hour_i, minute_i = int(hour), int(minute)
    if ampm.lower() == "pm" and hour_i != 12:
        hour_i += 12
    if ampm.lower() == "am" and hour_i == 12:
        hour_i = 0
    month = MONTH_MAP.get(month_name.lower(), 1)
    return datetime(int(year), month, int(day), hour_i, minute_i).timestamp()


@dataclass
class StreamEvent:
    memory_id: str
    content: str
    timestamp: float
    salience: float
    retention: float
    metadata: Dict[str, Any]
    is_planted: bool = False
    scenario_id: Optional[str] = None
    scenario_category: Optional[str] = None


def load_locomo_conversations(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_conversation_stream(
    conversation: Dict[str, Any], sample_id: str, base_offset: float = 0.0
) -> List[StreamEvent]:
    conv = conversation.get("conversation", conversation)
    events: List[StreamEvent] = []
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        session_ts = parse_locomo_datetime(conv.get(f"{sk}_date_time", "")) + base_offset
        for idx, turn in enumerate(conv.get(sk, [])):
            text = turn.get("text", "").strip()
            if not text:
                continue
            dia_id = turn.get("dia_id", f"{sk}:{idx}")
            events.append(
                StreamEvent(
                    memory_id=f"{sample_id}_{dia_id}",
                    content=text,
                    timestamp=session_ts + idx * 60.0,
                    salience=0.55,
                    retention=0.55,
                    metadata={"speaker": turn.get("speaker"), "dia_id": dia_id, "sample_id": sample_id},
                )
            )
    events.sort(key=lambda e: e.timestamp)
    return events


DELAY_SECONDS = {
    "1_hour": 3600.0,
    "1_day": 86400.0,
    "1_week": 7 * 86400.0,
    "1_month": 30 * 86400.0,
}

PRECURSOR_TEMPLATES_RELATED = [
    "Offhand remark HRB: we might revisit the {topic} plan next quarter.",
    "Casual note HRB: someone mentioned adjusting {topic} policies eventually.",
    "Low-key aside HRB: the {topic} idea came up briefly in hallway chat.",
]
PRECURSOR_TEMPLATES_UNRELATED = [
    "Offhand remark HRB: the office coffee machine needs a filter change.",
    "Casual note HRB: I might try a new lunch spot downtown tomorrow.",
    "Low-key aside HRB: the weather looks rainy for the weekend hike.",
]
TRIGGER_TEMPLATES_RELATED = [
    "IMPORTANT HRB: Leadership approved the {topic} initiative — effective immediately.",
    "CRITICAL HRB: The {topic} decision is finalized and announced company-wide.",
    "BREAKING HRB: {topic} rollout begins today after executive sign-off.",
]
TRIGGER_TEMPLATES_UNRELATED = [
    "IMPORTANT HRB: Company won a major industry award for innovation.",
    "CRITICAL HRB: New flagship product launch exceeds all sales targets.",
    "BREAKING HRB: Record-breaking quarterly earnings announced today.",
]
TOPICS = [
    "pricing", "hiring", "security", "expansion", "compliance",
    "partnership", "budget", "restructuring", "marketing", "platform",
]


def _pick(rng, items: List[str]) -> str:
    return items[int(rng.integers(0, len(items)))]


def subsample_stream(events: List[StreamEvent], max_filler: int = 80) -> List[StreamEvent]:
    """Keep all planted events; subsample native LoCoMo filler for tractable runtime."""
    planted = [e for e in events if e.is_planted]
    filler = [e for e in events if not e.is_planted]
    if len(filler) <= max_filler:
        return sorted(events, key=lambda e: e.timestamp)
    step = max(1, len(filler) // max_filler)
    return sorted(filler[::step][:max_filler] + planted, key=lambda e: e.timestamp)


def plant_hrb_scenarios(
    base_events: List[StreamEvent], rng, scenarios_per_category: int = 4
) -> Tuple[List[StreamEvent], List[Dict[str, Any]]]:
    """Inject the four HRB scenario categories (a-d) at each delay (Section 5.1)."""
    if not base_events:
        return [], []

    manifest: List[Dict[str, Any]] = []
    planted: List[StreamEvent] = []
    t_min = base_events[0].timestamp
    span = max(base_events[-1].timestamp - t_min, 1.0)
    categories = [
        "rescued_precursor",
        "heterosynaptic_control",
        "negative_no_trigger",
        "negative_no_precursor",
    ]

    scenario_idx = 0
    for delay_name, delay_sec in DELAY_SECONDS.items():
        for category in categories:
            for _ in range(scenarios_per_category):
                topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
                precursor_ts = t_min + float(rng.uniform(0.15, 0.75)) * span
                trigger_ts = precursor_ts + delay_sec
                query_ts = trigger_ts + min(delay_sec * 0.1, 86400.0)
                sid = f"hrb_{delay_name}_{scenario_idx}"
                scenario_idx += 1
                precursor_id = f"{sid}_precursor"

                if category == "rescued_precursor":
                    trigger_content = _pick(rng, TRIGGER_TEMPLATES_RELATED).format(topic=topic)
                elif category == "heterosynaptic_control":
                    trigger_content = _pick(rng, TRIGGER_TEMPLATES_UNRELATED).format(topic=topic)
                elif category == "negative_no_trigger":
                    trigger_content = None
                else:  # negative_no_precursor
                    trigger_content = _pick(rng, TRIGGER_TEMPLATES_RELATED).format(topic=topic)

                if category != "negative_no_precursor":
                    planted.append(
                        StreamEvent(
                            memory_id=precursor_id,
                            content=_pick(rng, PRECURSOR_TEMPLATES_RELATED).format(topic=topic),
                            timestamp=precursor_ts,
                            salience=0.2,
                            retention=0.25,
                            metadata={"topic": topic, "delay": delay_name, "hrb_planted": True},
                            is_planted=True,
                            scenario_id=sid,
                            scenario_category=category,
                        )
                    )

                trigger_id = f"{sid}_trigger"
                trigger = None
                if trigger_content is not None:
                    trigger = StreamEvent(
                        memory_id=trigger_id,
                        content=trigger_content,
                        timestamp=trigger_ts,
                        salience=0.92,
                        retention=0.92,
                        metadata={"topic": topic, "delay": delay_name, "hrb_planted": True},
                        is_planted=True,
                        scenario_id=sid,
                        scenario_category=category,
                    )
                    planted.append(trigger)

                manifest.append(
                    {
                        "scenario_id": sid,
                        "category": category,
                        "delay": delay_name,
                        "delay_seconds": delay_sec,
                        "topic": topic,
                        "precursor_id": precursor_id if category != "negative_no_precursor" else None,
                        "trigger_id": trigger_id if trigger is not None else None,
                        "query": f"HRB {topic} precursor decision",
                        "query_timestamp": query_ts,
                        "precursor_timestamp": precursor_ts if category != "negative_no_precursor" else None,
                        "trigger_timestamp": trigger_ts if trigger is not None else None,
                    }
                )

    return sorted(base_events + planted, key=lambda e: e.timestamp), manifest
