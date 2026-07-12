"""LongMemEval loader for the Hindsight Rescue Benchmark (second base stream).

LongMemEval (Wu et al., 2024/2025; https://github.com/xiaowu0162/LongMemEval) is a
long-term-memory evaluation set whose instances contain many timestamped chat
*sessions*. We read only the session turns as chronological filler for HRB — exactly
as we do with LoCoMo — and never use the QA questions/answers or the "needle" labels.
The dataset is NOT redistributed here; fetch it via ``benchmark/download_longmemeval.py``.

Each LongMemEval instance is treated as one "conversation" so it slots into the same
HRB machinery as a LoCoMo conversation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import StreamEvent

# LongMemEval haystack_dates look like "2023/05/20 (Sat) 02:21".
_DATE_FORMATS = [
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def _parse_date(text: Optional[str], fallback_index: int) -> float:
    """Parse a LongMemEval date string; fall back to synthetic daily spacing."""
    if text:
        cleaned = text.strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cleaned, fmt).timestamp()
            except ValueError:
                continue
    # Deterministic synthetic timeline: one session per day from a fixed epoch.
    return datetime(2023, 1, 1).timestamp() + fallback_index * 86400.0


def _turn_text(turn: Any) -> str:
    if isinstance(turn, dict):
        return str(turn.get("content") or turn.get("text") or "").strip()
    return str(turn).strip()


def _instance_to_events(instance: Dict[str, Any], sample_id: str) -> List[StreamEvent]:
    sessions = (
        instance.get("haystack_sessions")
        or instance.get("sessions")
        or []
    )
    dates = instance.get("haystack_dates") or instance.get("dates") or []
    events: List[StreamEvent] = []
    for s_idx, session in enumerate(sessions):
        date_str = dates[s_idx] if s_idx < len(dates) else None
        session_ts = _parse_date(date_str, s_idx)
        turns = session if isinstance(session, list) else session.get("turns", [])
        for t_idx, turn in enumerate(turns):
            text = _turn_text(turn)
            if not text:
                continue
            events.append(
                StreamEvent(
                    memory_id=f"{sample_id}_s{s_idx}_t{t_idx}",
                    content=text,
                    timestamp=session_ts + t_idx * 60.0,
                    salience=0.55,
                    retention=0.55,
                    metadata={"session": s_idx, "sample_id": sample_id},
                )
            )
    events.sort(key=lambda e: e.timestamp)
    return events


def load_longmemeval_streams(
    path: Path, n_conversations: int = 0
) -> List[Tuple[str, List[StreamEvent]]]:
    """Return [(sample_id, base_events), ...], one entry per LongMemEval instance."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or list(data.values())
    if n_conversations and n_conversations > 0:
        data = data[:n_conversations]

    streams: List[Tuple[str, List[StreamEvent]]] = []
    for idx, instance in enumerate(data):
        sample_id = str(
            instance.get("question_id")
            or instance.get("id")
            or f"lme_{idx}"
        )
        events = _instance_to_events(instance, sample_id)
        if events:
            streams.append((sample_id, events))
    return streams
