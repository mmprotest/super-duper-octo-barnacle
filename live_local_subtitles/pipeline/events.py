from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


UTC = timezone.utc


class EventType(StrEnum):
    AUDIO_CHUNK = "audio_chunk"
    ASR_RESULT = "asr_result"
    TRANSCRIPT_UPDATED = "transcript_updated"
    TRANSLATION_REQUESTED = "translation_requested"
    TRANSLATION_UPDATED = "translation_updated"
    STATUS = "status"
    ERROR = "error"


@dataclass(slots=True)
class PipelineEvent:
    type: EventType
    payload: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
