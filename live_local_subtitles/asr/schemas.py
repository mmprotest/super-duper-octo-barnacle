from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


UTC = timezone.utc


class SegmentState(StrEnum):
    PARTIAL = "partial"
    PROVISIONAL = "provisional"
    FINAL = "final"


class TranslationState(StrEnum):
    DRAFT = "draft"
    FINAL = "final"
    FAILED = "failed"


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    source_lang: str
    text: str
    state: SegmentState
    revision: int
    confidence: float | None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def copy_with(self, **updates: Any) -> "TranscriptSegment":
        payload = {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "source_lang": self.source_lang,
            "text": self.text,
            "state": self.state,
            "revision": self.revision,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        payload.update(updates)
        return TranscriptSegment(**payload)


@dataclass(slots=True)
class TranslationSegment:
    segment_id: str
    source_revision: int
    target_lang: str
    text: str
    state: TranslationState
    latency_ms: int
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    error: str | None = None


@dataclass(slots=True)
class ASRChunk:
    chunk_id: str
    audio: Any
    start_ms: int
    end_ms: int
    is_speech: bool
    emitted_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(slots=True)
class ASRResult:
    chunk_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    source_lang: str
    confidence: float | None
    no_speech_probability: float | None
    emitted_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(slots=True)
class ExportBundle:
    transcripts: list[TranscriptSegment]
    translations: list[TranslationSegment]
