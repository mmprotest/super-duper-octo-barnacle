from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from live_local_subtitles.asr.schemas import TranscriptSegment, TranslationSegment

UTC = timezone.utc


@dataclass(slots=True)
class RuntimeMetrics:
    started_at: datetime | None = None
    last_audio_chunk_ms: int | None = None
    asr_latency_ms: int | None = None
    translation_latency_ms: int | None = None
    backlog_audio: int = 0
    backlog_translation: int = 0
    dropped_translation_jobs: int = 0
    last_error: str | None = None


@dataclass(slots=True)
class AppState:
    transcripts: dict[str, TranscriptSegment] = field(default_factory=dict)
    translations: dict[str, TranslationSegment] = field(default_factory=dict)
    history_order: list[str] = field(default_factory=list)
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    running: bool = False
    target_lang: str = "English"

    def upsert_transcript(self, segment: TranscriptSegment) -> None:
        if segment.segment_id not in self.transcripts:
            self.history_order.append(segment.segment_id)
        existing = self.transcripts.get(segment.segment_id)
        if existing is None or segment.revision >= existing.revision:
            self.transcripts[segment.segment_id] = segment

    def upsert_translation(self, segment: TranslationSegment) -> None:
        existing = self.translations.get(segment.segment_id)
        if existing is None or segment.source_revision >= existing.source_revision:
            self.translations[segment.segment_id] = segment

    def ordered_transcripts(self) -> list[TranscriptSegment]:
        return [self.transcripts[key] for key in self.history_order if key in self.transcripts]

    def ordered_translations(self) -> list[TranslationSegment]:
        return [self.translations[key] for key in self.history_order if key in self.translations]
