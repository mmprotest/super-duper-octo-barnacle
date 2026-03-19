from __future__ import annotations

from dataclasses import dataclass, field

from live_local_subtitles.asr.schemas import TranslationSegment


@dataclass(slots=True)
class TranslationCache:
    """Tracks latest translated revision to avoid redundant LLM calls."""

    _segments: dict[tuple[str, int, str], TranslationSegment] = field(default_factory=dict)

    def get(self, segment_id: str, revision: int, target_lang: str) -> TranslationSegment | None:
        return self._segments.get((segment_id, revision, target_lang))

    def put(self, segment: TranslationSegment) -> None:
        self._segments[(segment.segment_id, segment.source_revision, segment.target_lang)] = segment
