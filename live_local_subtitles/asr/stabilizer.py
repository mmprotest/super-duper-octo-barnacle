from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher

from live_local_subtitles.asr.schemas import ASRResult, SegmentState, TranscriptSegment


UTC = timezone.utc


@dataclass(slots=True)
class StabilizerConfig:
    stability_similarity: float = 0.88
    provisional_repeats: int = 2
    final_silence_ms: int = 900
    max_segment_gap_ms: int = 1_200


@dataclass(slots=True)
class _TrackedSegment:
    segment: TranscriptSegment
    consecutive_matches: int = 0
    last_text: str = ""
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class TranscriptStabilizer:
    """Turns noisy overlapping ASR windows into stable segment revisions."""

    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self._tracked: dict[str, _TrackedSegment] = {}

    def update(self, result: ASRResult) -> TranscriptSegment:
        now = datetime.now(tz=UTC)
        tracked = self._tracked.get(result.segment_id)
        if tracked is None:
            segment = TranscriptSegment(
                segment_id=result.segment_id,
                start_ms=result.start_ms,
                end_ms=result.end_ms,
                source_lang=result.source_lang,
                text=result.text.strip(),
                state=SegmentState.PARTIAL,
                revision=1,
                confidence=result.confidence,
                created_at=now,
                updated_at=now,
            )
            tracked = _TrackedSegment(segment=segment, last_text=segment.text, last_seen_at=now)
            self._tracked[result.segment_id] = tracked
            return segment

        similarity = self._similarity(tracked.last_text, result.text)
        next_revision = tracked.segment.revision + 1 if result.text.strip() != tracked.segment.text else tracked.segment.revision
        if similarity >= self.config.stability_similarity:
            tracked.consecutive_matches += 1
        else:
            tracked.consecutive_matches = 0

        state = SegmentState.PARTIAL
        if tracked.consecutive_matches >= self.config.provisional_repeats:
            state = SegmentState.PROVISIONAL
        if tracked.segment.state == SegmentState.FINAL:
            state = SegmentState.FINAL

        tracked.last_text = result.text.strip()
        tracked.last_seen_at = now
        tracked.segment = tracked.segment.copy_with(
            start_ms=min(tracked.segment.start_ms, result.start_ms),
            end_ms=max(tracked.segment.end_ms, result.end_ms),
            text=result.text.strip(),
            state=state,
            revision=next_revision,
            confidence=result.confidence,
            updated_at=now,
        )
        return tracked.segment

    def mark_final_due_to_pause(self, latest_end_ms: int) -> list[TranscriptSegment]:
        finalized: list[TranscriptSegment] = []
        now = datetime.now(tz=UTC)
        for segment_id, tracked in list(self._tracked.items()):
            if tracked.segment.state == SegmentState.FINAL:
                continue
            silence_ms = latest_end_ms - tracked.segment.end_ms
            if silence_ms >= self.config.final_silence_ms:
                tracked.segment = tracked.segment.copy_with(
                    state=SegmentState.FINAL,
                    revision=tracked.segment.revision + 1,
                    updated_at=now,
                )
                finalized.append(tracked.segment)
            elif latest_end_ms - tracked.segment.start_ms > self.config.max_segment_gap_ms and tracked.segment.state == SegmentState.PROVISIONAL:
                tracked.segment = tracked.segment.copy_with(
                    state=SegmentState.FINAL,
                    revision=tracked.segment.revision + 1,
                    updated_at=now,
                )
                finalized.append(tracked.segment)
        return finalized

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        return SequenceMatcher(a=left.strip(), b=right.strip()).ratio()
