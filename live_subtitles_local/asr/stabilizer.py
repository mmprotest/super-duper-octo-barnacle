from __future__ import annotations

from dataclasses import replace
import time
import uuid

from live_subtitles_local.asr.schemas import TranscriptSegment


class TranscriptStabilizer:
    """Merge incremental ASR text into revisable segments with partial/provisional/final states."""

    def __init__(self, provisional_debounce_ms: int, final_silence_ms: int, material_change_ratio: float = 0.12) -> None:
        self.provisional_debounce_ms = provisional_debounce_ms
        self.final_silence_ms = final_silence_ms
        self.material_change_ratio = material_change_ratio
        self._current: TranscriptSegment | None = None
        self._last_text_change_at: float | None = None
        self._last_audio_at: float | None = None

    @property
    def current_segment(self) -> TranscriptSegment | None:
        return self._current

    def ingest_update(
        self,
        text: str,
        start_ms: int,
        end_ms: int,
        source_lang: str | None,
        confidence: float | None,
        now: float | None = None,
    ) -> TranscriptSegment | None:
        now = now or time.time()
        cleaned = " ".join(text.split())
        self._last_audio_at = now
        if not cleaned:
            return self._maybe_transition(now)

        if self._current is None or self._current.state == "final":
            self._current = TranscriptSegment(
                segment_id=str(uuid.uuid4()),
                start_ms=start_ms,
                end_ms=end_ms,
                source_lang=source_lang,
                text=cleaned,
                state="partial",
                revision=0,
                confidence=confidence,
                created_at=now,
                updated_at=now,
            )
            self._last_text_change_at = now
            return replace(self._current)

        changed = cleaned != self._current.text
        revision = self._current.revision
        if changed and self._is_material_change(self._current.text, cleaned):
            revision += 1
            self._last_text_change_at = now
        elif changed and self._last_text_change_at is None:
            self._last_text_change_at = now

        self._current = replace(
            self._current,
            end_ms=max(end_ms, self._current.end_ms),
            source_lang=source_lang or self._current.source_lang,
            text=cleaned,
            state="partial" if changed else self._current.state,
            revision=revision,
            confidence=confidence,
            updated_at=now,
        )
        return replace(self._current)

    def on_silence(self, now: float | None = None) -> TranscriptSegment | None:
        now = now or time.time()
        return self._maybe_transition(now)

    def flush_final(self, now: float | None = None) -> TranscriptSegment | None:
        now = now or time.time()
        if self._current is None:
            return None
        self._current = replace(self._current, state="final", updated_at=now)
        segment = replace(self._current)
        return segment

    def _maybe_transition(self, now: float) -> TranscriptSegment | None:
        if self._current is None:
            return None
        last_text_change_at = self._last_text_change_at or self._current.updated_at
        last_audio_at = self._last_audio_at or self._current.updated_at
        stable_ms = int((now - last_text_change_at) * 1000)
        silent_ms = int((now - last_audio_at) * 1000)

        next_state = self._current.state
        if self._current.text and stable_ms >= self.provisional_debounce_ms and self._current.state == "partial":
            next_state = "provisional"
        if self._current.text and silent_ms >= self.final_silence_ms:
            next_state = "final"

        if next_state != self._current.state:
            self._current = replace(self._current, state=next_state, updated_at=now)
            return replace(self._current)
        return None

    def _is_material_change(self, old: str, new: str) -> bool:
        if not old:
            return True
        old_tokens = old.split()
        new_tokens = new.split()
        delta = abs(len(new_tokens) - len(old_tokens))
        changed_words = sum(1 for a, b in zip(old_tokens, new_tokens) if a != b)
        ratio = (delta + changed_words) / max(len(old_tokens), 1)
        return ratio >= self.material_change_ratio
