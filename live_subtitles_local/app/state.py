from __future__ import annotations

from dataclasses import dataclass, field, replace
import threading
from typing import Any

from live_subtitles_local.asr.schemas import SessionConfig, TranscriptSegment, TranslationSegment


@dataclass(slots=True)
class WorkerHealth:
    rtc_running: bool = False
    asr_loaded: bool = False
    translation_ready: bool = False


@dataclass(slots=True)
class MetricsSnapshot:
    audio_queue_size: int = 0
    translation_queue_size: int = 0
    last_asr_latency_ms: int | None = None
    last_translation_latency_ms: int | None = None


@dataclass(slots=True)
class AppSnapshot:
    config: SessionConfig
    live_transcript: TranscriptSegment | None
    transcript_history: list[TranscriptSegment]
    translation_history: list[TranslationSegment]
    current_translation: TranslationSegment | None
    worker_health: WorkerHealth
    metrics: MetricsSnapshot
    last_error_message: str | None
    debug: dict[str, Any] = field(default_factory=dict)


class ThreadSafeAppState:
    def __init__(self, config: SessionConfig) -> None:
        self._lock = threading.RLock()
        self._config = config
        self._live_transcript: TranscriptSegment | None = None
        self._transcript_history: list[TranscriptSegment] = []
        self._translation_history: list[TranslationSegment] = []
        self._current_translation: TranslationSegment | None = None
        self._worker_health = WorkerHealth()
        self._metrics = MetricsSnapshot()
        self._last_error_message: str | None = None
        self._debug: dict[str, Any] = {}

    def snapshot(self) -> AppSnapshot:
        with self._lock:
            return AppSnapshot(
                config=replace(self._config),
                live_transcript=replace(self._live_transcript) if self._live_transcript else None,
                transcript_history=[replace(item) for item in self._transcript_history],
                translation_history=[replace(item) for item in self._translation_history],
                current_translation=replace(self._current_translation) if self._current_translation else None,
                worker_health=replace(self._worker_health),
                metrics=replace(self._metrics),
                last_error_message=self._last_error_message,
                debug=dict(self._debug),
            )

    def update_config(self, config: SessionConfig) -> None:
        with self._lock:
            self._config = config

    def upsert_live_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._live_transcript = segment
            if segment.state == "final":
                if not self._transcript_history or self._transcript_history[-1].segment_id != segment.segment_id:
                    self._transcript_history.append(segment)
                else:
                    self._transcript_history[-1] = segment

    def upsert_translation(self, translation: TranslationSegment) -> None:
        with self._lock:
            self._current_translation = translation
            if translation.state == "final":
                if not self._translation_history or self._translation_history[-1].segment_id != translation.segment_id:
                    self._translation_history.append(translation)
                else:
                    self._translation_history[-1] = translation
            self._metrics.last_translation_latency_ms = translation.latency_ms

    def set_worker_health(self, **kwargs: bool) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._worker_health, key, value)

    def set_metrics(self, **kwargs: int | None) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._metrics, key, value)

    def set_error(self, message: str | None) -> None:
        with self._lock:
            self._last_error_message = message

    def set_debug(self, key: str, value: Any) -> None:
        with self._lock:
            self._debug[key] = value
