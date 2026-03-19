from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from live_local_subtitles.asr.schemas import SegmentState, TranscriptSegment, TranslationSegment, TranslationState
from live_local_subtitles.translation.cache import TranslationCache
from live_local_subtitles.translation.llm_client import LocalOpenAICompatibleClient
from live_local_subtitles.translation.prompts import build_translation_messages

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass(slots=True)
class TranslatorConfig:
    enabled: bool = True
    debounce_ms: int = 300
    request_timeout_s: float = 6.0
    context_segments: int = 3
    stale_revision_margin: int = 0


class TranslationWorker:
    """Single-flight translator that prioritizes freshest subtitle segments."""

    def __init__(
        self,
        client: LocalOpenAICompatibleClient,
        config: TranslatorConfig,
        context_provider: Callable[[], Sequence[TranscriptSegment]],
        on_result: Callable[[TranslationSegment], None],
    ) -> None:
        self.client = client
        self.config = config
        self.context_provider = context_provider
        self.on_result = on_result
        self.cache = TranslationCache()
        self._queue: queue.LifoQueue[TranscriptSegment] = queue.LifoQueue(maxsize=8)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_revision_by_segment: dict[str, int] = {}
        self._target_lang = "English"

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="translation-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def set_target_lang(self, target_lang: str) -> None:
        self._target_lang = target_lang

    def submit(self, segment: TranscriptSegment) -> None:
        if not self.config.enabled or segment.state == SegmentState.PARTIAL:
            return
        self._latest_revision_by_segment[segment.segment_id] = segment.revision
        cached = self.cache.get(segment.segment_id, segment.revision, target_lang=self._target_lang)
        if cached is not None:
            self.on_result(cached)
            return
        try:
            self._queue.put_nowait(segment)
        except queue.Full:
            self._drop_stale_jobs()
            self._queue.put_nowait(segment)

    def _drop_stale_jobs(self) -> None:
        retained: list[TranscriptSegment] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            latest_revision = self._latest_revision_by_segment.get(item.segment_id, item.revision)
            if item.revision >= latest_revision - self.config.stale_revision_margin:
                retained.append(item)
        for item in retained[: self._queue.maxsize - 1]:
            self._queue.put_nowait(item)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                segment = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            latest_revision = self._latest_revision_by_segment.get(segment.segment_id, segment.revision)
            if segment.revision < latest_revision:
                continue
            time.sleep(self.config.debounce_ms / 1000)
            latest_revision = self._latest_revision_by_segment.get(segment.segment_id, segment.revision)
            if segment.revision < latest_revision:
                continue
            started = time.perf_counter()
            created_at = datetime.now(tz=UTC)
            try:
                context = list(self.context_provider())[-self.config.context_segments :]
                text = self.client.translate(build_translation_messages(segment, self._target_lang, context))
                state = TranslationState.FINAL if segment.state == SegmentState.FINAL else TranslationState.DRAFT
                translated = TranslationSegment(
                    segment_id=segment.segment_id,
                    source_revision=segment.revision,
                    target_lang=self._target_lang,
                    text=text,
                    state=state,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    created_at=created_at,
                    updated_at=datetime.now(tz=UTC),
                )
                self.cache.put(translated)
            except Exception as exc:
                logger.warning("Translation failed for %s: %s", segment.segment_id, exc)
                translated = TranslationSegment(
                    segment_id=segment.segment_id,
                    source_revision=segment.revision,
                    target_lang=self._target_lang,
                    text="",
                    state=TranslationState.FAILED,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    created_at=created_at,
                    updated_at=datetime.now(tz=UTC),
                    error=str(exc),
                )
            self.on_result(translated)
