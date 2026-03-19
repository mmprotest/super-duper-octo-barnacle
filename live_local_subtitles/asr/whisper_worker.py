from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from live_local_subtitles.asr.schemas import ASRChunk, ASRResult

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional runtime dependency guard
    WhisperModel = None  # type: ignore[assignment]


@dataclass(slots=True)
class WhisperConfig:
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    best_of: int = 1
    language: str | None = None
    task: str = "transcribe"
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    no_speech_threshold: float = 0.5
    word_timestamps: bool = False
    temperature: float = 0.0


class FasterWhisperWorker:
    """Background ASR worker that keeps a single warm Whisper model loaded."""

    def __init__(self, config: WhisperConfig, on_result) -> None:  # noqa: ANN001
        self.config = config
        self.on_result = on_result
        self._model: Any | None = None
        self._queue: queue.Queue[ASRChunk] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="faster-whisper-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def submit(self, chunk: ASRChunk) -> None:
        if not chunk.is_speech:
            return
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def _run(self) -> None:
        self._ensure_model()
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                result = self._transcribe_chunk(chunk)
            except Exception:
                logger.exception("ASR inference failed")
                continue
            if result is not None:
                latency_ms = int((time.perf_counter() - started) * 1000)
                logger.debug("ASR result for %s in %sms", chunk.chunk_id, latency_ms)
                self.on_result(result)

    def _transcribe_chunk(self, chunk: ASRChunk) -> ASRResult | None:
        assert self._model is not None
        segments, info = self._model.transcribe(
            np.asarray(chunk.audio, dtype=np.float32),
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            language=self.config.language,
            task=self.config.task,
            condition_on_previous_text=self.config.condition_on_previous_text,
            vad_filter=self.config.vad_filter,
            no_speech_threshold=self.config.no_speech_threshold,
            word_timestamps=self.config.word_timestamps,
            temperature=self.config.temperature,
        )
        text_parts: list[str] = []
        min_start = chunk.start_ms
        max_end = chunk.end_ms
        avg_logprob = None
        no_speech_probability = getattr(info, "no_speech_prob", None)
        for seg in segments:
            content = getattr(seg, "text", "").strip()
            if not content:
                continue
            text_parts.append(content)
            min_start = min(min_start, int(float(getattr(seg, "start", 0.0)) * 1000) + chunk.start_ms)
            max_end = max(max_end, int(float(getattr(seg, "end", 0.0)) * 1000) + chunk.start_ms)
            avg_logprob = getattr(seg, "avg_logprob", avg_logprob)
        if not text_parts:
            return None
        confidence = None if avg_logprob is None else max(0.0, min(1.0, 1.0 + float(avg_logprob) / 5.0))
        segment_id = f"seg-{chunk.end_ms // 1000:08d}"
        return ASRResult(
            chunk_id=chunk.chunk_id,
            segment_id=segment_id,
            start_ms=min_start,
            end_ms=max_end,
            text=" ".join(text_parts).strip(),
            source_lang=getattr(info, "language", self.config.language or "auto"),
            confidence=confidence,
            no_speech_probability=no_speech_probability,
        )
