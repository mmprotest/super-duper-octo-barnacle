from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from live_subtitles_local.rtc.stream_events import AudioFrameEvent

TARGET_SAMPLE_RATE = 16000


def normalize_audio_frame(event: AudioFrameEvent) -> np.ndarray:
    samples = event.samples
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.dtype == np.int16:
        normalized = samples.astype(np.float32) / 32768.0
    else:
        normalized = samples.astype(np.float32)
    if event.sample_rate != TARGET_SAMPLE_RATE:
        normalized = _resample_linear(normalized, event.sample_rate, TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.shape[0] / source_rate
    target_length = max(int(duration * target_rate), 1)
    old_positions = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


@dataclass
class AudioChunkAccumulator:
    target_chunk_ms: int = 800
    sample_rate: int = TARGET_SAMPLE_RATE

    def __post_init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque()
        self._sample_count = 0

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.target_chunk_ms / 1000)

    def add_frame(self, audio_frame: np.ndarray) -> list[np.ndarray]:
        self._buffer.append(audio_frame)
        self._sample_count += audio_frame.shape[0]
        chunks: list[np.ndarray] = []
        while self._sample_count >= self.target_samples:
            chunks.append(self._pop_chunk(self.target_samples))
        return chunks

    def flush(self) -> np.ndarray:
        return self._pop_chunk(self._sample_count) if self._sample_count else np.array([], dtype=np.float32)

    def _pop_chunk(self, size: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=np.float32)
        remaining = size
        parts: list[np.ndarray] = []
        while remaining > 0 and self._buffer:
            head = self._buffer[0]
            if head.shape[0] <= remaining:
                parts.append(self._buffer.popleft())
                self._sample_count -= head.shape[0]
                remaining -= head.shape[0]
            else:
                parts.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                self._sample_count -= remaining
                remaining = 0
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


@dataclass(slots=True)
class SilenceDetector:
    threshold: float = 0.008

    def is_silent(self, audio_frame: np.ndarray) -> bool:
        if audio_frame.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio_frame))))
        return rms < self.threshold

    def boundary_hint(self, audio_frame: np.ndarray) -> bool:
        # Silence currently acts as the segment boundary cue for chunked local transcription.
        return self.is_silent(audio_frame)
