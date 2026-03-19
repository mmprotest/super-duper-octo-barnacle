from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(slots=True)
class AudioBufferSnapshot:
    write_index: int
    oldest_index: int
    fill_samples: int
    capacity_samples: int
    latest_timestamp: float | None
    dropped_samples: int
    dropped_chunks: int


class RollingAudioBuffer:
    """Thread-safe bounded PCM ring buffer that keeps the newest audio."""

    def __init__(self, sample_rate: int, max_seconds: float) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self._chunks: deque[tuple[int, np.ndarray, float]] = deque()
        self._fill_samples = 0
        self._write_index = 0
        self._dropped_samples = 0
        self._dropped_chunks = 0
        self._max_seconds = max_seconds

    @property
    def capacity_samples(self) -> int:
        return max(int(self.sample_rate * self._max_seconds), 1)

    def reconfigure(self, max_seconds: float) -> None:
        with self._lock:
            self._max_seconds = max_seconds
            self._trim_to_capacity()

    def append(self, samples: np.ndarray, timestamp: float | None = None) -> None:
        if samples.size == 0:
            return
        chunk = np.ascontiguousarray(samples, dtype=np.float32)
        ts = timestamp or time.time()
        with self._lock:
            self._chunks.append((self._write_index, chunk, ts))
            self._write_index += int(chunk.shape[0])
            self._fill_samples += int(chunk.shape[0])
            self._trim_to_capacity()

    def snapshot(self) -> AudioBufferSnapshot:
        with self._lock:
            oldest_index = self._chunks[0][0] if self._chunks else self._write_index
            latest_timestamp = self._chunks[-1][2] if self._chunks else None
            return AudioBufferSnapshot(
                write_index=self._write_index,
                oldest_index=oldest_index,
                fill_samples=self._fill_samples,
                capacity_samples=self.capacity_samples,
                latest_timestamp=latest_timestamp,
                dropped_samples=self._dropped_samples,
                dropped_chunks=self._dropped_chunks,
            )

    def clear(self, reset_counters: bool = True) -> None:
        with self._lock:
            self._chunks.clear()
            self._fill_samples = 0
            self._write_index = 0
            if reset_counters:
                self._dropped_samples = 0
                self._dropped_chunks = 0

    def read_latest(self, sample_count: int) -> tuple[np.ndarray, int, int]:
        if sample_count <= 0:
            snapshot = self.snapshot()
            return np.array([], dtype=np.float32), snapshot.write_index, snapshot.write_index
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32), self._write_index, self._write_index
            requested_end = self._write_index
            requested_start = max(self._chunks[0][0], requested_end - sample_count)
            remaining_start = requested_start
            parts: list[np.ndarray] = []
            for chunk_start, chunk, _ in self._chunks:
                chunk_end = chunk_start + int(chunk.shape[0])
                if chunk_end <= remaining_start:
                    continue
                if chunk_start >= requested_end:
                    break
                local_start = max(remaining_start - chunk_start, 0)
                local_end = min(requested_end - chunk_start, int(chunk.shape[0]))
                if local_end > local_start:
                    parts.append(chunk[local_start:local_end])
                    remaining_start = chunk_start + local_end
                if remaining_start >= requested_end:
                    break
            audio = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
            return np.ascontiguousarray(audio, dtype=np.float32), requested_start, requested_end

    def _trim_to_capacity(self) -> None:
        capacity = self.capacity_samples
        while self._fill_samples > capacity and self._chunks:
            chunk_start, chunk, timestamp = self._chunks[0]
            overflow = self._fill_samples - capacity
            if overflow >= chunk.shape[0]:
                self._chunks.popleft()
                self._fill_samples -= int(chunk.shape[0])
                self._dropped_samples += int(chunk.shape[0])
                self._dropped_chunks += 1
                continue
            trimmed = np.ascontiguousarray(chunk[overflow:], dtype=np.float32)
            self._chunks[0] = (chunk_start + overflow, trimmed, timestamp)
            self._fill_samples -= overflow
            self._dropped_samples += overflow
            self._dropped_chunks += 1
            break
