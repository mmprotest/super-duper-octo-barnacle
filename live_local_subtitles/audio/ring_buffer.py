from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class AudioRingBuffer:
    """Fixed-size mono audio ring buffer optimized for rolling window reads."""

    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self.dtype = dtype
        self._buffer = np.zeros(capacity_samples, dtype=dtype)
        self._write_index = 0
        self._size = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_written(self) -> int:
        return self._total_written

    def extend(self, samples: Iterable[float] | np.ndarray) -> None:
        chunk = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, dtype=self.dtype)
        if chunk.ndim != 1:
            raise ValueError("AudioRingBuffer only supports mono 1D arrays")
        if chunk.size == 0:
            return
        if chunk.size >= self.capacity_samples:
            self._buffer[:] = chunk[-self.capacity_samples :]
            self._write_index = 0
            self._size = self.capacity_samples
            self._total_written += int(chunk.size)
            return

        end_index = self._write_index + chunk.size
        if end_index <= self.capacity_samples:
            self._buffer[self._write_index : end_index] = chunk
        else:
            split = self.capacity_samples - self._write_index
            self._buffer[self._write_index :] = chunk[:split]
            self._buffer[: chunk.size - split] = chunk[split:]
        self._write_index = (self._write_index + chunk.size) % self.capacity_samples
        self._size = min(self.capacity_samples, self._size + chunk.size)
        self._total_written += int(chunk.size)

    def latest(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=self.dtype)
        count = min(count, self._size)
        start = (self._write_index - count) % self.capacity_samples
        if start < self._write_index or self._size < self.capacity_samples:
            return self._buffer[start : start + count].copy()
        return np.concatenate((self._buffer[start:], self._buffer[: self._write_index])).copy()

    def window(self, start_offset: int, count: int) -> np.ndarray:
        if start_offset < 0:
            raise ValueError("start_offset must be >= 0")
        if count <= 0:
            return np.zeros(0, dtype=self.dtype)
        available = min(self._size, self.capacity_samples)
        if start_offset >= available:
            return np.zeros(0, dtype=self.dtype)
        count = min(count, available - start_offset)
        data = self.latest(available)
        return data[start_offset : start_offset + count].copy()
