from __future__ import annotations

import numpy as np

from live_local_subtitles.audio.ring_buffer import AudioRingBuffer



def test_ring_buffer_returns_latest_samples_in_order() -> None:
    buffer = AudioRingBuffer(capacity_samples=5)
    buffer.extend(np.array([1, 2, 3], dtype=np.float32))
    buffer.extend(np.array([4, 5, 6], dtype=np.float32))

    latest = buffer.latest(5)

    assert latest.tolist() == [2, 3, 4, 5, 6]



def test_ring_buffer_window_reads_offset_from_oldest_available() -> None:
    buffer = AudioRingBuffer(capacity_samples=6)
    buffer.extend(np.array([10, 11, 12, 13, 14, 15], dtype=np.float32))

    window = buffer.window(2, 3)

    assert window.tolist() == [12, 13, 14]
