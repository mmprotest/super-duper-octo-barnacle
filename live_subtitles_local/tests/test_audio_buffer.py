import numpy as np

from live_subtitles_local.pipeline.audio_buffer import RollingAudioBuffer


def test_rolling_audio_buffer_keeps_latest_audio() -> None:
    buffer = RollingAudioBuffer(sample_rate=10, max_seconds=1.0)
    buffer.append(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32), timestamp=1.0)
    buffer.append(np.array([7, 8, 9, 10, 11], dtype=np.float32), timestamp=2.0)

    latest, start_index, end_index = buffer.read_latest(10)
    snapshot = buffer.snapshot()

    assert latest.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert start_index == 1
    assert end_index == 11
    assert snapshot.fill_samples == 10
    assert snapshot.dropped_samples == 1
    assert snapshot.dropped_chunks == 1
