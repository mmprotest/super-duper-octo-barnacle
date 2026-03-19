from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from live_local_subtitles.asr.schemas import ASRChunk
from live_local_subtitles.audio.resample import resample_audio
from live_local_subtitles.audio.ring_buffer import AudioRingBuffer
from live_local_subtitles.audio.vad import EnergyVAD

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional runtime dependency guard
    sd = None


@dataclass(slots=True)
class AudioCaptureConfig:
    device: int | None = None
    channels: int = 1
    input_sample_rate: int = 16_000
    target_sample_rate: int = 16_000
    frame_ms: int = 100
    window_ms: int = 2_000
    step_ms: int = 400
    overlap_ms: int = 1_600
    max_buffer_ms: int = 10_000
    vad_threshold: float = 0.012
    vad_hangover_ms: int = 450


class MicrophoneAudioCapture:
    """Continuously captures microphone audio and emits rolling windows."""

    def __init__(self, config: AudioCaptureConfig, on_chunk: Callable[[ASRChunk], None]) -> None:
        self.config = config
        self.on_chunk = on_chunk
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        capacity_samples = int(config.target_sample_rate * config.max_buffer_ms / 1000)
        self._ring_buffer = AudioRingBuffer(capacity_samples=capacity_samples)
        self._vad = EnergyVAD(
            sample_rate=config.target_sample_rate,
            threshold=config.vad_threshold,
            hangover_ms=config.vad_hangover_ms,
        )
        self._last_emit_total_samples = 0
        self._stream = None

    @staticmethod
    def list_input_devices() -> list[dict[str, str | int | float]]:
        if sd is None:
            return []
        devices = sd.query_devices()
        return [
            {
                "index": idx,
                "name": dev["name"],
                "default_samplerate": dev["default_samplerate"],
                "max_input_channels": dev["max_input_channels"],
            }
            for idx, dev in enumerate(devices)
            if dev.get("max_input_channels", 0) > 0
        ]

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed; microphone capture is unavailable")
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="microphone-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                logger.debug("Ignoring microphone stream close failure", exc_info=True)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.warning("Microphone status: %s", status)
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(mono)

    def _run(self) -> None:
        frame_samples = int(self.config.input_sample_rate * self.config.frame_ms / 1000)
        step_samples = int(self.config.target_sample_rate * self.config.step_ms / 1000)
        window_samples = int(self.config.target_sample_rate * self.config.window_ms / 1000)
        self._stream = sd.InputStream(
            device=self.config.device,
            samplerate=self.config.input_sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=frame_samples,
            callback=self._audio_callback,
        )
        with self._stream:
            while not self._stop_event.is_set():
                try:
                    audio = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.config.input_sample_rate != self.config.target_sample_rate:
                    audio = resample_audio(audio, self.config.input_sample_rate, self.config.target_sample_rate)
                self._ring_buffer.extend(audio)
                total_written = self._ring_buffer.total_written
                if total_written - self._last_emit_total_samples < step_samples:
                    continue
                window = self._ring_buffer.latest(window_samples)
                decision = self._vad.evaluate(window[-step_samples:] if window.size >= step_samples else window)
                end_ms = int(total_written / self.config.target_sample_rate * 1000)
                start_ms = max(0, end_ms - int(window.size / self.config.target_sample_rate * 1000))
                self._last_emit_total_samples = total_written
                self.on_chunk(
                    ASRChunk(
                        chunk_id=str(uuid.uuid4()),
                        audio=window,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        is_speech=decision.is_speech,
                    )
                )
                time.sleep(0.001)
