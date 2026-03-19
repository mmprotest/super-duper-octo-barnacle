from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from live_subtitles_local.app.state import ThreadSafeAppState
from live_subtitles_local.asr.schemas import SessionConfig, TranscriptSegment
from live_subtitles_local.asr.stabilizer import TranscriptStabilizer
from live_subtitles_local.asr.whisper_worker import WhisperWorker
from live_subtitles_local.config.loader import load_yaml_file
from live_subtitles_local.pipeline.audio_buffer import RollingAudioBuffer
from live_subtitles_local.pipeline.queues import PipelineQueues, TranslationJob
from live_subtitles_local.pipeline.scheduler import PipelineScheduler
from live_subtitles_local.rtc.audio_handlers import TARGET_SAMPLE_RATE
from live_subtitles_local.rtc.stream_events import AudioFrameEvent
from live_subtitles_local.translation.llm_client import OpenAICompatibleClient
from live_subtitles_local.translation.translator import SegmentTranslator

logger = logging.getLogger(__name__)


class LiveSubtitleOrchestrator:
    def __init__(self, config_path: str) -> None:
        self.config = self.load_config(config_path)
        self.state = ThreadSafeAppState(self.config)
        self.queues = PipelineQueues(queue.Queue(maxsize=self.config.translation_queue_size))
        self.scheduler = PipelineScheduler(self.queues)
        self.audio_buffer = RollingAudioBuffer(TARGET_SAMPLE_RATE, self.config.audio_ring_buffer_seconds)
        self.stabilizer = TranscriptStabilizer(
            provisional_debounce_ms=self.config.provisional_debounce_ms,
            final_silence_ms=self.config.final_silence_ms,
        )
        self._asr_worker: WhisperWorker | None = None
        self._translator_client: OpenAICompatibleClient | None = None
        self._translator: SegmentTranslator | None = None
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._is_running = False
        self._metrics_lock = threading.RLock()
        self._asr_latency_total_ms = 0
        self._asr_latency_samples = 0
        self._last_processed_window_end = 0
        self._ensure_workers()
        self._update_buffer_metrics(receiver_queue_capacity=self.config.receiver_queue_size)

    @staticmethod
    def _load_raw_config(config_path: str) -> dict[str, Any]:
        return load_yaml_file(config_path)

    @classmethod
    def load_config(cls, config_path: str) -> SessionConfig:
        raw = cls._load_raw_config(config_path)
        return SessionConfig(**{key: value for key, value in raw.items() if key in SessionConfig.__dataclass_fields__})

    @property
    def running(self) -> bool:
        return self._is_running

    def _ensure_workers(self) -> None:
        self.state.set_error(None)
        try:
            self._asr_worker = WhisperWorker(
                model_name=self.config.whisper_model_name,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
            self.state.set_worker_health(asr_loaded=True)
        except Exception as exc:
            self._asr_worker = None
            self.state.set_worker_health(asr_loaded=False)
            self.state.set_error(str(exc))
            logger.warning("ASR worker unavailable: %s", exc)

        self._translator_client = OpenAICompatibleClient(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model_name=self.config.llm_model_name,
        )
        ready, message = self._translator_client.check_health()
        self._translator = SegmentTranslator(self._translator_client)
        self.state.set_worker_health(translation_ready=ready, translator_last_error=message)

    def start(self) -> None:
        if self._threads:
            return
        self.audio_buffer.clear(reset_counters=True)
        self._last_processed_window_end = 0
        with self._metrics_lock:
            self._asr_latency_total_ms = 0
            self._asr_latency_samples = 0
        self.state.set_metrics(
            audio_queue_size=0,
            translation_queue_size=0,
            receiver_queue_size=0,
            last_asr_latency_ms=0,
            average_asr_latency_ms=0,
            asr_worker_lag_ms=0,
            last_translation_latency_ms=0,
            ring_buffer_fill_samples=0,
            ring_buffer_capacity_samples=self.audio_buffer.capacity_samples,
            ring_buffer_fill_ms=0,
            dropped_audio_chunks=0,
            dropped_audio_samples=0,
            dropped_stale_asr_windows=0,
            stale_asr_skips=0,
            dropped_translation_jobs=0,
            received_audio_frames=0,
        )
        self._stop_event.clear()
        self._is_running = True
        self._threads = [
            threading.Thread(target=self._run_asr_loop, name="asr-loop", daemon=True),
            threading.Thread(target=self._run_translation_loop, name="translation-loop", daemon=True),
            threading.Thread(target=self._run_state_loop, name="state-loop", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._is_running = False
        final_segment = self.stabilizer.flush_final()
        if final_segment is not None:
            self.state.upsert_live_segment(final_segment)
            self._maybe_submit_translation(final_segment)
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._threads = []
        self.audio_buffer.clear(reset_counters=True)
        self._drain_queue(self.queues.translation_queue)
        self.state.set_metrics(audio_queue_size=0, translation_queue_size=0, receiver_queue_size=0)
        self._update_buffer_metrics(receiver_queue_capacity=self.config.receiver_queue_size)

    def update_config(self, config: SessionConfig) -> None:
        previous = self.config
        self.config = config
        self.state.update_config(config)
        self.stabilizer.provisional_debounce_ms = config.provisional_debounce_ms
        self.stabilizer.final_silence_ms = config.final_silence_ms
        self.audio_buffer.reconfigure(config.audio_ring_buffer_seconds)
        if self.queues.translation_queue.maxsize != config.translation_queue_size:
            self.queues = PipelineQueues(queue.Queue(maxsize=config.translation_queue_size))
            self.scheduler = PipelineScheduler(self.queues)
        self._update_buffer_metrics(receiver_queue_capacity=config.receiver_queue_size)
        if (
            previous.whisper_model_name != config.whisper_model_name
            or previous.whisper_device != config.whisper_device
            or previous.whisper_compute_type != config.whisper_compute_type
            or previous.llm_base_url != config.llm_base_url
            or previous.llm_model_name != config.llm_model_name
            or previous.llm_api_key != config.llm_api_key
        ):
            self._ensure_workers()

    def submit_audio_frame(self, event: AudioFrameEvent) -> None:
        if not self._is_running:
            return
        self.audio_buffer.append(event.samples, timestamp=event.timestamp)
        self.state.increment_metric("received_audio_frames")
        self._update_buffer_metrics()

    def update_receiver_metrics(self, queue_size: int | None, queue_capacity: int | None = None) -> None:
        self._update_buffer_metrics(receiver_queue_size=queue_size, receiver_queue_capacity=queue_capacity)

    def _submit_translation_job(self, segment: TranscriptSegment) -> None:
        dropped = self.scheduler.submit_translation(TranslationJob(segment=segment))
        if dropped:
            self.state.increment_metric("dropped_translation_jobs", dropped)
        self.state.set_metrics(translation_queue_size=self.queues.translation_queue.qsize())

    def _maybe_submit_translation(self, segment: TranscriptSegment | None) -> None:
        if self._translator and self._translator.should_translate(segment):
            self._submit_translation_job(segment)

    @staticmethod
    def _drain_queue(target_queue: queue.Queue) -> None:
        try:
            while True:
                target_queue.get_nowait()
        except queue.Empty:
            return

    def _record_asr_latency(self, latency_ms: int) -> int:
        with self._metrics_lock:
            self._asr_latency_total_ms += latency_ms
            self._asr_latency_samples += 1
            return int(self._asr_latency_total_ms / self._asr_latency_samples)

    def _update_buffer_metrics(
        self,
        receiver_queue_size: int | None = None,
        receiver_queue_capacity: int | None = None,
    ) -> None:
        snapshot = self.audio_buffer.snapshot()
        updates = {
            "ring_buffer_fill_samples": snapshot.fill_samples,
            "ring_buffer_capacity_samples": snapshot.capacity_samples,
            "ring_buffer_fill_ms": int(snapshot.fill_samples * 1000 / TARGET_SAMPLE_RATE),
            "dropped_audio_chunks": snapshot.dropped_chunks,
            "dropped_audio_samples": snapshot.dropped_samples,
        }
        if receiver_queue_size is not None:
            updates["receiver_queue_size"] = receiver_queue_size
            updates["audio_queue_size"] = receiver_queue_size
        if receiver_queue_capacity is not None:
            updates["receiver_queue_capacity"] = receiver_queue_capacity
        self.state.set_metrics(**updates)

    def _process_latest_audio_window(self) -> None:
        if self._asr_worker is None:
            return
        buffer_snapshot = self.audio_buffer.snapshot()
        if buffer_snapshot.fill_samples <= 0:
            self._update_buffer_metrics()
            return
        window_samples = min(int(self.config.asr_window_ms * TARGET_SAMPLE_RATE / 1000), buffer_snapshot.fill_samples)
        if window_samples <= 0:
            return
        hop_samples = max(int((self.config.asr_window_ms - self.config.asr_overlap_ms) * TARGET_SAMPLE_RATE / 1000), 1)
        current_end = buffer_snapshot.write_index
        if current_end == self._last_processed_window_end:
            self._update_buffer_metrics()
            return
        if self._last_processed_window_end:
            advanced = max(current_end - self._last_processed_window_end, 0)
            skipped = max((advanced // hop_samples) - 1, 0)
            if skipped > 0:
                self.state.increment_metric("dropped_stale_asr_windows", skipped)
                self.state.increment_metric("stale_asr_skips", 1)
        audio, start_index, end_index = self.audio_buffer.read_latest(window_samples)
        if audio.size == 0:
            self._update_buffer_metrics()
            return
        started = time.time()
        result = self._asr_worker.transcribe_chunk(
            audio,
            language=self.config.source_language if self.config.source_language_mode == "fixed" else None,
        )
        finished = time.time()
        lag_ms = 0
        if buffer_snapshot.latest_timestamp is not None:
            lag_ms = max(int((finished - buffer_snapshot.latest_timestamp) * 1000), 0)
        latency_ms = int((finished - started) * 1000)
        average_latency_ms = self._record_asr_latency(latency_ms)
        segment = self.stabilizer.ingest_update(
            result.text,
            result.start_ms,
            result.end_ms,
            result.language,
            result.confidence,
            now=finished,
        )
        self._last_processed_window_end = end_index
        self.state.set_metrics(
            last_asr_latency_ms=latency_ms,
            average_asr_latency_ms=average_latency_ms,
            asr_worker_lag_ms=lag_ms,
        )
        self._update_buffer_metrics()
        self.state.set_debug(
            "last_asr_window",
            {
                "start_index": start_index,
                "end_index": end_index,
                "window_ms": int(audio.shape[0] * 1000 / TARGET_SAMPLE_RATE),
                "configured_poll_interval_ms": self.config.asr_poll_interval_ms,
                "configured_overlap_ms": self.config.asr_overlap_ms,
            },
        )
        self.state.set_debug("last_asr_result", result.raw_segments)
        if segment is not None:
            self.state.upsert_live_segment(segment)
            self._maybe_submit_translation(segment)

    def _run_asr_loop(self) -> None:
        while not self._stop_event.is_set():
            self._process_latest_audio_window()
            self._stop_event.wait(max(self.config.asr_poll_interval_ms / 1000.0, 0.05))

    def _run_translation_loop(self) -> None:
        while not self._stop_event.is_set() or not self.queues.translation_queue.empty():
            try:
                job = self.queues.translation_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._translator is None:
                continue
            history = self.state.snapshot().transcript_history
            translated = self._translator.translate_segment(job.segment, self.config, history)
            if translated is None:
                if self._translator_client is not None:
                    self.state.set_worker_health(
                        translation_ready=False,
                        translator_last_error=self._translator_client.last_error,
                    )
                continue
            latest = self.state.snapshot().live_transcript
            translated = self._translator.discard_stale(translated, latest)
            if translated is None:
                self.state.increment_metric("dropped_translation_jobs")
                continue
            self.state.upsert_translation(translated)
            self.state.set_worker_health(translation_ready=True, translator_last_error=None)
            self.state.set_metrics(translation_queue_size=self.queues.translation_queue.qsize())

    def _run_state_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.1)
            transitioned = self.stabilizer.on_silence()
            if transitioned is not None:
                self.state.upsert_live_segment(transitioned)
                self._maybe_submit_translation(transitioned)
