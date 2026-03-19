from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from live_subtitles_local.app.state import ThreadSafeAppState
from live_subtitles_local.asr.schemas import SessionConfig
from live_subtitles_local.asr.stabilizer import TranscriptStabilizer
from live_subtitles_local.asr.whisper_worker import WhisperWorker
from live_subtitles_local.config.loader import load_yaml_file
from live_subtitles_local.pipeline.queues import PipelineQueues, TranslationJob
from live_subtitles_local.pipeline.scheduler import PipelineScheduler
from live_subtitles_local.rtc.audio_handlers import AudioChunkAccumulator, SilenceDetector
from live_subtitles_local.rtc.stream_events import AudioFrameEvent
from live_subtitles_local.translation.llm_client import OpenAICompatibleClient
from live_subtitles_local.translation.translator import SegmentTranslator

logger = logging.getLogger(__name__)


class LiveSubtitleOrchestrator:
    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._raw_settings = self._load_raw_config(config_path)
        self.config = self.load_config(config_path)
        self.state = ThreadSafeAppState(self.config)
        self.queues = PipelineQueues()
        self.scheduler = PipelineScheduler(self.queues)
        self.accumulator = AudioChunkAccumulator(target_chunk_ms=self._raw_settings.get("audio_chunk_ms", 800))
        self.silence_detector = SilenceDetector(threshold=self._raw_settings.get("silence_threshold", 0.008))
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
        self._ensure_workers()

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
        flushed = self.accumulator.flush()
        if flushed.size:
            self._process_audio_chunk(flushed)
        final_segment = self.stabilizer.flush_final()
        if final_segment is not None:
            self.state.upsert_live_segment(final_segment)
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._threads = []
        self._drain_queue(self.queues.audio_queue)
        self._drain_queue(self.queues.translation_queue)
        self.state.set_metrics(audio_queue_size=0, translation_queue_size=0)

    def update_config(self, config: SessionConfig) -> None:
        previous = self.config
        self.config = config
        self.state.update_config(config)
        self.stabilizer.provisional_debounce_ms = config.provisional_debounce_ms
        self.stabilizer.final_silence_ms = config.final_silence_ms
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
        audio = event.samples.astype(np.float32)
        try:
            self.queues.audio_queue.put_nowait(audio)
            snapshot = self.state.snapshot()
            self.state.set_metrics(
                audio_queue_size=self.queues.audio_queue.qsize(),
                received_audio_frames=snapshot.metrics.received_audio_frames + 1,
            )
        except queue.Full:
            self.state.set_error("Audio queue is full; lower chunk size or increase processing throughput.")
        if self.silence_detector.boundary_hint(audio):
            transitioned = self.stabilizer.on_silence()
            if transitioned is not None:
                self.state.upsert_live_segment(transitioned)
                if self._translator and self._translator.should_translate(transitioned):
                    self._submit_translation_job(transitioned)


    def _submit_translation_job(self, segment) -> None:
        self.scheduler.submit_translation(TranslationJob(segment=segment))
        self.state.set_metrics(translation_queue_size=self.queues.translation_queue.qsize())

    @staticmethod
    def _drain_queue(target_queue: queue.Queue) -> None:
        try:
            while True:
                target_queue.get_nowait()
        except queue.Empty:
            return

    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size == 0 or self._asr_worker is None:
            return
        started = time.time()
        result = self._asr_worker.transcribe_chunk(
            chunk,
            language=self.config.source_language if self.config.source_language_mode == "fixed" else None,
        )
        segment = self.stabilizer.ingest_update(
            result.text,
            result.start_ms,
            result.end_ms,
            result.language,
            result.confidence,
        )
        latency_ms = int((time.time() - started) * 1000)
        self.state.set_metrics(last_asr_latency_ms=latency_ms, audio_queue_size=self.queues.audio_queue.qsize())
        self.state.set_debug("last_asr_result", result.raw_segments)
        if segment is not None:
            self.state.upsert_live_segment(segment)
            if self._translator and self._translator.should_translate(segment):
                self._submit_translation_job(segment)

    def _run_asr_loop(self) -> None:
        while not self._stop_event.is_set() or not self.queues.audio_queue.empty():
            try:
                audio = self.queues.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            for chunk in self.accumulator.add_frame(audio):
                self._process_audio_chunk(chunk)

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
                if self._translator and self._translator.should_translate(transitioned):
                    self._submit_translation_job(transitioned)
