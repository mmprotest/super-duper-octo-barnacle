from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Full
from typing import Any

import yaml

from live_local_subtitles.app.state import AppState
from live_local_subtitles.asr.schemas import ASRChunk, ASRResult, ExportBundle, SegmentState, TranscriptSegment, TranslationSegment
from live_local_subtitles.asr.stabilizer import StabilizerConfig, TranscriptStabilizer
from live_local_subtitles.asr.whisper_worker import FasterWhisperWorker, WhisperConfig
from live_local_subtitles.audio.capture import AudioCaptureConfig, MicrophoneAudioCapture
from live_local_subtitles.export.srt import build_srt
from live_local_subtitles.export.text_export import build_plain_text
from live_local_subtitles.export.transcript_json import build_transcript_json
from live_local_subtitles.translation.llm_client import LocalOpenAICompatibleClient, OpenAICompatibleConfig
from live_local_subtitles.translation.translator import TranslationWorker, TranslatorConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeConfig:
    audio: AudioCaptureConfig
    whisper: WhisperConfig
    stabilizer: StabilizerConfig
    llm: OpenAICompatibleConfig
    translator: TranslatorConfig
    target_lang: str = "English"

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(
            audio=AudioCaptureConfig(**(data.get("audio") or {})),
            whisper=WhisperConfig(**(data.get("whisper") or {})),
            stabilizer=StabilizerConfig(**(data.get("stabilizer") or {})),
            llm=OpenAICompatibleConfig(**(data.get("llm") or {})),
            translator=TranslatorConfig(**(data.get("translator") or {})),
            target_lang=(data.get("ui") or {}).get("target_lang", "English"),
        )


class LocalSubtitlesOrchestrator:
    """Wires capture, ASR, stabilization, translation, and UI-facing state together."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.state = AppState(target_lang=config.target_lang)
        self.stabilizer = TranscriptStabilizer(config.stabilizer)
        self.asr_worker = FasterWhisperWorker(config.whisper, on_result=self.handle_asr_result)
        self.translator = TranslationWorker(
            client=LocalOpenAICompatibleClient(config.llm),
            config=config.translator,
            context_provider=self.recent_final_segments,
            on_result=self.handle_translation_result,
        )
        self.audio_capture = MicrophoneAudioCapture(config.audio, on_chunk=self.handle_audio_chunk)

    def start(self) -> None:
        self.state.running = True
        self.translator.set_target_lang(self.state.target_lang)
        self.asr_worker.start()
        self.translator.start()
        self.audio_capture.start()

    def stop(self) -> None:
        self.audio_capture.stop()
        self.asr_worker.stop()
        self.translator.stop()
        self.state.running = False

    def set_target_lang(self, target_lang: str) -> None:
        self.state.target_lang = target_lang
        self.translator.set_target_lang(target_lang)

    def handle_audio_chunk(self, chunk: ASRChunk) -> None:
        self.state.metrics.last_audio_chunk_ms = chunk.end_ms
        try:
            self.asr_worker.submit(chunk)
        except Full:
            self.state.metrics.last_error = "ASR backlog full; dropping oldest audio window"

    def handle_asr_result(self, result: ASRResult) -> None:
        segment = self.stabilizer.update(result)
        self.state.upsert_transcript(segment)
        if segment.state in {SegmentState.PROVISIONAL, SegmentState.FINAL}:
            self.translator.submit(segment)
        for finalized in self.stabilizer.mark_final_due_to_pause(result.end_ms):
            self.state.upsert_transcript(finalized)
            self.translator.submit(finalized)

    def handle_translation_result(self, result: TranslationSegment) -> None:
        self.state.metrics.translation_latency_ms = result.latency_ms
        self.state.upsert_translation(result)

    def recent_final_segments(self) -> Sequence[TranscriptSegment]:
        return [seg for seg in self.state.ordered_transcripts() if seg.state == SegmentState.FINAL]

    def latest_segments(self) -> tuple[TranscriptSegment | None, TranslationSegment | None]:
        transcripts = self.state.ordered_transcripts()
        if not transcripts:
            return None, None
        transcript = transcripts[-1]
        translation = self.state.translations.get(transcript.segment_id)
        return transcript, translation

    def export_bundle(self) -> ExportBundle:
        return ExportBundle(
            transcripts=self.state.ordered_transcripts(),
            translations=self.state.ordered_translations(),
        )

    def export_json(self) -> str:
        return build_transcript_json(self.export_bundle())

    def export_text(self) -> str:
        return build_plain_text(self.export_bundle())

    def export_srt(self) -> str:
        return build_srt(self.export_bundle())

    def health_snapshot(self) -> dict[str, Any]:
        llm_ok, llm_message = self.translator.client.healthcheck()
        return {
            "running": self.state.running,
            "target_lang": self.state.target_lang,
            "llm_ok": llm_ok,
            "llm_message": llm_message,
            "metrics": self.state.metrics,
            "segments": len(self.state.transcripts),
        }
