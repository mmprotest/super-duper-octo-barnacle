from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - import guard for environments without CUDA libs.
    WhisperModel = None  # type: ignore[assignment]


@dataclass(slots=True)
class ASRChunkResult:
    text: str
    start_ms: int
    end_ms: int
    language: str | None
    confidence: float | None
    raw_segments: list[dict[str, Any]]


class WhisperWorker:
    """Thin wrapper around faster-whisper with local GPU defaults."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not available. Install dependencies and ensure local CUDA-compatible libraries exist."
            )
        if device != "cuda":
            logger.warning("WhisperWorker is configured for %s; local GPU (cuda) is recommended for latency.", device)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe_chunk(self, audio_np: np.ndarray, language: str | None = None) -> ASRChunkResult:
        if audio_np.size == 0:
            return ASRChunkResult(text="", start_ms=0, end_ms=0, language=language, confidence=None, raw_segments=[])

        audio = audio_np.astype(np.float32)
        # NOTE: For true low-latency streaming, chunk size/VAD overlap must be tuned with FastRTC frame cadence.
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            word_timestamps=False,
            temperature=0.0,
        )
        segment_dicts: list[dict[str, Any]] = []
        texts: list[str] = []
        confidences: list[float] = []
        first_start = 0.0
        last_end = 0.0
        for index, segment in enumerate(segments):
            if index == 0:
                first_start = float(segment.start)
            last_end = float(segment.end)
            texts.append(segment.text.strip())
            avg = getattr(segment, "avg_logprob", None)
            if avg is not None:
                confidences.append(max(0.0, min(1.0, 1.0 + float(avg) / 5.0)))
            segment_dicts.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text,
                    "avg_logprob": avg,
                }
            )
        confidence = sum(confidences) / len(confidences) if confidences else None
        return ASRChunkResult(
            text=" ".join(filter(None, texts)).strip(),
            start_ms=int(first_start * 1000),
            end_ms=int(last_end * 1000),
            language=getattr(info, "language", language),
            confidence=confidence,
            raw_segments=segment_dicts,
        )
