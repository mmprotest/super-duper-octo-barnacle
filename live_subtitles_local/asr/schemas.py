from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import time

TranscriptState = Literal["partial", "provisional", "final"]
TranslationState = Literal["draft", "final"]
MicrophoneState = Literal["inactive", "starting", "active", "failed"]


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    source_lang: str | None
    text: str
    state: TranscriptState
    revision: int
    confidence: float | None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class TranslationSegment:
    segment_id: str
    source_revision: int
    target_lang: str
    text: str
    state: TranslationState
    latency_ms: int
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SessionConfig:
    target_language: str = "en"
    source_language_mode: str = "auto"
    source_language: str | None = None
    whisper_model_name: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "local"
    llm_model_name: str = "qwen/qwen3-9b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 96
    show_source: bool = True
    show_translation: bool = True
    max_visible_lines: int = 6
    provisional_debounce_ms: int = 450
    final_silence_ms: int = 1200
