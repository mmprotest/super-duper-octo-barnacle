from __future__ import annotations

from dataclasses import dataclass, field
import queue

from live_subtitles_local.asr.schemas import TranscriptSegment


@dataclass(slots=True)
class TranslationJob:
    segment: TranscriptSegment


@dataclass(slots=True)
class PipelineQueues:
    translation_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))
