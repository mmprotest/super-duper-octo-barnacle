from __future__ import annotations

from dataclasses import dataclass, field
from queue import LifoQueue, Queue

from live_local_subtitles.asr.schemas import ASRChunk, TranscriptSegment


@dataclass(slots=True)
class PipelineQueues:
    audio_chunks: Queue[ASRChunk] = field(default_factory=lambda: Queue(maxsize=8))
    transcript_updates: Queue[TranscriptSegment] = field(default_factory=lambda: Queue(maxsize=32))
    translation_requests: LifoQueue[TranscriptSegment] = field(default_factory=lambda: LifoQueue(maxsize=8))
    translation_updates: Queue = field(default_factory=lambda: Queue(maxsize=32))
    ui_events: Queue = field(default_factory=lambda: Queue(maxsize=64))
