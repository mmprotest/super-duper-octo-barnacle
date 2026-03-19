from __future__ import annotations

import queue

from live_subtitles_local.pipeline.queues import PipelineQueues, TranslationJob


class PipelineScheduler:
    """Drops stale translation jobs and favors newer work while keeping ASR responsive."""

    def __init__(self, queues: PipelineQueues) -> None:
        self.queues = queues

    def submit_translation(self, job: TranslationJob) -> int:
        drained: list[TranslationJob] = []
        try:
            while True:
                drained.append(self.queues.translation_queue.get_nowait())
        except queue.Empty:
            pass
        kept: list[TranslationJob] = []
        dropped = 0
        for existing in drained:
            same_segment = existing.segment.segment_id == job.segment.segment_id
            stale_revision = existing.segment.revision <= job.segment.revision
            if same_segment and stale_revision:
                dropped += 1
                continue
            kept.append(existing)
        for item in kept:
            self.queues.translation_queue.put_nowait(item)
        self.queues.translation_queue.put_nowait(job)
        return dropped
