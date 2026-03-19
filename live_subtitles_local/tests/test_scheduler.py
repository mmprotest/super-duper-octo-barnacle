import queue

from live_subtitles_local.asr.schemas import TranscriptSegment
from live_subtitles_local.pipeline.queues import PipelineQueues, TranslationJob
from live_subtitles_local.pipeline.scheduler import PipelineScheduler


def _segment(segment_id: str, revision: int) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id,
        start_ms=0,
        end_ms=500,
        source_lang="en",
        text=f"text-{revision}",
        state="provisional",
        revision=revision,
        confidence=0.9,
    )


def test_scheduler_discards_stale_translation_revisions() -> None:
    queues = PipelineQueues(translation_queue=queue.Queue(maxsize=8))
    scheduler = PipelineScheduler(queues)

    scheduler.submit_translation(TranslationJob(segment=_segment("seg-1", 0)))
    dropped = scheduler.submit_translation(TranslationJob(segment=_segment("seg-1", 2)))

    assert dropped == 1
    queued = queues.translation_queue.get_nowait()
    assert queued.segment.revision == 2
