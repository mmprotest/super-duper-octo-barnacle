from __future__ import annotations

from live_local_subtitles.asr.schemas import ASRResult, SegmentState
from live_local_subtitles.asr.stabilizer import StabilizerConfig, TranscriptStabilizer



def test_stabilizer_promotes_segment_to_provisional_after_repeated_similar_updates() -> None:
    stabilizer = TranscriptStabilizer(StabilizerConfig(stability_similarity=0.8, provisional_repeats=1))

    first = stabilizer.update(
        ASRResult(
            chunk_id="c1",
            segment_id="seg-1",
            start_ms=0,
            end_ms=1000,
            text="hello world",
            source_lang="en",
            confidence=0.9,
            no_speech_probability=0.0,
        )
    )
    second = stabilizer.update(
        ASRResult(
            chunk_id="c2",
            segment_id="seg-1",
            start_ms=0,
            end_ms=1200,
            text="hello world!",
            source_lang="en",
            confidence=0.95,
            no_speech_probability=0.0,
        )
    )

    assert first.state == SegmentState.PARTIAL
    assert second.state == SegmentState.PROVISIONAL
    assert second.revision == 2



def test_stabilizer_marks_segment_final_after_pause_threshold() -> None:
    stabilizer = TranscriptStabilizer(StabilizerConfig(final_silence_ms=500, provisional_repeats=0))
    stabilizer.update(
        ASRResult(
            chunk_id="c1",
            segment_id="seg-2",
            start_ms=0,
            end_ms=1000,
            text="stable line",
            source_lang="en",
            confidence=0.9,
            no_speech_probability=0.0,
        )
    )

    finalized = stabilizer.mark_final_due_to_pause(latest_end_ms=1600)

    assert len(finalized) == 1
    assert finalized[0].state == SegmentState.FINAL
