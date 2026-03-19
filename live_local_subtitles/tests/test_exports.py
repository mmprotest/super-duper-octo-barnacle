from __future__ import annotations

from live_local_subtitles.asr.schemas import ExportBundle, SegmentState, TranscriptSegment, TranslationSegment, TranslationState
from live_local_subtitles.export.srt import build_srt



def test_srt_export_includes_translation_line() -> None:
    transcript = TranscriptSegment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1500,
        source_lang="en",
        text="Hello world",
        state=SegmentState.FINAL,
        revision=2,
        confidence=0.95,
    )
    translation = TranslationSegment(
        segment_id="seg-1",
        source_revision=2,
        target_lang="Spanish",
        text="Hola mundo",
        state=TranslationState.FINAL,
        latency_ms=120,
    )

    output = build_srt(ExportBundle(transcripts=[transcript], translations=[translation]))

    assert "00:00:00,000 --> 00:00:01,500" in output
    assert "Hello world" in output
    assert "Hola mundo" in output
