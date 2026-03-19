from live_subtitles_local.asr.schemas import TranscriptSegment
from live_subtitles_local.export.srt import export_srt


def test_export_srt_uses_only_final_segments() -> None:
    segments = [
        TranscriptSegment("1", 0, 1000, "en", "Hello", "final", 0, 0.9),
        TranscriptSegment("2", 1000, 1800, "en", "ignored", "partial", 0, 0.9),
    ]
    srt = export_srt(segments)
    assert "Hello" in srt
    assert "ignored" not in srt
    assert "00:00:00,000 --> 00:00:01,000" in srt
