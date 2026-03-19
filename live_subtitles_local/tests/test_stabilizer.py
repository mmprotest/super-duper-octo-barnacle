from live_subtitles_local.asr.stabilizer import TranscriptStabilizer


def test_stabilizer_progresses_to_provisional_and_final() -> None:
    stabilizer = TranscriptStabilizer(provisional_debounce_ms=200, final_silence_ms=400)
    segment = stabilizer.ingest_update("hello world", 0, 500, "en", 0.9, now=1.0)
    assert segment is not None
    assert segment.state == "partial"

    provisional = stabilizer.on_silence(now=1.25)
    assert provisional is not None
    assert provisional.state == "provisional"

    final = stabilizer.on_silence(now=1.45)
    assert final is not None
    assert final.state == "final"


def test_stabilizer_increments_revision_only_on_material_change() -> None:
    stabilizer = TranscriptStabilizer(provisional_debounce_ms=200, final_silence_ms=400)
    first = stabilizer.ingest_update("hello world", 0, 500, "en", 0.8, now=1.0)
    assert first is not None
    second = stabilizer.ingest_update("hello brave world", 0, 700, "en", 0.8, now=1.1)
    assert second is not None
    assert second.revision == 1
