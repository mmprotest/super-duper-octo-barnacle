from live_subtitles_local.asr.schemas import SessionConfig, TranscriptSegment, TranslationSegment
from live_subtitles_local.translation.translator import SegmentTranslator


class DummyClient:
    def translate_chat(self, messages, temperature, max_tokens):
        return "hola"


def test_discard_stale_translation_revision() -> None:
    translator = SegmentTranslator(client=DummyClient())
    latest = TranscriptSegment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1000,
        source_lang="en",
        text="hello there",
        state="provisional",
        revision=2,
        confidence=0.9,
    )
    candidate = TranslationSegment(
        segment_id="seg-1",
        source_revision=1,
        target_lang="es",
        text="hola",
        state="draft",
        latency_ms=50,
    )
    assert translator.discard_stale(candidate, latest) is None


def test_translate_provisional_segment() -> None:
    translator = SegmentTranslator(client=DummyClient())
    config = SessionConfig(target_language="es")
    segment = TranscriptSegment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1000,
        source_lang="en",
        text="hello",
        state="provisional",
        revision=0,
        confidence=0.9,
    )
    translated = translator.translate_segment(segment, config, recent_finalized_source=[])
    assert translated is not None
    assert translated.text == "hola"
