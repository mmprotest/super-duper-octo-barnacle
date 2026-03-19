from __future__ import annotations

from dataclasses import replace
import time

from live_subtitles_local.asr.schemas import SessionConfig, TranscriptSegment, TranslationSegment
from live_subtitles_local.translation.cache import TranslationCache
from live_subtitles_local.translation.llm_client import OpenAICompatibleClient
from live_subtitles_local.translation.prompts import build_translation_messages


class SegmentTranslator:
    def __init__(self, client: OpenAICompatibleClient, cache: TranslationCache | None = None) -> None:
        self.client = client
        self.cache = cache or TranslationCache()

    def should_translate(self, segment: TranscriptSegment | None) -> bool:
        return bool(segment and segment.text and segment.state in {"provisional", "final"})

    def translate_segment(
        self,
        segment: TranscriptSegment,
        config: SessionConfig,
        recent_finalized_source: list[TranscriptSegment],
    ) -> TranslationSegment | None:
        if not self.should_translate(segment):
            return None
        context_lines = [entry.text for entry in recent_finalized_source[-3:]]
        cache_key = (segment.text, config.target_language, tuple(context_lines))
        started = time.time()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return TranslationSegment(
                segment_id=segment.segment_id,
                source_revision=segment.revision,
                target_lang=config.target_language,
                text=cached,
                state="final" if segment.state == "final" else "draft",
                latency_ms=0,
                created_at=started,
                updated_at=started,
            )

        messages = build_translation_messages(segment.text, config.target_language, context_lines)
        translated = self.client.translate_chat(
            messages=messages,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        if not translated:
            return None
        finished = time.time()
        self.cache.set(cache_key, translated)
        return TranslationSegment(
            segment_id=segment.segment_id,
            source_revision=segment.revision,
            target_lang=config.target_language,
            text=translated,
            state="final" if segment.state == "final" else "draft",
            latency_ms=int((finished - started) * 1000),
            created_at=started,
            updated_at=finished,
        )

    @staticmethod
    def discard_stale(
        candidate: TranslationSegment,
        latest_segment: TranscriptSegment | None,
    ) -> TranslationSegment | None:
        if latest_segment is None:
            return candidate
        if candidate.segment_id != latest_segment.segment_id:
            return None
        if candidate.source_revision != latest_segment.revision:
            return None
        return replace(candidate)
