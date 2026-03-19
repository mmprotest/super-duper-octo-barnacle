from __future__ import annotations

import json
from dataclasses import asdict

from live_subtitles_local.asr.schemas import TranscriptSegment, TranslationSegment


def export_transcript_txt(segments: list[TranscriptSegment]) -> str:
    return "\n".join(segment.text for segment in segments if segment.state == "final")


def export_transcript_json(
    source_segments: list[TranscriptSegment],
    translations: list[TranslationSegment],
) -> str:
    payload = {
        "source_segments": [asdict(segment) for segment in source_segments],
        "translations": [asdict(translation) for translation in translations],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
