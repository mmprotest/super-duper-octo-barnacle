from __future__ import annotations

import json
from dataclasses import asdict

from live_subtitles_local.asr.schemas import TranscriptSegment, TranslationSegment


def export_jsonl(source_segments: list[TranscriptSegment], translations: list[TranslationSegment]) -> str:
    translations_by_id = {item.segment_id: item for item in translations}
    lines = []
    for segment in source_segments:
        payload = {"source": asdict(segment)}
        translation = translations_by_id.get(segment.segment_id)
        if translation is not None:
            payload["translation"] = asdict(translation)
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)
