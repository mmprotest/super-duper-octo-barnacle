from __future__ import annotations

import json

from live_local_subtitles.asr.schemas import ExportBundle



def build_transcript_json(bundle: ExportBundle) -> str:
    payload = {
        "transcripts": [
            {
                "segment_id": segment.segment_id,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "source_lang": segment.source_lang,
                "text": segment.text,
                "state": segment.state.value,
                "revision": segment.revision,
                "confidence": segment.confidence,
                "created_at": segment.created_at.isoformat(),
                "updated_at": segment.updated_at.isoformat(),
            }
            for segment in bundle.transcripts
        ],
        "translations": [
            {
                "segment_id": segment.segment_id,
                "source_revision": segment.source_revision,
                "target_lang": segment.target_lang,
                "text": segment.text,
                "state": segment.state.value,
                "latency_ms": segment.latency_ms,
                "created_at": segment.created_at.isoformat(),
                "updated_at": segment.updated_at.isoformat(),
                "error": segment.error,
            }
            for segment in bundle.translations
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
