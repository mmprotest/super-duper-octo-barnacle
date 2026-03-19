from __future__ import annotations

from live_local_subtitles.asr.schemas import ExportBundle, SegmentState



def _format_timestamp(ms: int) -> str:
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"



def build_srt(bundle: ExportBundle) -> str:
    entries: list[str] = []
    index = 1
    translations = {item.segment_id: item for item in bundle.translations}
    for segment in bundle.transcripts:
        if segment.state != SegmentState.FINAL or not segment.text.strip():
            continue
        lines = [segment.text.strip()]
        translation = translations.get(segment.segment_id)
        if translation and translation.text.strip():
            lines.append(translation.text.strip())
        entries.append(
            f"{index}\n{_format_timestamp(segment.start_ms)} --> {_format_timestamp(segment.end_ms)}\n" + "\n".join(lines)
        )
        index += 1
    return "\n\n".join(entries) + ("\n" if entries else "")
