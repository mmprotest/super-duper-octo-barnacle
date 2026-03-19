from __future__ import annotations

from live_local_subtitles.asr.schemas import ExportBundle, SegmentState



def build_plain_text(bundle: ExportBundle) -> str:
    translations = {item.segment_id: item for item in bundle.translations}
    lines: list[str] = []
    for segment in bundle.transcripts:
        if segment.state != SegmentState.FINAL:
            continue
        lines.append(segment.text.strip())
        translation = translations.get(segment.segment_id)
        if translation and translation.text.strip():
            lines.append(f"[{translation.target_lang}] {translation.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")
