from __future__ import annotations

from html import escape

from live_local_subtitles.asr.schemas import SegmentState, TranscriptSegment, TranslationSegment, TranslationState



def render_live_subtitles(
    transcript: TranscriptSegment | None,
    translation: TranslationSegment | None,
    show_source: bool,
    show_translation: bool,
) -> str:
    blocks: list[str] = []
    if show_source and transcript is not None:
        css_class = "subtitle-final" if transcript.state == SegmentState.FINAL else "subtitle-partial"
        blocks.append(f'<div class="{css_class}">{escape(transcript.text)}</div>')
    if show_translation and translation is not None and translation.text:
        css_class = "translation-final" if translation.state == TranslationState.FINAL else "translation-draft"
        blocks.append(f'<div class="{css_class}">{escape(translation.text)}</div>')
    if not blocks:
        blocks.append('<div class="subtitle-empty">Waiting for speech…</div>')
    return "\n".join(blocks)
