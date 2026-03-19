from __future__ import annotations

from collections.abc import Sequence

from live_local_subtitles.asr.schemas import TranscriptSegment


SYSTEM_PROMPT = (
    "You translate spoken subtitle segments. Output only the translated subtitle text. "
    "Keep it concise, natural, and readable. Do not explain. Do not add notes. "
    "Do not repeat the source text."
)



def build_translation_messages(
    segment: TranscriptSegment,
    target_lang: str,
    recent_context: Sequence[TranscriptSegment],
) -> list[dict[str, str]]:
    context_lines = [ctx.text.strip() for ctx in recent_context if ctx.segment_id != segment.segment_id and ctx.text.strip()]
    context_block = "\n".join(f"- {line}" for line in context_lines[-3:])
    user_content = (
        f"Translate the CURRENT subtitle segment into {target_lang}.\n"
        f"Source language: {segment.source_lang}.\n"
        f"Current segment:\n{segment.text.strip()}\n"
    )
    if context_block:
        user_content += f"Recent finalized context:\n{context_block}\n"
    user_content += "Return translation only."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
