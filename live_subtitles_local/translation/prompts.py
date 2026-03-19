from __future__ import annotations


def build_translation_messages(
    current_text: str,
    target_language: str,
    recent_context: list[str] | None = None,
) -> list[dict[str, str]]:
    context_lines = "\n".join(f"- {line}" for line in recent_context or []) or "- (none)"
    system = (
        "You translate live subtitle segments. Output only the translated subtitle text in the requested target language. "
        "Be concise, natural, and faithful. No explanations, notes, source text, markup, or repetition."
    )
    user = (
        f"Target language: {target_language}\n"
        f"Recent finalized context:\n{context_lines}\n\n"
        f"Current source segment:\n{current_text}\n\n"
        "Return only the translated subtitle text."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
