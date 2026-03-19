from __future__ import annotations

from live_subtitles_local.asr.schemas import TranscriptSegment


def format_timestamp(ms: int) -> str:
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def export_srt(final_segments: list[TranscriptSegment]) -> str:
    finals = [segment for segment in final_segments if segment.state == "final"]
    blocks = []
    for index, segment in enumerate(finals, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(segment.start_ms)} --> {format_timestamp(segment.end_ms)}\n{segment.text}\n"
        )
    return "\n".join(blocks)
