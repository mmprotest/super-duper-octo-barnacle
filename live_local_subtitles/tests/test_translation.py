from __future__ import annotations

import time

from live_local_subtitles.asr.schemas import SegmentState, TranscriptSegment, TranslationState
from live_local_subtitles.translation.llm_client import OpenAICompatibleConfig
from live_local_subtitles.translation.prompts import build_translation_messages
from live_local_subtitles.translation.translator import TranslationWorker, TranslatorConfig


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def translate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        current = messages[-1]["content"].split("Current segment:\n", 1)[1].split("\n", 1)[0]
        return f"translated::{current}"

    def healthcheck(self) -> tuple[bool, str]:
        return True, "ok"



def test_translation_prompt_is_concise_and_current_segment_only() -> None:
    segment = TranscriptSegment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1000,
        source_lang="en",
        text="Hello there",
        state=SegmentState.PROVISIONAL,
        revision=2,
        confidence=0.9,
    )
    context = [segment.copy_with(segment_id="seg-0", text="Previous sentence", state=SegmentState.FINAL)]

    messages = build_translation_messages(segment, "Spanish", context)

    assert messages[0]["role"] == "system"
    assert "Return translation only" in messages[1]["content"]
    assert "Previous sentence" in messages[1]["content"]
    assert "Hello there" in messages[1]["content"]



def test_translation_worker_drops_stale_revision_and_keeps_latest() -> None:
    client = FakeClient()
    results = []
    worker = TranslationWorker(
        client=client,
        config=TranslatorConfig(debounce_ms=10),
        context_provider=lambda: [],
        on_result=results.append,
    )
    worker.set_target_lang("Spanish")
    worker.start()
    old_segment = TranscriptSegment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1000,
        source_lang="en",
        text="Hello",
        state=SegmentState.PROVISIONAL,
        revision=1,
        confidence=0.9,
    )
    new_segment = old_segment.copy_with(text="Hello world", revision=2)

    worker.submit(old_segment)
    worker.submit(new_segment)
    time.sleep(0.2)
    worker.stop()

    assert results
    assert results[-1].source_revision == 2
    assert results[-1].state in {TranslationState.DRAFT, TranslationState.FINAL}
    assert client.calls[-1][-1]["content"].find("Hello world") != -1
