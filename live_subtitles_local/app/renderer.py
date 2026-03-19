from __future__ import annotations

import streamlit as st

from live_subtitles_local.app.state import AppSnapshot


STATE_EMOJI = {"partial": "🟡", "provisional": "🟠", "final": "🟢"}


def render_subtitles(snapshot: AppSnapshot, debug: bool = False) -> None:
    live = snapshot.live_transcript
    current_translation = snapshot.current_translation

    if snapshot.config.show_source:
        st.subheader("Current source subtitle")
        if live:
            st.markdown(f"### {STATE_EMOJI.get(live.state, '•')} {live.text}")
            st.caption(f"state={live.state} revision={live.revision} lang={live.source_lang or 'auto'}")
        else:
            st.info("Waiting for microphone audio...")

    if snapshot.config.show_translation:
        st.subheader("Current translated subtitle")
        if current_translation:
            st.markdown(f"### {current_translation.text}")
            st.caption(
                f"state={current_translation.state} latency={current_translation.latency_ms} ms target={current_translation.target_lang}"
            )
        else:
            st.info("Translation will appear after a segment becomes provisional.")

    st.subheader("Recent history")
    history = snapshot.transcript_history[-snapshot.config.max_visible_lines :]
    if not history:
        st.caption("Finalized segments will accumulate here.")
    for segment in reversed(history):
        marker = STATE_EMOJI.get(segment.state, "•")
        st.markdown(f"- {marker} **{segment.text}**")

    st.subheader("Latency and health")
    col1, col2, col3 = st.columns(3)
    col1.metric("ASR latency", snapshot.metrics.last_asr_latency_ms or 0)
    col2.metric("Translation latency", snapshot.metrics.last_translation_latency_ms or 0)
    col3.metric("Audio queue", snapshot.metrics.audio_queue_size)
    st.caption(
        f"RTC running={snapshot.worker_health.rtc_running}, ASR loaded={snapshot.worker_health.asr_loaded}, "
        f"translation ready={snapshot.worker_health.translation_ready}, translation queue={snapshot.metrics.translation_queue_size}"
    )

    if snapshot.last_error_message:
        st.error(snapshot.last_error_message)

    if debug:
        st.subheader("Debug")
        st.json(snapshot.debug)
