from __future__ import annotations

import streamlit as st

from live_subtitles_local.app.state import AppSnapshot


STATE_EMOJI = {"partial": "🟡", "provisional": "🟠", "final": "🟢"}
MIC_STATUS = {
    "inactive": ("⚪", "Microphone inactive"),
    "starting": ("🟡", "Microphone starting"),
    "active": ("🟢", "Microphone active"),
    "failed": ("🔴", "Microphone failed"),
}


def render_subtitles(snapshot: AppSnapshot, debug: bool = False) -> None:
    live = snapshot.live_transcript
    current_translation = snapshot.current_translation
    mic_icon, mic_label = MIC_STATUS[snapshot.worker_health.microphone_state]

    st.subheader("Session status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Microphone", mic_label)
    col2.metric("ASR worker", "ready" if snapshot.worker_health.asr_loaded else "unavailable")
    col3.metric("Translator", "ready" if snapshot.worker_health.translation_ready else "unavailable")
    st.caption(
        f"{mic_icon} {mic_label} · audio frames={snapshot.metrics.received_audio_frames} · "
        f"audio queue={snapshot.metrics.audio_queue_size} · translation queue={snapshot.metrics.translation_queue_size}"
    )
    if snapshot.worker_health.translator_last_error:
        st.caption(f"Translator health detail: {snapshot.worker_health.translator_last_error}")

    if snapshot.config.show_source:
        st.subheader("Current source subtitle")
        if live:
            st.markdown(f"### {STATE_EMOJI.get(live.state, '•')} {live.text}")
            st.caption(f"state={live.state} revision={live.revision} lang={live.source_lang or 'auto'}")
        else:
            if snapshot.worker_health.microphone_state == "active":
                st.info("Listening for speech. Speak into the browser microphone input.")
            elif snapshot.worker_health.microphone_state == "starting":
                st.info("Waiting for the browser to grant microphone access and start streaming.")
            elif snapshot.worker_health.microphone_state == "failed":
                st.error("The browser microphone stream did not start. Check permissions, HTTPS/localhost, and WebRTC support.")
            else:
                st.info("Microphone streaming is inactive.")

    if snapshot.config.show_translation:
        st.subheader("Current translated subtitle")
        if current_translation:
            st.markdown(f"### {current_translation.text}")
            st.caption(
                f"state={current_translation.state} latency={current_translation.latency_ms} ms target={current_translation.target_lang}"
            )
        else:
            if snapshot.worker_health.translation_ready:
                st.info("Translation appears after Whisper stabilizes a segment to provisional or final.")
            else:
                st.warning("Translation is unavailable because the local OpenAI-compatible endpoint is not healthy.")

    st.subheader("Recent history")
    history = snapshot.transcript_history[-snapshot.config.max_visible_lines :]
    if not history:
        st.caption("Finalized segments will accumulate here after speech ends or silence is detected.")
    for segment in reversed(history):
        marker = STATE_EMOJI.get(segment.state, "•")
        st.markdown(f"- {marker} **{segment.text}**")

    st.subheader("Latency")
    col1, col2 = st.columns(2)
    col1.metric("ASR latency", snapshot.metrics.last_asr_latency_ms or 0)
    col2.metric("Translation latency", snapshot.metrics.last_translation_latency_ms or 0)

    if snapshot.last_error_message:
        st.error(snapshot.last_error_message)

    if debug:
        st.subheader("Debug")
        st.json(snapshot.debug)
