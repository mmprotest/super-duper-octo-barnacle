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
    health = snapshot.worker_health
    mic_icon, mic_label = MIC_STATUS[health.microphone_state]

    st.subheader("Session status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Microphone", mic_label)
    col2.metric("ASR worker", "ready" if health.asr_loaded else "unavailable")
    col3.metric("Translator", "ready" if health.translation_ready else "unavailable")
    st.caption(
        f"{mic_icon} {mic_label} · audio frames={snapshot.metrics.received_audio_frames} · "
        f"receiver queue={snapshot.metrics.receiver_queue_size if snapshot.metrics.receiver_queue_size is not None else 'n/a'}/{snapshot.metrics.receiver_queue_capacity} · "
        f"translation queue={snapshot.metrics.translation_queue_size}"
    )
    if health.translator_last_error:
        st.caption(f"Translator health detail: {health.translator_last_error}")
    if snapshot.metrics.dropped_audio_chunks or snapshot.metrics.dropped_stale_asr_windows:
        st.error(
            "Audio ingest overloaded: stale audio work is being dropped to preserve the latest transcript."
        )
    elif snapshot.metrics.receiver_queue_size and snapshot.metrics.receiver_queue_capacity:
        if snapshot.metrics.receiver_queue_size >= int(snapshot.metrics.receiver_queue_capacity * 0.8):
            st.warning("Receiver queue is nearing capacity; ASR is close to falling behind.")

    st.subheader("RTC configuration")
    rtc_col1, rtc_col2, rtc_col3 = st.columns(3)
    rtc_col1.metric("RTC mode", health.rtc_mode)
    rtc_col2.metric("TURN", "configured" if health.rtc_turn_configured else "not configured")
    rtc_col3.metric("RTC failure source", "likely RTC config" if health.microphone_state == "failed" and not health.rtc_turn_configured and health.rtc_mode == "remote_env" else "not implied")
    st.caption(health.rtc_description)
    st.info(health.rtc_connection_guidance)

    if snapshot.config.show_source:
        st.subheader("Current source subtitle")
        if live:
            st.markdown(f"### {STATE_EMOJI.get(live.state, '•')} {live.text}")
            st.caption(f"state={live.state} revision={live.revision} lang={live.source_lang or 'auto'}")
        else:
            if health.microphone_state == "active":
                st.info("Listening for speech. Speak into the browser microphone input.")
            elif health.microphone_state == "starting":
                st.info("Waiting for the browser to grant microphone access and start streaming.")
            elif health.microphone_state == "failed":
                if health.rtc_mode == "remote_env" and not health.rtc_turn_configured:
                    st.error(
                        "The browser microphone stream did not start. Remote mode is running without TURN, so RTC configuration is a likely cause."
                    )
                else:
                    st.error(
                        "The browser microphone stream did not start. Check permissions, HTTPS/localhost, and browser WebRTC support before assuming TURN is required."
                    )
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
            if health.translation_ready:
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
    col1, col2, col3 = st.columns(3)
    col1.metric("ASR latency", snapshot.metrics.last_asr_latency_ms or 0)
    col2.metric("Translation latency", snapshot.metrics.last_translation_latency_ms or 0)
    col3.metric("ASR lag", snapshot.metrics.asr_worker_lag_ms or 0)

    st.subheader("Pipeline metrics")
    pipe1, pipe2, pipe3 = st.columns(3)
    pipe1.metric(
        "Ring buffer fill",
        f"{snapshot.metrics.ring_buffer_fill_ms} ms",
        delta=f"{snapshot.metrics.ring_buffer_fill_samples}/{snapshot.metrics.ring_buffer_capacity_samples} samples",
    )
    pipe2.metric("Avg ASR latency", snapshot.metrics.average_asr_latency_ms or 0)
    pipe3.metric("Dropped stale ASR windows", snapshot.metrics.dropped_stale_asr_windows)
    st.caption(
        f"dropped audio chunks={snapshot.metrics.dropped_audio_chunks} · dropped audio samples={snapshot.metrics.dropped_audio_samples} · "
        f"stale-skip events={snapshot.metrics.stale_asr_skips} · dropped translation jobs={snapshot.metrics.dropped_translation_jobs}"
    )

    if snapshot.last_error_message:
        st.error(snapshot.last_error_message)

    if debug:
        st.subheader("Debug")
        st.json(snapshot.debug)
