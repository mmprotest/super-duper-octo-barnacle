from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from live_subtitles_local.app.renderer import render_subtitles
from live_subtitles_local.asr.schemas import SessionConfig
from live_subtitles_local.pipeline.orchestrator import LiveSubtitleOrchestrator
from live_subtitles_local.rtc.audio_handlers import normalize_audio_frame
from live_subtitles_local.rtc.rtc_config import resolve_rtc_settings
from live_subtitles_local.rtc.stream_events import AudioFrameEvent

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
WEBRTC_KEY = "microphone-stream"
STREAM_REQUEST_TS = "stream_request_ts"
STREAM_DESIRED = "stream_desired"


def get_orchestrator() -> LiveSubtitleOrchestrator:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = LiveSubtitleOrchestrator(CONFIG_PATH)
    return st.session_state.orchestrator


def _ensure_stream_state() -> None:
    st.session_state.setdefault(STREAM_DESIRED, False)
    st.session_state.setdefault(STREAM_REQUEST_TS, None)


def _sidebar_config(defaults: SessionConfig) -> tuple[SessionConfig, bool, bool, bool]:
    with st.sidebar:
        st.header("Controls")
        start_clicked = st.button("Start", use_container_width=True)
        stop_clicked = st.button("Stop", use_container_width=True)
        debug = st.toggle("Debug", value=False)
        config = SessionConfig(
            target_language=st.text_input("Target language", value=defaults.target_language),
            source_language_mode=st.selectbox(
                "Source language mode",
                ["auto", "fixed"],
                index=0 if defaults.source_language_mode == "auto" else 1,
            ),
            source_language=st.text_input("Source language", value=defaults.source_language or "") or None,
            whisper_model_name=st.text_input("Whisper model", value=defaults.whisper_model_name),
            whisper_device=defaults.whisper_device,
            whisper_compute_type=st.selectbox(
                "Compute type",
                ["float16", "int8_float16", "int8"],
                index=["float16", "int8_float16", "int8"].index(defaults.whisper_compute_type),
            ),
            llm_base_url=st.text_input("LLM base URL", value=defaults.llm_base_url),
            llm_api_key=defaults.llm_api_key,
            llm_model_name=st.text_input("LLM model", value=defaults.llm_model_name),
            llm_temperature=defaults.llm_temperature,
            llm_max_tokens=defaults.llm_max_tokens,
            show_source=st.toggle("Show source", value=defaults.show_source),
            show_translation=st.toggle("Show translation", value=defaults.show_translation),
            max_visible_lines=defaults.max_visible_lines,
            provisional_debounce_ms=defaults.provisional_debounce_ms,
            final_silence_ms=defaults.final_silence_ms,
        )
    return config, start_clicked, stop_clicked, debug


def _sync_rtc_status(orchestrator: LiveSubtitleOrchestrator, rtc_settings) -> None:
    orchestrator.state.set_worker_health(
        rtc_mode=rtc_settings.mode,
        rtc_turn_configured=rtc_settings.turn_configured,
        rtc_description=rtc_settings.description,
        rtc_connection_guidance=rtc_settings.connection_guidance,
    )
    orchestrator.state.set_debug(
        "rtc_configuration",
        {
            "mode": rtc_settings.mode,
            "turn_configured": rtc_settings.turn_configured,
            "ice_server_count": rtc_settings.ice_server_count,
            "sources": rtc_settings.sources,
            "warnings": rtc_settings.warnings,
            "frontend_rtc_configuration": rtc_settings.frontend_rtc_configuration,
            "server_rtc_configuration": rtc_settings.server_rtc_configuration,
        },
    )


def _set_microphone_state(orchestrator: LiveSubtitleOrchestrator, desired: bool, playing: bool, signalling: bool) -> None:
    if playing:
        state = "active"
    elif desired:
        request_ts = st.session_state.get(STREAM_REQUEST_TS)
        grace_elapsed = bool(request_ts and (time.time() - request_ts) > 8)
        state = "failed" if grace_elapsed and not signalling else "starting"
    else:
        state = "inactive"
    orchestrator.state.set_worker_health(microphone_state=state)


def _drain_audio_frames(orchestrator: LiveSubtitleOrchestrator, ctx) -> int:
    if ctx.audio_receiver is None:
        return 0
    try:
        frames = ctx.audio_receiver.get_frames(timeout=0.05)
    except Exception:
        return 0

    frame_count = 0
    for frame in frames:
        samples = frame.to_ndarray()
        if samples.ndim == 2:
            samples = samples.T
        if not np.issubdtype(samples.dtype, np.integer):
            samples = np.clip(samples, -1.0, 1.0)
        audio_event = AudioFrameEvent(
            samples=samples,
            sample_rate=frame.sample_rate,
            channels=1 if samples.ndim == 1 else samples.shape[1],
        )
        normalized = normalize_audio_frame(audio_event)
        orchestrator.submit_audio_frame(
            AudioFrameEvent(samples=normalized, sample_rate=16000, channels=1, timestamp=audio_event.timestamp)
        )
        frame_count += 1
    return frame_count


def render_app() -> None:
    st.set_page_config(page_title="Local Live Subtitles", layout="wide")
    st.title("Local live transcription + translation subtitles")
    st.caption(
        "This app is Streamlit-native: microphone transport runs through streamlit-webrtc with explicit ICE config. "
        "Hugging Face TURN is disabled; local mode uses direct/local WebRTC only."
    )

    _ensure_stream_state()
    rtc_settings = resolve_rtc_settings()
    orchestrator = get_orchestrator()
    _sync_rtc_status(orchestrator, rtc_settings)
    defaults = orchestrator.state.snapshot().config
    config, start_clicked, stop_clicked, debug = _sidebar_config(defaults)
    orchestrator.update_config(config)

    if rtc_settings.warnings:
        for warning in rtc_settings.warnings:
            st.warning(warning)

    if start_clicked:
        st.session_state[STREAM_DESIRED] = True
        st.session_state[STREAM_REQUEST_TS] = time.time()
        orchestrator.start()
    if stop_clicked:
        st.session_state[STREAM_DESIRED] = False
        st.session_state[STREAM_REQUEST_TS] = None
        orchestrator.stop()

    desired_streaming = st.session_state[STREAM_DESIRED]

    ctx = webrtc_streamer(
        key=WEBRTC_KEY,
        mode=WebRtcMode.SENDONLY,
        desired_playing_state=desired_streaming,
        media_stream_constraints={"audio": True, "video": False},
        frontend_rtc_configuration=rtc_settings.frontend_rtc_configuration,
        server_rtc_configuration=rtc_settings.server_rtc_configuration,
        audio_receiver_size=256,
        sendback_audio=False,
    )

    if desired_streaming and ctx.state.playing and not orchestrator.running:
        orchestrator.start()
    if not desired_streaming and orchestrator.running:
        orchestrator.stop()

    _set_microphone_state(orchestrator, desired_streaming, ctx.state.playing, ctx.state.signalling)
    drained = _drain_audio_frames(orchestrator, ctx) if ctx.state.playing else 0
    if drained:
        orchestrator.state.set_debug("last_streamlit_webrtc_frame_batch", {"frames": drained})

    def _live_fragment() -> None:
        snapshot = orchestrator.state.snapshot()
        render_subtitles(snapshot, debug=debug)
        st.subheader("How this build really works")
        st.markdown(
            "- **Capture:** browser microphone via `streamlit-webrtc` (`webrtc_streamer`, SENDONLY mode).\n"
            f"- **RTC mode:** `{rtc_settings.mode}`; TURN configured=`{rtc_settings.turn_configured}`.\n"
            "- **ASR:** chunked audio is sent to local `faster-whisper`.\n"
            "- **Translation:** stable segments are translated via the configured local OpenAI-compatible endpoint.\n"
            "- **No Hugging Face TURN fallback remains:** all RTC config is explicit and environment-driven."
        )
        st.subheader("Still not magic")
        st.markdown(
            "- Browser microphone access still depends on localhost/HTTPS and user permission.\n"
            "- Whisper latency depends heavily on GPU speed and chunk size.\n"
            "- Translation remains segment-level, not token-level streaming.\n"
            "- Remote deployments may still need TURN, but local localhost usage typically does not."
        )

    if hasattr(st, "fragment"):
        @st.fragment(run_every="500ms")
        def _fragment() -> None:
            _live_fragment()

        _fragment()
    else:
        st.warning("This Streamlit version does not support `st.fragment`; the UI will only refresh on reruns.")
        _live_fragment()


if __name__ == "__main__":
    render_app()
