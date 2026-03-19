from __future__ import annotations

from pathlib import Path

import streamlit as st

from live_subtitles_local.app.renderer import render_subtitles
from live_subtitles_local.pipeline.orchestrator import LiveSubtitleOrchestrator
from live_subtitles_local.asr.schemas import SessionConfig

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


def get_orchestrator() -> LiveSubtitleOrchestrator:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = LiveSubtitleOrchestrator(CONFIG_PATH)
    return st.session_state.orchestrator


def _sidebar_config(defaults: SessionConfig) -> tuple[SessionConfig, bool, bool, bool]:
    with st.sidebar:
        st.header("Controls")
        start_clicked = st.button("Start", use_container_width=True)
        stop_clicked = st.button("Stop", use_container_width=True)
        debug = st.toggle("Debug", value=False)
        config = SessionConfig(
            input_device=st.text_input("Input device", value=defaults.input_device or "") or None,
            target_language=st.text_input("Target language", value=defaults.target_language),
            source_language_mode=st.selectbox("Source language mode", ["auto", "fixed"], index=0 if defaults.source_language_mode == "auto" else 1),
            source_language=st.text_input("Source language", value=defaults.source_language or "") or None,
            whisper_model_name=st.text_input("Whisper model", value=defaults.whisper_model_name),
            whisper_device=defaults.whisper_device,
            whisper_compute_type=st.selectbox("Compute type", ["float16", "int8_float16", "int8"], index=["float16", "int8_float16", "int8"].index(defaults.whisper_compute_type)),
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


def render_app() -> None:
    st.set_page_config(page_title="Local Live Subtitles", layout="wide")
    st.title("Local-first live transcription + translation subtitles")
    st.caption("FastRTC handles real-time audio transport; Streamlit renders state snapshots only.")

    orchestrator = get_orchestrator()
    defaults = orchestrator.state.snapshot().config
    config, start_clicked, stop_clicked, debug = _sidebar_config(defaults)
    orchestrator.update_config(config)

    if start_clicked:
        orchestrator.start()
    if stop_clicked:
        orchestrator.stop()

    def _live_fragment() -> None:
        snapshot = orchestrator.state.snapshot()
        render_subtitles(snapshot, debug=debug)

    if hasattr(st, "fragment"):
        @st.fragment(run_every="500ms")
        def _fragment() -> None:
            _live_fragment()

        _fragment()
    else:
        st.info("Upgrade Streamlit for st.fragment-based live refresh; rendering current snapshot once per rerun.")
        _live_fragment()

    st.subheader("Runtime notes")
    st.markdown(
        "- Wire FastRTC microphone callbacks into `FastRTCSession.handle_inbound_audio(...)` in the final deployment.\n"
        "- The app keeps translation visible until a newer non-stale segment translation arrives.\n"
        "- Background workers never call Streamlit APIs directly; they only mutate thread-safe state."
    )


if __name__ == "__main__":
    render_app()
