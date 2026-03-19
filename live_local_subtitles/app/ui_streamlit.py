from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from live_local_subtitles.app.subtitle_renderer import render_live_subtitles
from live_local_subtitles.pipeline.orchestrator import LocalSubtitlesOrchestrator, RuntimeConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
AUTO_REFRESH_MS = 350



def _get_orchestrator() -> LocalSubtitlesOrchestrator:
    if "orchestrator" not in st.session_state:
        config = RuntimeConfig.from_yaml(str(CONFIG_PATH))
        st.session_state.orchestrator = LocalSubtitlesOrchestrator(config)
    return st.session_state.orchestrator



def _render_sidebar(orchestrator: LocalSubtitlesOrchestrator) -> None:
    st.sidebar.header("Controls")
    devices = orchestrator.audio_capture.list_input_devices()
    device_options = {f"{dev['index']}: {dev['name']}": dev["index"] for dev in devices} or {"Default device": None}
    selected_device_label = st.sidebar.selectbox("Microphone device", list(device_options.keys()))
    orchestrator.audio_capture.config.device = device_options[selected_device_label]

    if st.sidebar.button("Start", use_container_width=True, disabled=orchestrator.state.running):
        orchestrator.start()
    if st.sidebar.button("Stop", use_container_width=True, disabled=not orchestrator.state.running):
        orchestrator.stop()

    source_mode = st.sidebar.radio("Source language", ["Auto detect", "Fixed"], horizontal=True)
    fixed_lang = st.sidebar.text_input("Fixed source language", value=orchestrator.config.whisper.language or "en")
    orchestrator.config.whisper.language = None if source_mode == "Auto detect" else fixed_lang.strip() or "en"

    target_lang = st.sidebar.selectbox(
        "Target language",
        ["English", "Spanish", "French", "German", "Japanese", "Korean", "Chinese"],
        index=0,
    )
    orchestrator.set_target_lang(target_lang)

    show_source = st.sidebar.toggle("Show source", value=True)
    show_translation = st.sidebar.toggle("Show translation", value=True)
    st.session_state["show_source"] = show_source
    st.session_state["show_translation"] = show_translation

    st.sidebar.subheader("Local LLM")
    orchestrator.config.llm.base_url = st.sidebar.text_input("Base URL", value=orchestrator.config.llm.base_url)
    orchestrator.config.llm.model = st.sidebar.text_input("Model", value=orchestrator.config.llm.model)
    orchestrator.config.llm.api_key = st.sidebar.text_input("API key", value=orchestrator.config.llm.api_key, type="password")
    orchestrator.config.llm.timeout_s = st.sidebar.number_input("Timeout (s)", min_value=1.0, max_value=60.0, value=float(orchestrator.config.llm.timeout_s), step=0.5)
    orchestrator.config.llm.temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=float(orchestrator.config.llm.temperature), step=0.05)
    orchestrator.config.llm.max_tokens = st.sidebar.number_input("Max tokens", min_value=16, max_value=512, value=int(orchestrator.config.llm.max_tokens), step=8)



def _render_main(orchestrator: LocalSubtitlesOrchestrator) -> None:
    st.title("Live Local Subtitles")
    st.caption("Local microphone capture + faster-whisper + local OpenAI-compatible translation.")

    transcript, translation = orchestrator.latest_segments()
    subtitle_html = render_live_subtitles(
        transcript,
        translation,
        show_source=st.session_state.get("show_source", True),
        show_translation=st.session_state.get("show_translation", True),
    )
    st.markdown(
        """
        <style>
          .subtitle-final, .subtitle-partial, .translation-final, .translation-draft, .subtitle-empty {
            padding: 0.25rem 0;
            line-height: 1.35;
            font-size: 1.45rem;
          }
          .subtitle-final { color: #ffffff; font-weight: 700; }
          .subtitle-partial { color: #ffd166; font-style: italic; }
          .translation-final { color: #8ecae6; font-weight: 600; }
          .translation-draft { color: #bde0fe; font-style: italic; }
          .subtitle-shell {
            background: #0b132b;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            min-height: 8rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="subtitle-shell">{subtitle_html}</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Transcript history")
        ordered = orchestrator.state.ordered_transcripts()
        translation_map = orchestrator.state.translations
        for segment in reversed(ordered[-50:]):
            translation_segment = translation_map.get(segment.segment_id)
            st.markdown(f"**{segment.state.value.upper()}** · `{segment.segment_id}` · rev {segment.revision}")
            st.write(segment.text)
            if translation_segment and translation_segment.text:
                st.caption(f"{translation_segment.target_lang}: {translation_segment.text}")
    with right:
        st.subheader("Debug / latency")
        snapshot = orchestrator.health_snapshot()
        st.json(
            {
                "running": snapshot["running"],
                "target_lang": snapshot["target_lang"],
                "llm_ok": snapshot["llm_ok"],
                "llm_message": snapshot["llm_message"],
                "translation_latency_ms": orchestrator.state.metrics.translation_latency_ms,
                "last_audio_chunk_ms": orchestrator.state.metrics.last_audio_chunk_ms,
                "last_error": orchestrator.state.metrics.last_error,
                "segments": snapshot["segments"],
            }
        )

        st.subheader("Export")
        st.download_button("Export JSON", data=orchestrator.export_json(), file_name="transcript.json", mime="application/json")
        st.download_button("Export TXT", data=orchestrator.export_text(), file_name="transcript.txt", mime="text/plain")
        st.download_button("Export SRT", data=orchestrator.export_srt(), file_name="subtitles.srt", mime="application/x-subrip")



def run_streamlit_app() -> None:
    st.set_page_config(page_title="Live Local Subtitles", page_icon="🎙️", layout="wide")
    orchestrator = _get_orchestrator()
    _render_sidebar(orchestrator)
    _render_main(orchestrator)
    if orchestrator.state.running:
        time.sleep(AUTO_REFRESH_MS / 1000)
        st.rerun()
