from __future__ import annotations

import logging
from typing import Callable

from live_subtitles_local.rtc.stream_events import AudioFrameEvent, SessionEvent

logger = logging.getLogger(__name__)

AudioCallback = Callable[[AudioFrameEvent], None]
SessionCallback = Callable[[SessionEvent], None]


class FastRTCSession:
    """Owns real-time streaming lifecycle behind a narrow abstraction.

    The app depends on this interface rather than raw FastRTC primitives. The exact UI binding
    to FastRTC components depends on the final deployment pattern (WebRTC/WebSocket transport).
    """

    def __init__(self) -> None:
        self._audio_callbacks: list[AudioCallback] = []
        self._session_callbacks: list[SessionCallback] = []
        self._running = False
        self._transport = None
        try:
            import fastrtc  # type: ignore  # pragma: no cover

            self._transport = fastrtc
        except Exception:
            logger.warning("FastRTC is not importable yet; session can still be instantiated for UI wiring/tests.")

    @property
    def running(self) -> bool:
        return self._running

    def register_audio_callback(self, callback: AudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def register_session_callback(self, callback: SessionCallback) -> None:
        self._session_callbacks.append(callback)

    def start_stream(self) -> None:
        if self._running:
            return
        self._running = True
        # TODO: Bind FastRTC microphone track/session primitives here once the exact deployment shape is chosen.
        self._emit_session(SessionEvent(kind="started", message="FastRTC session marked as running"))

    def stop_stream(self) -> None:
        if not self._running:
            return
        self._running = False
        # TODO: Close/detach the concrete FastRTC session/peer connection here.
        self._emit_session(SessionEvent(kind="stopped", message="FastRTC session stopped"))

    def handle_inbound_audio(self, event: AudioFrameEvent) -> None:
        # This method is the intended sink for FastRTC on_frame/on_chunk callbacks.
        for callback in list(self._audio_callbacks):
            callback(event)

    def _emit_session(self, event: SessionEvent) -> None:
        for callback in list(self._session_callbacks):
            callback(event)
