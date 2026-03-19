from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VoiceActivityDecision:
    is_speech: bool
    rms: float
    peak: float
    silence_ms: int


class EnergyVAD:
    """Straightforward energy gate with a hangover window for speech continuity."""

    def __init__(self, sample_rate: int, threshold: float = 0.012, hangover_ms: int = 450) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self._silence_ms = 0

    def evaluate(self, audio: np.ndarray) -> VoiceActivityDecision:
        if audio.size == 0:
            self._silence_ms += 0
            return VoiceActivityDecision(False, 0.0, 0.0, self._silence_ms)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        frame_ms = int((audio.size / self.sample_rate) * 1000)
        if rms >= self.threshold or peak >= self.threshold * 3:
            self._silence_ms = 0
            return VoiceActivityDecision(True, rms, peak, self._silence_ms)
        self._silence_ms += frame_ms
        return VoiceActivityDecision(self._silence_ms <= self.hangover_ms, rms, peak, self._silence_ms)
