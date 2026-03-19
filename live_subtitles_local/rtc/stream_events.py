from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

import numpy as np


@dataclass(slots=True)
class AudioFrameEvent:
    samples: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class SessionEvent:
    kind: Literal["started", "stopped", "error"]
    message: str = ""
    timestamp: float = field(default_factory=time.time)
