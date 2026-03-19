# Live Local Subtitles

Production-oriented scaffold for a fully local, low-latency live transcription + translation subtitles app built with Python 3.11+, faster-whisper, a local OpenAI-compatible LLM endpoint, and Streamlit.

## Proposed repo tree

```text
live_local_subtitles/
├── app/
│   ├── state.py
│   ├── subtitle_renderer.py
│   └── ui_streamlit.py
├── audio/
│   ├── capture.py
│   ├── resample.py
│   ├── ring_buffer.py
│   └── vad.py
├── asr/
│   ├── schemas.py
│   ├── stabilizer.py
│   └── whisper_worker.py
├── config/
│   └── default.yaml
├── export/
│   ├── srt.py
│   ├── text_export.py
│   └── transcript_json.py
├── pipeline/
│   ├── events.py
│   ├── orchestrator.py
│   └── queues.py
├── tests/
│   ├── test_exports.py
│   ├── test_ring_buffer.py
│   ├── test_stabilizer.py
│   └── test_translation.py
├── translation/
│   ├── cache.py
│   ├── llm_client.py
│   ├── prompts.py
│   └── translator.py
├── __init__.py
└── main.py
pyproject.toml
README.md
```

## Architecture and runtime flow

### Runtime pipeline

1. **Microphone capture**
   - `audio.capture.MicrophoneAudioCapture` reads mono microphone frames with `sounddevice`.
   - Frames are resampled to the ASR target rate when needed.
   - Audio is appended into a rolling `AudioRingBuffer`.
   - Every step interval, a rolling window is emitted as an `ASRChunk`.

2. **VAD + rolling window policy**
   - A lightweight energy VAD gates obviously empty windows.
   - Default policy:
     - frame size: `100 ms`
     - step size: `400 ms`
     - rolling window: `2200 ms`
     - overlap: `1800 ms`
     - VAD hangover: `450 ms`
   - This keeps latency low while preserving enough context for Whisper to avoid fragmenting speech too aggressively.

3. **ASR worker**
   - `asr.whisper_worker.FasterWhisperWorker` keeps a single faster-whisper model warm-loaded.
   - Only speech-marked chunks are submitted.
   - Queue is intentionally small; oldest work is dropped before fresh work is delayed.
   - The worker emits normalized `ASRResult` objects keyed to segment ids.

4. **Segment stabilization**
   - `asr.stabilizer.TranscriptStabilizer` merges overlapping ASR revisions for the same rolling segment.
   - Segment states:
     - `partial`: fresh or still changing
     - `provisional`: repeated with high similarity enough times to look stable
     - `final`: stable and followed by enough silence / gap
   - Translation is only triggered for `provisional` and `final` states.

5. **Translation worker**
   - `translation.translator.TranslationWorker` runs off a LIFO queue so the newest segment wins.
   - It debounces updates and drops stale revisions.
   - It never retranslates an unchanged `(segment_id, revision, target_lang)` pair.
   - Translation failure does not block source subtitle display.

6. **UI state**
   - `pipeline.orchestrator.LocalSubtitlesOrchestrator` owns runtime components and app state.
   - `app.ui_streamlit` only renders state and mutates runtime settings.
   - All transcript and translation updates are keyed by `segment_id` and `revision`.

### Data model

#### `TranscriptSegment`
- `segment_id`
- `start_ms`
- `end_ms`
- `source_lang`
- `text`
- `state: partial | provisional | final`
- `revision`
- `confidence`
- `created_at`
- `updated_at`

#### `TranslationSegment`
- `segment_id`
- `source_revision`
- `target_lang`
- `text`
- `state: draft | final | failed`
- `latency_ms`
- `created_at`
- `updated_at`
- `error`

## Design decisions

### Local-only runtime
- Runtime depends only on local microphone access, local GPU/CPU inference, and a loopback-accessible local LLM endpoint.
- After model install, the app is designed to function offline.
- No runtime cloud API integration exists in the codebase.

### Practical streaming policy
- **Step size:** `400 ms` for responsive updates without overloading ASR.
- **Window size:** `2200 ms` to give Whisper enough context for punctuation and word completion.
- **Overlap:** implicit via window minus step; ensures partial revisions can converge instead of flapping.
- **Provisional threshold:** same segment text must stabilize at similarity `>= 0.88` for two consecutive updates.
- **Finalization:** silence or gap `>= 900 ms`, or a sufficiently long stable gap.
- **Translation trigger:** provisional and final only.
- **Replacement policy:** newer translation for the same `segment_id` replaces older revisions.
- **Stale job policy:** translation queue is LIFO and drops older revisions when backlogged.
- **Flicker control:** partial subtitles are shown separately and styled differently; translation waits for stable text.

### Failure handling stance
- Mic disconnects should surface as runtime errors while leaving the app process alive.
- Empty/non-speech audio is filtered early.
- If ASR lags, stale work is dropped instead of queueing indefinitely.
- Translation timeouts or local API failures create failed translation segments but source subtitles continue.
- Local model server health is surfaced in the debug panel.

### Why this architecture
- It is modular enough to test core logic outside Streamlit.
- It avoids over-engineering: one capture loop, one ASR worker, one translation worker, one orchestrator.
- It explicitly optimizes for perceived subtitle stability over trying to stream every token.

## Translation prompt contract

System prompt policy:
- translate only the current segment
- optionally include a small amount of recent final context
- return translation only
- keep subtitle text concise and readable
- no explanations, notes, or source repetition
- deterministic, low-temperature inference

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

```bash
streamlit run live_local_subtitles/app/ui_streamlit.py
```

Or:

```bash
python -m live_local_subtitles.main
```

## Local model assumptions

### faster-whisper
- Intended to run on CUDA when available.
- Default model config is `large-v3` with `float16` on `cuda`.
- If you run on CPU, update `config/default.yaml` accordingly.

### Local translation endpoint
Default config targets:
- base URL: `http://127.0.0.1:1234/v1`
- model: `qwen/qwen3.5-9b-instruct`

Any OpenAI-compatible local server can be swapped in so long as `/models` and `/chat/completions` behave conventionally.

## Testing

```bash
pytest
```

## TODOs for the last 10–20%
- Tune VAD thresholds per microphone and room noise profile.
- Improve segment iding across overlaps so adjacent speaker turns split more naturally.
- Add optional word-level timestamp smoothing when using Whisper models that expose strong word timings.
- Add a dedicated UI event bus instead of relying on Streamlit reruns for refresh.
- Add richer observability for dropped ASR chunks and queue depth over time.
- Add real device reconnect handling and a user-facing retry flow.
- Benchmark GPU contention between local ASR and local LLM inference on the target machine.
