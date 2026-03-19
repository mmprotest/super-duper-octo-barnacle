# Local-first live subtitles MVP

This repository now uses the architecture that is most likely to produce a working local MVP without pretending Streamlit is something it is not:

- **Browser microphone capture:** `streamlit-webrtc`
- **Transcription:** local `faster-whisper`
- **Translation:** local OpenAI-compatible endpoint such as `http://127.0.0.1:1234/v1`
- **UI + polling:** Streamlit

## Why FastRTC was removed

FastRTC can be a good fit for a dedicated realtime service, but the previous app was not actually using it. It exposed a fake `Start` button, a fake input-device field, and a placeholder transport layer that never connected browser microphone audio to Whisper.

For a local MVP inside one Streamlit app, `streamlit-webrtc` is the cleaner choice because it gives Streamlit a real browser microphone transport instead of a scaffold. That means this repository is now honestly Streamlit-native rather than a Streamlit UI wrapped around an unwired FastRTC abstraction.

## Architecture

```text
Browser microphone (streamlit-webrtc / WebRTC)
  -> Streamlit app drains inbound audio frames
  -> rtc.audio_handlers normalize/chunk audio
  -> asr.whisper_worker transcribes chunks on local GPU
  -> asr.stabilizer merges updates into revisable segments
  -> translation.translator translates provisional/final segments only
  -> app.state stores thread-safe snapshots
  -> app.renderer/ui_streamlit render rolling subtitles in Streamlit
```

## What works cleanly

- Start/Stop now control an actual browser microphone stream request plus backend processing.
- Whisper receives real audio frames from the browser media stream.
- Translation health is derived from an actual `/models` health check plus runtime translation success/failure.
- The UI shows microphone state as `inactive`, `starting`, `active`, or `failed`.

## What does not work magically

- This is still **not token-level realtime translation**. Translation happens on stabilized segments.
- Browser microphone access still depends on **localhost or HTTPS** and user permission.
- If your local model server does not expose a compatible `/models` route, translator health will show as unavailable until translation requests succeed.
- Whisper latency depends heavily on GPU, model size, and chunk duration.

## Local runtime requirements

- Python 3.11+
- Local CUDA-capable GPU for `faster-whisper` (defaults use `device=cuda`)
- A local OpenAI-compatible translation endpoint serving your translation model
- Browser microphone access on `localhost` or another secure context

## Project layout

- `live_subtitles_local/app`: Streamlit UI, rendering, and app state
- `live_subtitles_local/rtc`: audio normalization/chunking helpers and stream event dataclasses
- `live_subtitles_local/asr`: dataclasses, Whisper wrapper, and transcript stabilizer
- `live_subtitles_local/translation`: prompt, client, translator, and in-memory cache
- `live_subtitles_local/pipeline`: orchestration, queues, and stale job scheduling
- `live_subtitles_local/export`: transcript/SRT/JSONL export helpers
- `live_subtitles_local/config/default.yaml`: sane local defaults
- `live_subtitles_local/tests`: lightweight unit tests

## Running it

1. Create and activate a virtualenv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start your local OpenAI-compatible translation server.
4. Start the Streamlit app:
   ```bash
   python live_subtitles_local/main.py
   ```
5. Open the displayed localhost URL in a browser.
6. Click **Start** in the sidebar and allow microphone access when the browser asks.

## Exact commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python live_subtitles_local/main.py
```

## Known limitations / still stubbed

- There is no device picker yet; capture uses the browser's selected/default microphone through WebRTC.
- Transcript stabilization is chunk-based rather than a true incremental streaming decoder.
- Export actions are implemented in code but not yet surfaced as clickable UI controls.
- No screenshot artifact is included in this repository because that depends on browser tooling outside the app itself.
