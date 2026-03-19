# Local-first live subtitles MVP

This repository contains a local-only MVP for low-latency live transcription + translation subtitles. It uses **FastRTC** for real-time microphone transport, **faster-whisper** on a local GPU for transcription, and a **local OpenAI-compatible LLM endpoint** (for example `http://127.0.0.1:1234/v1`) for segment-level translation.

## Architecture

```text
FastRTC microphone/session
  -> rtc.audio_handlers normalize/chunk audio
  -> asr.whisper_worker transcribes chunks on local GPU
  -> asr.stabilizer merges updates into revisable segments
  -> translation.translator translates provisional/final segments only
  -> app.state stores thread-safe snapshots
  -> app.renderer/ui_streamlit render rolling subtitles in Streamlit
```

### Why FastRTC

FastRTC is the intended real-time transport backbone. The app keeps FastRTC behind `rtc/fastrtc_session.py` so the rest of the pipeline is not coupled to raw transport APIs. The current code intentionally leaves a narrow TODO for the final application-specific FastRTC binding instead of replacing it with a homemade polling loop.

### Why Streamlit is UI only

Streamlit is used to render controls and snapshots from thread-safe state. Background worker threads do not call Streamlit directly. This keeps UI rerenders isolated and avoids using Streamlit as the streaming transport layer.

## Local runtime requirements

- Python 3.11+
- Local CUDA-capable GPU for `faster-whisper` (the default config assumes `device=cuda`)
- A local OpenAI-compatible translation endpoint serving a Qwen 3.5 9B-class model, e.g. LM Studio or another local server on `http://127.0.0.1:1234/v1`
- Microphone access in the final FastRTC deployment context

## Project layout

- `live_subtitles_local/app`: Streamlit UI, rendering, and app state
- `live_subtitles_local/rtc`: FastRTC session abstraction and audio helpers
- `live_subtitles_local/asr`: dataclasses, whisper wrapper, and transcript stabilizer
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
3. Ensure your local OpenAI-compatible translation server is running.
4. Start the Streamlit app:
   ```bash
   python live_subtitles_local/main.py
   ```

## FastRTC integration note

`rtc/fastrtc_session.py` intentionally exposes `start_stream()`, `stop_stream()`, and `handle_inbound_audio(...)`. In a production wiring step, connect your FastRTC microphone/session callbacks to `handle_inbound_audio(...)`. This repo keeps that glue thin and explicit because exact FastRTC component selection depends on your chosen Streamlit/FastRTC embedding pattern.

## Stabilization and translation behavior

- `partial`: live unstable text while speech is ongoing
- `provisional`: text has remained stable long enough to translate after debounce
- `final`: silence/boundary froze the segment for history/export

The translator only runs for provisional/final segments, caches translations, and drops stale results when a newer transcript revision wins.

## Export support

- TXT transcript export (`export/transcript.py`)
- JSON transcript export (`export/transcript.py`)
- SRT export using final segments only (`export/srt.py`)
- JSONL export (`export/jsonl_export.py`)

## Known limitations

- The final FastRTC app-specific callback wiring still needs to be connected.
- Real-time chunk size, VAD overlap, and silence thresholds will need tuning for your microphone, language mix, and GPU.
- `faster-whisper` streaming is approximated through chunked incremental transcription; exact low-latency behavior will depend on chunk cadence.
- The translation layer is segment-level, not token-level, by design.

## Next improvements

- Add explicit device selection UI backed by real microphone enumeration.
- Persist exports directly from the UI.
- Add richer subtitle styling and confidence visualization.
- Tune chunk overlap and context windows per language pair.
