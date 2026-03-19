# Local-first live subtitles MVP

This repository is built for a **local-first** browser-to-local-server workflow:

- **Browser microphone capture:** `streamlit-webrtc`
- **Transcription:** local `faster-whisper`
- **Translation:** local OpenAI-compatible endpoint such as `http://127.0.0.1:1234/v1`
- **UI + polling:** Streamlit

## Why Hugging Face TURN was removed

`streamlit-webrtc` can auto-populate ICE servers when no RTC configuration is provided. In practice, that meant the app could implicitly reach for third-party TURN/STUN providers, including a Hugging Face TURN credential flow exposed through the underlying WebRTC package.

That is the wrong tradeoff for this project:

- the app is supposed to work locally without depending on Hugging Face,
- localhost browser-to-local-server audio capture does **not** normally need TURN,
- startup should not fail just because an external TURN provider is unavailable,
- remote TURN should be explicit and environment-driven instead of hidden fallback behavior.

This repository now passes **explicit RTC configuration** into `streamlit-webrtc` so Hugging Face TURN is not used at all.

## Does localhost actually need TURN?

Usually, **no**.

For a browser connecting to a server running on the same machine via `localhost`, WebRTC can typically work with direct/local ICE candidates and no TURN relay. In that setup, the usual failure causes are:

- browser microphone permission was denied,
- the page is not running on `localhost` or HTTPS,
- browser WebRTC support is blocked or broken,
- some other local networking/browser policy issue exists.

TURN becomes much more important for **remote deployments**, especially when browsers and servers sit behind NAT, restrictive firewalls, containers, proxies, or different networks.

## RTC configuration modes

The app now supports two explicit modes:

### `LOCAL_DEV=true`

Use direct/local ICE configuration only.

- No Hugging Face TURN
- No implicit external TURN lookup
- No fake TURN defaults
- The UI clearly reports that TURN is not configured

Example:

```bash
LOCAL_DEV=true python live_subtitles_local/main.py
```

### `REMOTE_DEPLOYMENT=true`

Allow external ICE/TURN configuration from environment variables.

Supported configuration paths:

1. **Generic explicit ICE config**
   - `RTC_ICE_SERVERS_JSON` with a JSON list of ICE server objects
2. **Cloudflare/custom TURN style env vars**
   - `STUN_URLS`
   - `TURN_URLS`
   - `TURN_USERNAME`
   - `TURN_CREDENTIAL`
3. **Twilio credentials**
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`

If `REMOTE_DEPLOYMENT=true` is set without TURN, the app still starts, but the UI and logs will say that TURN is **not configured** and remote WebRTC connectivity may fail because of RTC config.

## Architecture

```text
Browser microphone (streamlit-webrtc / WebRTC)
  -> Streamlit app drains inbound audio frames as fast as possible
  -> rtc.audio_handlers cheaply normalizes/resamples PCM
  -> pipeline.audio_buffer keeps a bounded latest-audio ring buffer
  -> orchestrator ASR worker wakes every ~350 ms and transcribes a rolling 3.2 s window with overlap
  -> asr.stabilizer merges updates into revisable segments
  -> translation worker translates provisional/final segments only
  -> app.state stores thread-safe snapshots
  -> app.renderer/ui_streamlit render rolling subtitles in Streamlit
```

## What works cleanly

- Start/Stop control a real browser microphone stream.
- Whisper now runs off the dedicated ASR worker rather than the WebRTC receive path.
- RTC configuration is explicit instead of relying on implicit provider fallback.
- The UI shows whether RTC is using local/direct mode or remote env-driven mode.
- The UI shows whether TURN is configured.
- Translation health is derived from an actual `/models` health check plus runtime translation success/failure.
- The UI shows microphone state as `inactive`, `starting`, `active`, or `failed`.
- The UI surfaces receiver backlog, ring-buffer fill, ASR lag, average ASR latency, and overload/drop signals.

## What does not work magically

- This is still **not token-level realtime translation**.
- Browser microphone access still depends on **localhost or HTTPS** and user permission.
- Whisper latency depends heavily on GPU, model size, and chunk duration.
- If ASR falls behind, the app explicitly skips stale intermediate windows and keeps the newest audio/transcript work.
- Remote deployments may still require TURN depending on network topology.

## Local runtime requirements

- Python 3.11+
- Local CUDA-capable GPU for `faster-whisper` (defaults use `device=cuda`)
- A local OpenAI-compatible translation endpoint serving your translation model
- Browser microphone access on `localhost` or another secure context

## Project layout

- `live_subtitles_local/app`: Streamlit UI, rendering, and app state
- `live_subtitles_local/rtc`: audio normalization/chunking helpers and RTC mode/config helpers
- `live_subtitles_local/asr`: dataclasses, Whisper wrapper, and transcript stabilizer
- `live_subtitles_local/translation`: prompt, client, translator, and in-memory cache
- `live_subtitles_local/pipeline`: orchestration, queues, and stale job scheduling
- `live_subtitles_local/pipeline/audio_buffer.py`: bounded latest-audio PCM ring buffer and overload accounting
- `live_subtitles_local/export`: transcript/SRT/JSONL export helpers
- `live_subtitles_local/config/default.yaml`: sane local defaults
- `live_subtitles_local/tests`: lightweight unit tests

## Running it locally

1. Create and activate a virtualenv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start your local OpenAI-compatible translation server.
4. Start the app in local mode:
   ```bash
   LOCAL_DEV=true python live_subtitles_local/main.py
   ```
5. Open the displayed localhost URL in a browser.
6. Click **Start** in the sidebar and allow microphone access when prompted.

## Remote deployment TURN examples

### Example: custom or Cloudflare-style TURN env vars

```bash
export REMOTE_DEPLOYMENT=true
export STUN_URLS=stun:stun.cloudflare.com:3478
export TURN_URLS=turn:turn.example.com:3478?transport=udp,turns:turn.example.com:5349
export TURN_USERNAME=your-username
export TURN_CREDENTIAL=your-password
python live_subtitles_local/main.py
```

### Example: explicit ICE JSON

```bash
export REMOTE_DEPLOYMENT=true
export RTC_ICE_SERVERS_JSON='[
  {"urls": ["stun:stun.cloudflare.com:3478"]},
  {"urls": ["turn:turn.example.com:3478"], "username": "user", "credential": "pass"}
]'
python live_subtitles_local/main.py
```

### Example: Twilio

```bash
export REMOTE_DEPLOYMENT=true
export TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python live_subtitles_local/main.py
```

## Exact local commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
LOCAL_DEV=true python live_subtitles_local/main.py
```

## Known limitations / still stubbed

- There is no device picker yet; capture uses the browser's selected/default microphone through WebRTC.
- Transcript stabilization is chunk-based rather than a true incremental streaming decoder.
- Export actions are implemented in code but not yet surfaced as clickable UI controls.
- No screenshot artifact is included in this repository because browser screenshot tooling is not available in this environment.
