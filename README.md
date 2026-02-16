# Silero VAD v5 vs v6 Comparison

A live, browser-based tool to compare **Silero Voice Activity Detection v5 and v6** side by side. Record from your microphone and see speech start/end events from both models in real time.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

## Why?

Silero VAD v6 changed how speech segments are detected and merged compared to v5. This tool helps you visualise the differences — v6 tends to merge adjacent speech segments while v5 keeps them more granular.

## Quick start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/vadimatmurphy/vad-comparison.git
cd vad-comparison

# Download the v5 ONNX model into resources/
mkdir -p resources
curl -L -o resources/silero_vad_v5.onnx \
  "https://github.com/livekit/agents/raw/livekit-agents%401.2.6/livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/resources/silero_vad.onnx"

uv run python server.py
```

Open **http://localhost:8000** in your browser.

## How it works

1. **Click Record** — your browser captures mic audio at 16 kHz mono
2. Audio streams via WebSocket to the backend
3. The backend feeds the same audio to both VAD v5 and v6 in parallel
4. Speech events (SOS / EOS) are sent back and displayed in two columns
5. **Click Stop** — see the summary and play back what you just recorded

## Parameters

You can tweak these before recording:

| Parameter | Default | Description |
|-----------|---------|-------------|
| min silence (s) | 0.5 | Minimum silence duration to trigger end-of-speech |
| min speech (s) | 0.1 | Minimum speech duration to trigger start-of-speech |

## Project structure

```
├── server.py          # FastAPI web server with WebSocket handler
├── compare_vad.py     # Core VAD engine — loads v5/v6 models
├── index.html         # Browser UI (vanilla HTML/JS, no build step)
├── resources/
│   └── silero_vad_v5.onnx   # Silero VAD v5 ONNX model (download separately)
├── data/input/        # Sample audio files for testing
└── pyproject.toml     # Dependencies
```

## Dependencies

- [livekit-agents](https://github.com/livekit/agents) — audio utilities and VAD framework
- [livekit-plugins-silero](https://github.com/livekit/agents) — Silero VAD v6 plugin
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — web server
- The v5 ONNX model must be downloaded separately (see Quick start)

## CLI comparison (optional)

You can also compare VAD on audio files directly:

```bash
uv run python -c "
import asyncio
from compare_vad import load_vad
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.vad import VADEventType

async def run(use_v5):
    vad = load_vad(use_v5=use_v5)
    stream = vad.stream()
    async for frame in audio_frames_from_file('data/input/p_2tsga5unx60_RM_aF2pZomzb85e_audio.oga', sample_rate=16000, num_channels=1):
        stream.push_frame(frame)
    stream.end_input()
    async for event in stream:
        if event.type in (VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH):
            print(f'[{event.timestamp:7.3f}s] {event.type.name}')
    await stream.aclose()

print('=== v6 ===')
asyncio.run(run(False))
print('=== v5 ===')
asyncio.run(run(True))
"
```
