#!/usr/bin/env python3
"""Web server for live VAD v5 vs v6 comparison."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from livekit.agents.vad import VADEventType

from compare_vad import load_vad, pcm_to_frame, V5_MODEL_PATH

SAMPLE_RATE = 16000
app = FastAPI()


@app.get("/")
async def index():
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Read VAD params from the first message
    try:
        params_raw = await ws.receive_text()
        params = json.loads(params_raw)
    except (WebSocketDisconnect, json.JSONDecodeError):
        return

    min_silence = params.get("min_silence_duration", 0.5)
    min_speech = params.get("min_speech_duration", 0.1)

    # Load both models
    vad_v6 = load_vad(
        sample_rate=SAMPLE_RATE,
        use_v5=False,
        min_silence_duration=min_silence,
        min_speech_duration=min_speech,
    )
    vad_v5 = load_vad(
        sample_rate=SAMPLE_RATE,
        use_v5=True,
        min_silence_duration=min_silence,
        min_speech_duration=min_speech,
    )

    stream_v6 = vad_v6.stream()
    stream_v5 = vad_v5.stream()

    stop_event = asyncio.Event()

    async def read_events(stream, version: str):
        """Read VAD events from a stream and send them to the client."""
        try:
            async for event in stream:
                if stop_event.is_set():
                    break
                if event.type == VADEventType.START_OF_SPEECH:
                    await ws.send_text(json.dumps({
                        "version": version,
                        "type": "start",
                        "timestamp": round(event.timestamp, 3),
                        "speech_duration": round(event.speech_duration, 3),
                    }))
                elif event.type == VADEventType.END_OF_SPEECH:
                    await ws.send_text(json.dumps({
                        "version": version,
                        "type": "end",
                        "timestamp": round(event.timestamp, 3),
                        "silence_duration": round(event.silence_duration, 3),
                    }))
        except Exception:
            pass

    # Start event readers
    task_v6 = asyncio.create_task(read_events(stream_v6, "v6"))
    task_v5 = asyncio.create_task(read_events(stream_v5, "v5"))

    try:
        while True:
            data = await ws.receive_bytes()

            # Check for stop signal (empty message)
            if len(data) == 0:
                break

            frame = pcm_to_frame(data, sample_rate=SAMPLE_RATE)
            stream_v6.push_frame(frame)
            stream_v5.push_frame(frame)
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        stream_v6.end_input()
        stream_v5.end_input()
        await asyncio.gather(task_v6, task_v5, return_exceptions=True)
        await stream_v6.aclose()
        await stream_v5.aclose()


def main():
    import uvicorn

    if not V5_MODEL_PATH.exists():
        print(f"ERROR: V5 model not found at {V5_MODEL_PATH}")
        print("Please place silero_vad_v5.onnx in the resources/ directory.")
        return

    print("Starting VAD Comparison server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
