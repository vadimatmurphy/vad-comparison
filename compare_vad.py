"""Core VAD comparison engine — runs Silero VAD v5 and v6 on audio data."""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from dataclasses import dataclass, asdict
from pathlib import Path

import onnxruntime  # type: ignore
from livekit.agents.vad import VADEventType
from livekit.plugins import silero
from livekit.agents.utils.audio import AudioBuffer
from livekit import rtc

SUPPORTED_SAMPLE_RATES = (8000, 16000)
V5_MODEL_PATH = Path(__file__).parent / "resources" / "silero_vad_v5.onnx"

_resource_files = ExitStack()
atexit.register(_resource_files.close)


@dataclass
class VadEvent:
    type: str  # "start" | "end"
    timestamp: float
    speech_duration: float
    silence_duration: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_vad(
    *,
    sample_rate: int = 16000,
    use_v5: bool = False,
    min_silence_duration: float = 0.5,
    min_speech_duration: float = 0.1,
) -> silero.VAD:
    from livekit.plugins.silero import onnx_model

    if not use_v5:
        return silero.VAD.load(
            sample_rate=sample_rate,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
        )

    original_loader = onnx_model.new_inference_session

    def _custom_new_inference_session(force_cpu: bool):
        opts = onnxruntime.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        providers = None
        if (
            force_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            providers = ["CPUExecutionProvider"]

        return onnxruntime.InferenceSession(
            str(V5_MODEL_PATH), providers=providers, sess_options=opts
        )

    try:
        onnx_model.new_inference_session = _custom_new_inference_session
        return silero.VAD.load(
            sample_rate=sample_rate,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
        )
    finally:
        onnx_model.new_inference_session = original_loader


def pcm_to_frame(pcm_bytes: bytes, sample_rate: int = 16000) -> rtc.AudioFrame:
    """Convert raw 16-bit PCM bytes to an AudioFrame."""
    num_samples = len(pcm_bytes) // 2  # 16-bit = 2 bytes per sample
    return rtc.AudioFrame(
        data=pcm_bytes,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
