from __future__ import annotations

import io
import logging
import threading
from typing import TYPE_CHECKING

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config.config import get_settings

if TYPE_CHECKING:
    from config.config import WhisperSettings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: WhisperForConditionalGeneration | None = None
_processor: WhisperProcessor | None = None
_device: str | None = None


def _resolve_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_internal(device: str = "auto") -> tuple[WhisperForConditionalGeneration, WhisperProcessor, str]:
    global _model, _processor, _device
    settings: WhisperSettings = get_settings().multimodal.whisper
    target_device = _resolve_device(device or settings.device)
    with _model_lock:
        if _model is None:
            try:
                logger.info("Loading Whisper model=%s on device=%s", settings.model_name, target_device)
                _processor = WhisperProcessor.from_pretrained(settings.model_name)
                _model = WhisperForConditionalGeneration.from_pretrained(settings.model_name)
                _model.eval()
                _model.to(target_device)
                _device = target_device
                logger.info("Whisper model loaded")
            except Exception as e:
                logger.error("Failed to load Whisper model: %s", e)
                raise
    return _model, _processor, _device


def load_model(device: str = "auto") -> tuple[WhisperForConditionalGeneration, WhisperProcessor, str]:
    return _load_internal(device)


def warmup(device: str = "auto") -> None:
    model, processor, _ = _load_internal(device)
    settings: WhisperSettings = get_settings().multimodal.whisper
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 80, 3000, dtype=torch.float32)
            generated = model.generate(dummy.to(model.device), max_length=1)
            logger.info("Whisper warmup completed")
    except Exception as e:
        logger.warning("Whisper warmup failed: %s", e)


def _load_audio(audio_data: bytes) -> torch.Tensor:
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_data))
        return torch.tensor(audio, dtype=torch.float32), sr
    except Exception:
        try:
            import librosa

            audio, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            return torch.tensor(audio, dtype=torch.float32), sr
        except Exception:
            raise RuntimeError("Failed to load audio data: requires soundfile or librosa")


def transcribe(audio_data: bytes, device: str = "auto") -> str:
    model, processor, _ = _load_internal(device)
    settings: WhisperSettings = get_settings().multimodal.whisper
    try:
        audio_tensor, sr = _load_audio(audio_data)
        if sr != 16000:
            try:
                import torchaudio

                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
            except Exception:
                logger.warning("torchaudio not available, proceeding with original sample rate %d", sr)
        input_features = processor(
            audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                language=settings.language,
                task="transcribe",
            )
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        logger.error("transcribe failed: %s", e)
        raise


def extract_voiceprint(audio_data: bytes, device: str = "auto") -> list[float]:
    model, processor, _ = _load_internal(device)
    try:
        audio_tensor, sr = _load_audio(audio_data)
        if sr != 16000:
            try:
                import torchaudio

                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
            except Exception:
                logger.warning("torchaudio not available, proceeding with original sample rate %d", sr)
        input_features = processor(
            audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(model.device)
        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            if hasattr(encoder_outputs, "last_hidden_state"):
                hidden = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, tuple):
                hidden = encoder_outputs[0]
            else:
                hidden = encoder_outputs
            voiceprint = hidden.mean(dim=1).squeeze(0)
        voiceprint = voiceprint / voiceprint.norm().clamp(min=1e-12)
        return voiceprint.cpu().tolist()
    except Exception as e:
        logger.error("extract_voiceprint failed: %s", e)
        raise