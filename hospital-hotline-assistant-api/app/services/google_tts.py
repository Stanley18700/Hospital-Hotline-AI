"""Gemini Text-to-Speech client via Vertex AI.

Primary: Uses Gemini 2.5 Flash TTS with Kore (English) and Aoede (Thai) voices
for natural, human-like speech output optimized for medical hotline quality.
Gemini returns WAV or raw PCM, wrapped in WAV container if needed.

Fallback: If Gemini TTS fails, falls back to Cloud TTS with Neural2 (English)
and Standard (Thai) voices. Always returns MP3.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import pathlib
import struct
import wave
from collections.abc import AsyncGenerator

from app.config import settings

logger = logging.getLogger(__name__)


# Voice mapping for Gemini TTS (primary)
_GEMINI_VOICE_BY_LANGUAGE: dict[str, str] = {
    "en": "Kore",    # Calm, warm female voice for medical hotline
    "th": "Aoede",   # Warm female voice with multilingual support
}

# Fallback voice config for Cloud TTS (if Gemini TTS fails)
_FALLBACK_VOICE_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "en": {"language_code": "en-US", "name": "en-US-Neural2-F"},
    "th": {"language_code": "th-TH", "name": "th-TH-Standard-A"},
}


def _ensure_credentials_env() -> None:
    if settings.google_application_credentials:
        cred_path = settings.google_application_credentials
        if not pathlib.Path(cred_path).is_absolute():
            cred_path = str((pathlib.Path.cwd() / cred_path).resolve())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM audio (24kHz, 16-bit, mono) in a WAV container using stdlib only.
    
    Gemini TTS returns raw PCM bytes. This wraps them in a valid WAV header
    without external dependencies (no ffmpeg, pydub, or any system tools needed).
    """
    try:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)          # mono
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(24000)      # Gemini TTS sample rate
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()
    except Exception as exc:
        logger.exception("PCM to WAV conversion failed")
        raise RuntimeError(f"Audio conversion error: {exc}") from exc


def _make_wav_header(
    sample_rate: int = 24000,
    channels: int = 1,
    bit_depth: int = 16,
) -> bytes:
    """Generate a WAV header for a stream of unknown total length.

    The RIFF/WAVE format normally encodes ``chunk_size`` and ``data_size``
    up front, but during streaming we don't know how many PCM bytes will
    follow. We use ``0xFFFFFFFF`` placeholders, which every mainstream
    decoder (Chromium ``<audio>`` / MediaSource, ffmpeg, VLC) treats as
    "play until EOF". This lets the browser start playing as soon as
    the first PCM chunk arrives, cutting the 2-4 s pre-roll latency.

    Defaults match Gemini 2.5 Flash TTS output: 24 kHz, mono, 16-bit PCM.
    """

    byte_rate = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    data_size = 0xFFFFFFFF
    chunk_size = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, block_align, bit_depth,
        b"data", data_size,
    )


def _wrap_ssml(text: str, language: str) -> tuple[str, bool]:
    """
    Wrap text in basic SSML for supported languages.
    Returns (processed_text, is_ssml).
    Only English uses SSML for now.
    """
    if language != "en":
        return text, False
    # Add gentle pause after greeting-style sentences, calm breathing rhythm
    ssml = (
        "<speak>"
        + text.replace(". ", '. <break time="300ms"/> ')
        .replace("? ", '? <break time="400ms"/> ')
        + "</speak>"
    )
    return ssml, True


class GoogleTtsClient:
    """Gemini TTS client with Cloud TTS fallback."""

    def __init__(self) -> None:
        self._genai_client = None
        self._cloud_tts_client = None

    def _get_genai_client(self):
        """Get or create Gemini API client."""
        if self._genai_client is None:
            _ensure_credentials_env()
            try:
                from google import genai
                self._genai_client = genai.Client(
                    vertexai=True,
                    project=settings.google_cloud_project,
                    location=settings.google_cloud_location,
                )
            except Exception as exc:
                logger.warning(f"Failed to initialize Gemini client: {exc}")
                self._genai_client = None
        return self._genai_client

    def _get_cloud_tts_client(self):
        """Get or create Cloud TTS client for fallback."""
        if self._cloud_tts_client is None:
            _ensure_credentials_env()
            from google.cloud import texttospeech_v1 as tts
            self._cloud_tts_client = tts.TextToSpeechClient()
        return self._cloud_tts_client

    async def synthesize(self, *, text: str, language: str) -> bytes:
        """Synthesize speech. Returns raw MP3 bytes.

        Tries Gemini TTS first. Falls back to Cloud TTS if Gemini fails.
        Raises RuntimeError on configuration / API failure.
        """

        if not text.strip():
            raise ValueError("text must not be empty")

        return await asyncio.to_thread(self._synthesize_sync, text, language)

    async def synthesize_stream(
        self, *, text: str, language: str
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from Gemini TTS as they arrive.

        Unlike :meth:`synthesize`, this does NOT wait for the full audio
        before returning bytes. It opens a Gemini ``generate_content``
        streaming call and yields chunks as they land, prefixed with a
        WAV header so the browser ``<audio>`` element can start playing
        immediately. The header uses ``0xFFFFFFFF`` sentinel sizes (see
        :func:`_make_wav_header`) so we don't need to know the final
        duration up-front.

        Audio format matches Gemini 2.5 Flash TTS: 24 kHz, mono, 16-bit
        PCM. Chunks that are missing inline audio data (e.g. text-only
        progress events) are silently skipped.

        Raises:
            ValueError: if ``text`` is empty / whitespace.
            RuntimeError: if the Gemini client cannot be initialized or
                the streaming SDK is unavailable. Callers are expected
                to fall back to :meth:`synthesize` in that case.
        """

        if not text.strip():
            raise ValueError("text must not be empty")

        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. "
                "Add it via `uv sync` / `pip install google-genai`."
            ) from exc

        voice_name = _GEMINI_VOICE_BY_LANGUAGE.get(language, "Kore")
        processed_text, _ = _wrap_ssml(text, language)

        def _stream_sync():
            client = self._get_genai_client()
            if not client:
                raise RuntimeError("Gemini client not initialized")
            return client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-tts",
                contents=processed_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                ),
            )

        stream = await asyncio.to_thread(_stream_sync)

        yield _make_wav_header()

        # Iterate the synchronous SDK stream off the event-loop thread so
        # the per-chunk network read doesn't block other requests.
        sentinel = object()
        iterator = iter(stream)
        while True:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            try:
                part = chunk.candidates[0].content.parts[0]
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    yield inline.data
            except (IndexError, AttributeError):
                continue

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        """Primary: Gemini TTS. Fallback: Cloud TTS."""
        
        # Try Gemini TTS first
        try:
            return self._synthesize_gemini(text, language)
        except Exception as exc:
            logger.warning(
                f"Gemini TTS failed for language '{language}': {exc}. "
                "Falling back to Cloud TTS."
            )
            try:
                return self._synthesize_fallback(text, language)
            except Exception as fallback_exc:
                logger.exception("Both Gemini TTS and Cloud TTS fallback failed")
                raise RuntimeError(
                    f"TTS synthesis failed. Gemini: {exc}, Fallback: {fallback_exc}"
                ) from fallback_exc

    def _synthesize_gemini(self, text: str, language: str) -> bytes:
        """Synthesize using Gemini 2.5 Flash TTS via Vertex AI."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. "
                "Add it via `uv sync` / `pip install google-genai`."
            ) from exc

        client = self._get_genai_client()
        if not client:
            raise RuntimeError("Gemini client not initialized")

        voice_name = _GEMINI_VOICE_BY_LANGUAGE.get(language, "Kore")
        
        # Gemini TTS doesn't use SSML in the same way; use plain text
        processed_text, _ = _wrap_ssml(text, language)

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=processed_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )

        # Extract audio from response and check MIME type
        inline_data = response.candidates[0].content.parts[0].inline_data
        audio_bytes = inline_data.data
        mime_type = inline_data.mime_type or ""
        
        # If already in WAV format, return as-is; otherwise wrap PCM in WAV
        if mime_type.startswith("audio/wav") or mime_type.startswith("audio/wave"):
            # Already WAV, return directly
            return audio_bytes
        else:
            # Assume raw PCM or audio/L16 — wrap in WAV container
            wav_bytes = _pcm_to_wav(audio_bytes)
            return wav_bytes

    def _synthesize_fallback(self, text: str, language: str) -> bytes:
        """Fallback: Synthesize using Cloud Text-to-Speech (Neural2/Standard)."""
        try:
            from google.cloud import texttospeech_v1 as tts
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-texttospeech is not installed. "
                "Add it via `uv sync` / `pip install google-cloud-texttospeech`."
            ) from exc

        client = self._get_cloud_tts_client()
        
        voice_cfg = _FALLBACK_VOICE_BY_LANGUAGE.get(language) or _FALLBACK_VOICE_BY_LANGUAGE["en"]
        processed_text, is_ssml = _wrap_ssml(text, language)
        
        if is_ssml:
            synthesis_input = tts.SynthesisInput(ssml=processed_text)
        else:
            synthesis_input = tts.SynthesisInput(text=processed_text)
        
        voice = tts.VoiceSelectionParams(
            language_code=voice_cfg["language_code"],
            name=voice_cfg["name"],
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            speaking_rate=0.95,
            pitch=0.0,
            effects_profile_id=["telephony-class-application"],
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        return bytes(response.audio_content)
