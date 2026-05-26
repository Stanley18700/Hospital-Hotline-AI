"""Google Cloud Text-to-Speech client.

Returns MP3 audio bytes for a given text + language. Standard voices are used
to keep TTS cost low (~$4 / million characters).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib

from app.config import settings

logger = logging.getLogger(__name__)


# One persona per language, used everywhere Cloud TTS plays back text
# (chat playback, fallback synthesis). Picked to match the Gemini Live
# prebuilt voices we use for live calls (`Aoede` en / `Charon` th) so
# the assistant sounds like the same person across text-mode TTS and
# voice-mode Live API:
#   - en: en-US-Neural2-F — warm, calm female. Closer to Aoede than the
#     older Standard-C voice (which sounded older and more robotic).
#   - th: th-TH-Neural2-C — natural female Thai neural voice. Better
#     prosody than Standard-A and pairs with Charon for Thai calls.
_VOICE_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "en": {"language_code": "en-US", "name": "en-US-Neural2-F"},
    "th": {"language_code": "th-TH", "name": "th-TH-Neural2-C"},
}


def _ensure_credentials_env() -> None:
    if settings.google_application_credentials:
        cred_path = settings.google_application_credentials
        if not pathlib.Path(cred_path).is_absolute():
            cred_path = str((pathlib.Path.cwd() / cred_path).resolve())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


class GoogleTtsClient:
    """Thin wrapper around Cloud Text-to-Speech."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            _ensure_credentials_env()
            from google.cloud import texttospeech_v1 as tts

            self._client = tts.TextToSpeechClient()
        return self._client

    async def synthesize(self, *, text: str, language: str) -> bytes:
        """Synthesize speech. Returns raw MP3 bytes.

        Raises RuntimeError on configuration / API failure.
        """

        if not text.strip():
            raise ValueError("text must not be empty")

        voice_cfg = _VOICE_BY_LANGUAGE.get(language) or _VOICE_BY_LANGUAGE["en"]

        return await asyncio.to_thread(self._synthesize_sync, text, voice_cfg)

    def _synthesize_sync(self, text: str, voice_cfg: dict[str, str]) -> bytes:
        try:
            from google.cloud import texttospeech_v1 as tts
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-texttospeech is not installed. "
                "Add it via `uv sync` / `pip install google-cloud-texttospeech`."
            ) from exc

        client = self._get_client()

        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code=voice_cfg["language_code"],
            name=voice_cfg["name"],
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
        )

        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
        except Exception as exc:
            logger.exception("Cloud TTS synthesize_speech failed")
            raise RuntimeError(f"Cloud TTS error: {exc}") from exc

        return bytes(response.audio_content)
