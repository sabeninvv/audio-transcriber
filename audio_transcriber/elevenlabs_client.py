import logging
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from audio_transcriber.config import (
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_TRANSCRIPTION_LANGUAGES,
)
from audio_transcriber.exceptions import ElevenLabsTranscriptionError
from audio_transcriber.models import TranscriptionResult, TranscriptionSegment, ensure_subtitles_allowed

logger = logging.getLogger(__name__)

_SPEECH_TO_TEXT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
# Чанк до 1300 с распознаётся дольше дефолтных 5 с httpx.
_REQUEST_TIMEOUT_SECONDS = 600.0
# Обычная длина одной реплики субтитра.
_MAX_SEGMENT_SECONDS = 7.0
_SENTENCE_ENDINGS = (".", "?", "!", "…")


class _Word(BaseModel):
    text: str
    type: Literal["word", "spacing", "audio_event"]
    start: float | None = None
    end: float | None = None
    speaker_id: str | None = None


class _SpeechToTextResponse(BaseModel):
    text: str
    words: list[_Word]


class ElevenLabsTranscriptionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ELEVENLABS_MODEL,
        languages: tuple[str, ...] = DEFAULT_TRANSCRIPTION_LANGUAGES,
        keywords: tuple[str, ...] = (),
        diarization_enabled: bool = False,
        timestamps_enabled: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timestamps_enabled:
            ensure_subtitles_allowed(diarization_enabled=diarization_enabled)
        self._api_key = api_key
        self._model = model
        self._languages = languages
        self._keywords = keywords
        self._diarization_enabled = diarization_enabled
        self._timestamps_enabled = timestamps_enabled
        self._transport = transport

    async def transcribe_file(
        self,
        path: Path,
        *,
        prompt_suffix: str | None = None,
    ) -> TranscriptionResult:
        """Scribe не принимает промпт, поэтому prompt_suffix игнорируется."""
        form: dict[str, str | bool | list[str]] = {
            "model_id": self._model,
            "diarize": self._diarization_enabled,
            "tag_audio_events": False,
            "timestamps_granularity": "word",
        }
        # Scribe принимает один язык; несколько настроенных языков оставляют автоопределение.
        if len(self._languages) == 1:
            form["language_code"] = self._languages[0]
        if self._keywords:
            form["keyterms"] = list(self._keywords)

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as client:
                with path.open("rb") as audio_file:
                    response = await client.post(
                        _SPEECH_TO_TEXT_URL,
                        headers={"xi-api-key": self._api_key},
                        data=form,
                        files={"file": (path.name, audio_file, "audio/mpeg")},
                    )
        except httpx.HTTPError as error:
            logger.exception(
                "ElevenLabs transcription request failed for model=%s file=%s error_type=%s",
                self._model,
                path,
                type(error).__name__,
            )
            raise ElevenLabsTranscriptionError("ElevenLabs transcription request failed.") from error

        if response.is_error:
            logger.error(
                "ElevenLabs transcription request failed for model=%s file=%s status_code=%s body=%r",
                self._model,
                path,
                response.status_code,
                response.text,
            )
            raise ElevenLabsTranscriptionError("ElevenLabs transcription request failed.")

        try:
            payload = _SpeechToTextResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise ElevenLabsTranscriptionError("ElevenLabs response has an unexpected shape.") from error
        if not payload.text.strip():
            raise ElevenLabsTranscriptionError("ElevenLabs response does not include transcript text.")

        if not (self._diarization_enabled or self._timestamps_enabled):
            return TranscriptionResult(text=payload.text)
        return TranscriptionResult(text=payload.text, segments=_group_words(payload.words))


def _group_words(words: list[_Word]) -> tuple[TranscriptionSegment, ...]:
    """Склеивает слова в реплики: новая реплика на смене спикера, после конца предложения или по длине."""
    segments: list[TranscriptionSegment] = []
    texts: list[str] = []
    segment_start = segment_end = 0.0
    segment_speaker = ""
    sentence_closed = False
    for word in words:
        if word.type != "word" or word.start is None or word.end is None:
            continue
        speaker = word.speaker_id or ""
        if texts and (
            sentence_closed
            or speaker != segment_speaker
            or word.end - segment_start > _MAX_SEGMENT_SECONDS
        ):
            segments.append(
                TranscriptionSegment(
                    start_seconds=segment_start,
                    end_seconds=segment_end,
                    speaker=segment_speaker,
                    text=" ".join(texts),
                )
            )
            texts = []
        if not texts:
            segment_start, segment_speaker = word.start, speaker
        texts.append(word.text.strip())
        segment_end = word.end
        sentence_closed = word.text.rstrip().endswith(_SENTENCE_ENDINGS)

    if texts:
        segments.append(
            TranscriptionSegment(
                start_seconds=segment_start,
                end_seconds=segment_end,
                speaker=segment_speaker,
                text=" ".join(texts),
            )
        )
    return tuple(segments)
