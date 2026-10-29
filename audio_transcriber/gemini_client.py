import asyncio
import base64
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

from audio_transcriber.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TRANSCRIPTION_KEYWORDS,
    DEFAULT_TRANSCRIPTION_LANGUAGES,
    DEFAULT_TRANSCRIPTION_PROMPT,
)
from audio_transcriber.exceptions import GeminiTranscriptionError
from audio_transcriber.models import TranscriptionResult, build_prompt

logger = logging.getLogger(__name__)

_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Чанк до 1300 с распознаётся дольше дефолтных 5 с httpx.
_REQUEST_TIMEOUT_SECONDS = 600.0
# Без явной инструкции модель добавляет вступление и комментарии к тексту.
_TRANSCRIBE_INSTRUCTION = (
    "Transcribe this audio verbatim. "
    "Return only the transcript text, without timestamps, speaker labels, or comments."
)
_FINISH_REASON_STOP = "STOP"


class _Part(BaseModel):
    text: str = ""


class _Content(BaseModel):
    parts: list[_Part] = []


class _Candidate(BaseModel):
    content: _Content = _Content()
    finish_reason: str | None = Field(default=None, alias="finishReason")


class _PromptFeedback(BaseModel):
    block_reason: str | None = Field(default=None, alias="blockReason")


class _GenerateContentResponse(BaseModel):
    # Заблокированный промпт приходит без candidates, причина — в promptFeedback.
    candidates: list[_Candidate] = []
    prompt_feedback: _PromptFeedback = Field(default=_PromptFeedback(), alias="promptFeedback")


class GeminiTranscriptionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        prompt: str = DEFAULT_TRANSCRIPTION_PROMPT,
        languages: tuple[str, ...] = DEFAULT_TRANSCRIPTION_LANGUAGES,
        keywords: tuple[str, ...] = DEFAULT_TRANSCRIPTION_KEYWORDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._prompt = prompt
        self._languages = languages
        self._keywords = keywords
        self._transport = transport

    async def transcribe_file(
        self,
        path: Path,
        *,
        prompt_suffix: str | None = None,
    ) -> TranscriptionResult:
        audio_data = await asyncio.to_thread(_encode_audio, path)
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": build_prompt(self._instruction(), prompt_suffix)},
                        {
                            "inlineData": {
                                "mimeType": "audio/mp3",
                                "data": audio_data,
                            }
                        },
                    ]
                }
            ],
            # Дословной расшифровке рассуждения не нужны; отключить их у Gemini 3 нельзя.
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
        }

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post(
                    f"{_MODELS_URL}/{self._model}:generateContent",
                    headers={"x-goog-api-key": self._api_key},
                    json=body,
                )
        except httpx.HTTPError as error:
            logger.exception(
                "Gemini transcription request failed for model=%s file=%s error_type=%s",
                self._model,
                path,
                type(error).__name__,
            )
            raise GeminiTranscriptionError("Gemini transcription request failed.") from error

        if response.is_error:
            logger.error(
                "Gemini transcription request failed for model=%s file=%s status_code=%s body=%r",
                self._model,
                path,
                response.status_code,
                response.text,
            )
            raise GeminiTranscriptionError("Gemini transcription request failed.")

        try:
            payload = _GenerateContentResponse.model_validate_json(response.content)
        except ValidationError as error:
            logger.error(
                "Gemini response has an unexpected shape for model=%s file=%s body=%r",
                self._model,
                path,
                response.text,
            )
            raise GeminiTranscriptionError("Gemini response has an unexpected shape.") from error
        if not payload.candidates:
            raise GeminiTranscriptionError(
                "Gemini response does not include candidates, "
                f"block_reason={payload.prompt_feedback.block_reason}."
            )

        candidate = payload.candidates[0]
        # MAX_TOKENS и блокировка дают обрезанный текст, молча его не пишем.
        if candidate.finish_reason != _FINISH_REASON_STOP:
            raise GeminiTranscriptionError(
                f"Gemini transcription stopped with finish_reason={candidate.finish_reason}."
            )
        text = "".join(part.text for part in candidate.content.parts)
        if not text.strip():
            raise GeminiTranscriptionError("Gemini response does not include transcript text.")
        return TranscriptionResult(text=text)

    def _instruction(self) -> str:
        lines = [_TRANSCRIBE_INSTRUCTION, self._prompt]
        if self._languages:
            lines.append(f"Languages: {', '.join(self._languages)}.")
        if self._keywords:
            lines.append(f"Glossary: {', '.join(self._keywords)}.")
        return "\n".join(line for line in lines if line)


def _encode_audio(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
