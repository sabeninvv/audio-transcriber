import logging
import math
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI, OpenAIError

from audio_transcriber.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_SUBTITLE_MODEL,
    DEFAULT_TRANSCRIPTION_KEYWORDS,
    DEFAULT_TRANSCRIPTION_LANGUAGES,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSCRIPTION_PROMPT,
    OPENAI_MODEL_PREFIX,
)
from audio_transcriber.exceptions import OpenAITranscriptionError
from audio_transcriber.models import (
    TranscriptionResult,
    TranscriptionSegment,
    build_prompt,
    ensure_subtitles_allowed,
)

logger = logging.getLogger(__name__)

# Documented whisper-1 failure threshold: higher means the segment text collapsed.
_WHISPER_MAX_COMPRESSION_RATIO = 2.4


class SupportsAsyncTranscriptions(Protocol):
    audio: object


class OpenAITranscriptionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_TRANSCRIPTION_MODEL,
        prompt: str = DEFAULT_TRANSCRIPTION_PROMPT,
        languages: tuple[str, ...] = DEFAULT_TRANSCRIPTION_LANGUAGES,
        keywords: tuple[str, ...] = DEFAULT_TRANSCRIPTION_KEYWORDS,
        diarization_enabled: bool = False,
        diarization_model: str = DEFAULT_DIARIZATION_MODEL,
        subtitle_model: str = DEFAULT_SUBTITLE_MODEL,
        known_speaker_names: tuple[str, ...] = (),
        known_speaker_references: tuple[str, ...] = (),
        timestamps_enabled: bool = False,
        client: AsyncOpenAI | SupportsAsyncTranscriptions | None = None,
    ) -> None:
        if timestamps_enabled:
            ensure_subtitles_allowed(diarization_enabled=diarization_enabled)
        self._timestamps_enabled = timestamps_enabled
        self._model = model
        self._prompt = prompt
        self._languages = languages
        self._keywords = keywords
        self._diarization_enabled = diarization_enabled
        self._diarization_model = diarization_model
        self._subtitle_model = subtitle_model
        self._known_speaker_names = known_speaker_names
        self._known_speaker_references = known_speaker_references
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def transcribe_file(
        self,
        path: Path,
        *,
        prompt_suffix: str | None = None,
    ) -> TranscriptionResult:
        try:
            with path.open("rb") as audio_file:
                if self._timestamps_enabled:
                    response = await self._create_timestamped_transcription(audio_file=audio_file)
                    return TranscriptionResult(
                        text=_extract_transcript_text(response),
                        segments=_extract_timestamped_segments(response),
                    )

                if self._diarization_enabled:
                    response = await self._create_diarized_transcription(audio_file)
                    return TranscriptionResult(
                        text=_extract_transcript_text(response),
                        segments=_extract_diarized_segments(response),
                    )

                response = await self._create_standard_transcription(
                    audio_file=audio_file,
                    prompt_suffix=prompt_suffix,
                )
                return TranscriptionResult(text=_extract_transcript_text(response))
        except OpenAIError as error:
            logger.exception(
                "OpenAI transcription request failed for model=%s file=%s "
                "error_type=%s status_code=%s request_id=%s body=%r response_text=%r",
                self._active_model(),
                path,
                type(error).__name__,
                getattr(error, "status_code", None),
                getattr(error, "request_id", None),
                getattr(error, "body", None),
                _extract_response_text(error),
            )
            raise OpenAITranscriptionError("OpenAI transcription request failed.") from error

    async def _create_standard_transcription(
        self,
        *,
        audio_file: Any,
        prompt_suffix: str | None,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "file": audio_file,
            "model": self._model,
            "chunking_strategy": "auto",
            "response_format": "json",
        }
        prompt = build_prompt(self._prompt, prompt_suffix)
        if prompt:
            request_kwargs["prompt"] = prompt

        if self._model.removeprefix(OPENAI_MODEL_PREFIX) == DEFAULT_TRANSCRIPTION_MODEL:
            extra_body: dict[str, list[str]] = {}
            if self._languages:
                extra_body["languages"] = list(self._languages)
            if self._keywords:
                extra_body["keywords"] = list(self._keywords)
            if extra_body:
                request_kwargs["extra_body"] = extra_body
        elif self._languages:
            # Legacy model path accepts a single language hint.
            request_kwargs["language"] = self._languages[0]

        return await self._client.audio.transcriptions.create(**request_kwargs)

    async def _create_timestamped_transcription(self, *, audio_file: Any) -> Any:
        # whisper-1 treats prompt as previously spoken text and then repeats it.
        request_kwargs: dict[str, Any] = {
            "file": audio_file,
            "model": self._subtitle_model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        # whisper-1 accepts one language; several configured languages stay on auto-detect.
        if len(self._languages) == 1:
            request_kwargs["language"] = self._languages[0]
        return await self._client.audio.transcriptions.create(**request_kwargs)

    def _active_model(self) -> str:
        if self._timestamps_enabled:
            return self._subtitle_model
        if self._diarization_enabled:
            return self._diarization_model
        return self._model

    async def _create_diarized_transcription(self, audio_file: Any) -> Any:
        request_kwargs: dict[str, Any] = {
            "file": audio_file,
            "model": self._diarization_model,
            "chunking_strategy": "auto",
            "response_format": "diarized_json",
        }
        if self._known_speaker_names:
            request_kwargs["known_speaker_names"] = list(self._known_speaker_names)
        if self._known_speaker_references:
            request_kwargs["known_speaker_references"] = list(self._known_speaker_references)
        return await self._client.audio.transcriptions.create(**request_kwargs)


def _extract_response_text(error: OpenAIError) -> str | None:
    response = getattr(error, "response", None)
    if response is None:
        return None

    try:
        return getattr(response, "text", None)
    except Exception:
        return None


def _extract_transcript_text(response: Any) -> str:
    text = _coerce_response_value(response, "text")
    if not isinstance(text, str) or not text.strip():
        raise OpenAITranscriptionError("OpenAI response does not include transcript text.")
    return text


def _extract_diarized_segments(response: Any) -> tuple[TranscriptionSegment, ...]:
    raw_segments = _coerce_response_value(response, "segments")
    if not isinstance(raw_segments, list):
        return ()

    segments: list[TranscriptionSegment] = []
    for item in raw_segments:
        start = _coerce_response_value(item, "start")
        end = _coerce_response_value(item, "end")
        speaker = _coerce_response_value(item, "speaker")
        text = _coerce_response_value(item, "text")
        if not isinstance(start, (float, int)):
            continue
        if not isinstance(end, (float, int)):
            continue
        if not isinstance(speaker, str) or not speaker:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        segments.append(
            TranscriptionSegment(
                start_seconds=float(start),
                end_seconds=float(end),
                speaker=speaker,
                text=text.strip(),
            )
        )

    return tuple(segments)


def _extract_timestamped_segments(response: Any) -> tuple[TranscriptionSegment, ...]:
    raw_segments = _coerce_response_value(response, "segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise OpenAITranscriptionError("OpenAI response does not include transcript segments.")

    segments: list[TranscriptionSegment] = []
    previous_start: float | None = None
    for index, item in enumerate(raw_segments, start=1):
        start = _require_timestamp(item, "start", index)
        end = _require_timestamp(item, "end", index)
        if start < 0 or start >= end:
            logger.info(
                "Omitting whisper segment index=%s because its time range %s..%s is invalid.",
                index,
                start,
                end,
            )
            continue
        if previous_start is not None and start < previous_start:
            raise OpenAITranscriptionError(f"OpenAI transcript segment {index} is out of order.")
        previous_start = start
        text = _coerce_response_value(item, "text")
        if not isinstance(text, str) or not text.strip():
            logger.info("Omitting whisper segment index=%s because it has no text.", index)
            continue
        if _whisper_segment_repeated(item):
            logger.info(
                "Omitting repeated whisper segment index=%s compression_ratio=%s.",
                index,
                _coerce_response_value(item, "compression_ratio"),
            )
            continue
        segments.append(
            TranscriptionSegment(
                start_seconds=start,
                end_seconds=end,
                speaker="",
                text=text.strip(),
            )
        )
    if not segments:
        raise OpenAITranscriptionError("OpenAI response does not include transcript segments.")
    return tuple(segments)


def _whisper_segment_repeated(segment: Any) -> bool:
    value = _coerce_response_value(segment, "compression_ratio")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value > _WHISPER_MAX_COMPRESSION_RATIO


def _require_timestamp(container: Any, field: str, index: int) -> float:
    value = _coerce_response_value(container, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OpenAITranscriptionError(
            f"OpenAI transcript segment {index} does not include a {field} timestamp."
        )
    return float(value)


def _coerce_response_value(container: Any, field: str) -> Any:
    if hasattr(container, field):
        return getattr(container, field)
    if isinstance(container, dict):
        return container.get(field)
    return None
