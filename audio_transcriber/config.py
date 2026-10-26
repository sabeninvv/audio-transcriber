from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Mapping

from audio_transcriber.exceptions import TranscriptionError

DEFAULT_TRANSCRIPTION_MODEL = "gpt-transcribe"
DEFAULT_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
SUBTITLE_TRANSCRIPTION_MODEL = "whisper-1"
SUBTITLE_CHUNK_MAX_DURATION_SECONDS = 30
DEFAULT_DIARIZATION_ENABLED = False
DEFAULT_CHUNK_MAX_BYTES = 24 * 1024 * 1024
DEFAULT_CHUNK_MAX_DURATION_SECONDS = 1_300
DEFAULT_CHUNK_CONTEXT_CHARS = 220
DEFAULT_TRANSCRIPTION_LANGUAGES = ("ru", "en")
DEFAULT_TRANSCRIPTION_KEYWORDS = (
    "ReelsMaker",
    "enrichment",
    "For You",
    "UV sync",
    "scheduler",
    "ContentLoader",
    "media segments",
)
DEFAULT_TRANSCRIPTION_PROMPT = (
    "Technical team meeting in Russian with occasional English terms. "
    "Transcribe verbatim and preserve product names, acronyms, and service names."
)
FORBIDDEN_KEYWORD_CHARS = ("<", ">", "\n", "\r")


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    openai_api_key: str
    model: str = DEFAULT_TRANSCRIPTION_MODEL
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT
    languages: tuple[str, ...] = DEFAULT_TRANSCRIPTION_LANGUAGES
    keywords: tuple[str, ...] = DEFAULT_TRANSCRIPTION_KEYWORDS
    chunk_context_chars: int = DEFAULT_CHUNK_CONTEXT_CHARS
    diarization_enabled: bool = DEFAULT_DIARIZATION_ENABLED
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    known_speaker_names: tuple[str, ...] = ()
    known_speaker_references: tuple[str, ...] = ()
    chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES
    chunk_max_duration_seconds: int = DEFAULT_CHUNK_MAX_DURATION_SECONDS
    temp_dir: Path | None = None


def load_config(env: Mapping[str, str] | None = None) -> TranscriptionConfig:
    source = env or environ
    openai_api_key = source.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise TranscriptionError("OPENAI_API_KEY environment variable is required.")

    model = source.get("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL).strip()
    prompt = source.get("TRANSCRIPTION_PROMPT", DEFAULT_TRANSCRIPTION_PROMPT).strip()
    languages = _parse_csv_values(
        source.get("TRANSCRIPTION_LANGUAGES"),
        default=DEFAULT_TRANSCRIPTION_LANGUAGES,
    )
    keywords = _parse_csv_values(
        source.get("TRANSCRIPTION_KEYWORDS"),
        default=DEFAULT_TRANSCRIPTION_KEYWORDS,
    )
    _validate_keywords(keywords)

    chunk_context_chars = _parse_positive_int(
        source.get("TRANSCRIPTION_CHUNK_CONTEXT_CHARS"),
        var_name="TRANSCRIPTION_CHUNK_CONTEXT_CHARS",
        default=DEFAULT_CHUNK_CONTEXT_CHARS,
    )
    diarization_enabled = _parse_bool(
        source.get(
            "TRANSCRIPTION_DIARIZATION_ENABLED",
            "true" if DEFAULT_DIARIZATION_ENABLED else "false",
        )
    )
    diarization_model = source.get("TRANSCRIPTION_DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL).strip()
    known_speaker_names = _parse_csv_values(source.get("TRANSCRIPTION_KNOWN_SPEAKER_NAMES"))
    known_speaker_references = _parse_reference_values(
        source.get("TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES")
    )

    if len(known_speaker_names) != len(known_speaker_references):
        raise TranscriptionError(
            "TRANSCRIPTION_KNOWN_SPEAKER_NAMES and TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES "
            "must have the same number of entries."
        )
    if len(known_speaker_names) > 4:
        raise TranscriptionError("At most 4 known speakers are supported by the transcription API.")
    if (known_speaker_names or known_speaker_references) and not diarization_enabled:
        raise TranscriptionError(
            "Known speaker hints require TRANSCRIPTION_DIARIZATION_ENABLED=true."
        )

    return TranscriptionConfig(
        openai_api_key=openai_api_key,
        model=model,
        prompt=prompt,
        languages=languages,
        keywords=keywords,
        chunk_context_chars=chunk_context_chars,
        diarization_enabled=diarization_enabled,
        diarization_model=diarization_model,
        known_speaker_names=known_speaker_names,
        known_speaker_references=known_speaker_references,
    )


def _parse_csv_values(raw_value: str | None, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if raw_value is None:
        return default
    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return values if values else default


def _parse_reference_values(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return ()

    normalized = raw_value.strip()
    if not normalized:
        return ()
    if "|" in normalized:
        return tuple(part.strip() for part in normalized.split("|") if part.strip())
    if "\n" in normalized:
        return tuple(part.strip() for part in normalized.splitlines() if part.strip())
    return (normalized,)


def _validate_keywords(keywords: tuple[str, ...]) -> None:
    for keyword in keywords:
        if any(char in keyword for char in FORBIDDEN_KEYWORD_CHARS):
            raise TranscriptionError(
                "TRANSCRIPTION_KEYWORDS cannot contain '<', '>', carriage return, or line feed."
            )


def _parse_positive_int(raw_value: str | None, *, var_name: str, default: int) -> int:
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise TranscriptionError(f"{var_name} must be a positive integer.") from error

    if value <= 0:
        raise TranscriptionError(f"{var_name} must be a positive integer.")
    return value


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}
