from dataclasses import dataclass
from enum import StrEnum

from audio_transcriber.exceptions import TranscriptionError


class OutputFormat(StrEnum):
    MARKDOWN = "md"
    SRT = "srt"


def ensure_subtitles_allowed(*, diarization_enabled: bool) -> None:
    if diarization_enabled:
        raise TranscriptionError(
            "SRT subtitles cannot be combined with speaker diarization."
        )


def parse_output_format(value: OutputFormat | str) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    if isinstance(value, str):
        try:
            return OutputFormat(value)
        except ValueError as error:
            raise TranscriptionError(
                f"Unsupported output format: {value!r}. Expected 'md' or 'srt'."
            ) from error
    raise TranscriptionError(
        f"Unsupported output format: {value!r}. Expected 'md' or 'srt'."
    )


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    segments: tuple[TranscriptionSegment, ...] = ()


def build_prompt(base_prompt: str, prompt_suffix: str | None) -> str:
    """Склеивает промпт с хвостом предыдущего чанка; общая для клиентов с промптом."""
    prefix = base_prompt.strip()
    suffix = (prompt_suffix or "").strip()
    if not suffix:
        return prefix
    if not prefix:
        return f"Previous chunk context:\n{suffix}"
    return (
        f"{prefix}\n\n"
        "Use this trailing context from the previous chunk for continuity. "
        "Do not repeat it verbatim:\n"
        f"{suffix}"
    )
