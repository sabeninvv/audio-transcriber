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
