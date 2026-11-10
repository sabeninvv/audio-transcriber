import logging
import math
import re
from datetime import datetime
from typing import Sequence

from audio_transcriber.exceptions import TranscriptionError
from audio_transcriber.models import TranscriptionSegment

# ponytail: naive split, abbreviation list (Mr./т.е.) if false breaks matter
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[^\S\n]+")
# Реплика нулевой длины (слово с start == end) показывается столько, чтобы её успели прочитать.
_MIN_CUE_MILLISECONDS = 1_000

logger = logging.getLogger(__name__)


def generate_markdown_filename(
    *,
    source_stem: str,
    timestamp: datetime | None = None,
) -> str:
    return _timestamped_filename(source_stem=source_stem, timestamp=timestamp, extension="md")


def generate_subtitle_filename(
    *,
    source_stem: str,
    timestamp: datetime | None = None,
) -> str:
    return _timestamped_filename(source_stem=source_stem, timestamp=timestamp, extension="srt")


def _timestamped_filename(
    *,
    source_stem: str,
    timestamp: datetime | None,
    extension: str,
) -> str:
    generated_at = timestamp or datetime.now().astimezone()
    safe_timestamp = generated_at.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{source_stem}_{safe_timestamp}.{extension}"


def render_transcript_markdown(
    *,
    source_name: str,
    generated_at: datetime,
    chunk_count: int,
    original_format: str,
    transcript_text: str,
    speaker_segments: Sequence[TranscriptionSegment] | None = None,
) -> str:
    markdown = (
        "# Audio Transcription\n\n"
        f"- Source file: {source_name}\n"
        f"- Generated at: {generated_at.isoformat()}\n"
        f"- Chunks: {chunk_count}\n"
        f"- Original format: {original_format}\n"
        "- Working format: mp3\n\n"
        "## Transcript\n\n"
        f"{_break_sentences(transcript_text)}\n"
    )
    if not speaker_segments:
        return markdown

    speaker_lines = "\n".join(
        f"[{_format_seconds(segment.start_seconds)}] {_speaker_label(segment.speaker)}: {segment.text}"
        for segment in speaker_segments
    )
    return (
        f"{markdown}\n"
        "## Speaker Transcript\n\n"
        f"{speaker_lines}\n"
    )


def render_subtitles_srt(segments: Sequence[TranscriptionSegment]) -> str:
    """Собирает SRT из реплик; без реплик возвращает пустую строку.

    Рендер идёт после платных запросов, поэтому шум в данных провайдера не роняет файл:
    реплика без текста пропускается, реплика нулевой длины получает минимальную длительность.
    """
    blocks: list[str] = []
    for position, segment in enumerate(segments):
        start_ms = _milliseconds(segment.start_seconds)
        end_ms = _milliseconds(segment.end_seconds)
        text = " ".join(segment.text.split())
        if not text:
            logger.warning("Skipping SRT cue at %s: it has no text.", _format_srt_timestamp(start_ms))
            continue
        if end_ms <= start_ms:
            end_ms = start_ms + _MIN_CUE_MILLISECONDS
            if position + 1 < len(segments):
                next_start_ms = _milliseconds(segments[position + 1].start_seconds)
                if start_ms < next_start_ms < end_ms:
                    end_ms = next_start_ms
            logger.warning(
                "SRT cue at %s has an empty interval: extended to %s.",
                _format_srt_timestamp(start_ms),
                _format_srt_timestamp(end_ms),
            )
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(end_ms)}\n{text}"
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def _milliseconds(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise TranscriptionError("SRT subtitle time is invalid.")
    return math.floor(seconds * 1000 + 0.5)


def _format_srt_timestamp(total_milliseconds: int) -> str:
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _break_sentences(text: str) -> str:
    return _SENTENCE_BREAK.sub("\n", text.strip())


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(value))
    hours = total_seconds // 3_600
    minutes = (total_seconds % 3_600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _speaker_label(speaker: str) -> str:
    normalized = speaker.strip()
    if not normalized:
        return "Speaker ?"
    if normalized.lower().startswith("speaker"):
        return normalized
    return f"Speaker {normalized}"
