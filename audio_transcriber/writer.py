import re
from datetime import datetime
from typing import Sequence

from audio_transcriber.openai_client import TranscriptionSegment

# ponytail: naive split, abbreviation list (Mr./т.е.) if false breaks matter
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[^\S\n]+")


def generate_markdown_filename(
    *,
    source_stem: str,
    timestamp: datetime | None = None,
) -> str:
    generated_at = timestamp or datetime.now().astimezone()
    safe_timestamp = generated_at.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{source_stem}_{safe_timestamp}.md"


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
