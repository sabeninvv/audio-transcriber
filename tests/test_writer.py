from datetime import datetime, timezone

import pytest

from audio_transcriber.exceptions import TranscriptionError
from audio_transcriber.openai_client import TranscriptionSegment
from audio_transcriber.writer import (
    generate_markdown_filename,
    generate_subtitle_filename,
    render_subtitles_srt,
    render_transcript_markdown,
)


def test_generate_markdown_filename_uses_source_stem_and_timestamp() -> None:
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    filename = generate_markdown_filename(source_stem="meeting", timestamp=generated_at)

    assert filename == "meeting_2026-04-07T12-30-11.md"


def test_render_transcript_markdown_includes_metadata_and_body() -> None:
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    markdown = render_transcript_markdown(
        source_name="meeting.wav",
        generated_at=generated_at,
        chunk_count=4,
        original_format="wav",
        transcript_text="Line one.\nLine two.",
    )

    assert markdown == (
        "# Audio Transcription\n\n"
        "- Source file: meeting.wav\n"
        "- Generated at: 2026-04-07T12:30:11+00:00\n"
        "- Chunks: 4\n"
        "- Original format: wav\n"
        "- Working format: mp3\n\n"
        "## Transcript\n\n"
        "Line one.\n"
        "Line two.\n"
    )


def test_render_transcript_markdown_puts_each_sentence_on_its_own_line() -> None:
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    markdown = render_transcript_markdown(
        source_name="meeting.wav",
        generated_at=generated_at,
        chunk_count=2,
        original_format="wav",
        transcript_text="Hello. How are you? Fine! It's 3.14 now.\n\nNext chunk. Done.",
    )

    assert (
        "## Transcript\n\n"
        "Hello.\n"
        "How are you?\n"
        "Fine!\n"
        "It's 3.14 now.\n"
        "\n"
        "Next chunk.\n"
        "Done.\n"
    ) in markdown


def test_render_transcript_markdown_includes_speaker_sections_when_present() -> None:
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    markdown = render_transcript_markdown(
        source_name="meeting.wav",
        generated_at=generated_at,
        chunk_count=1,
        original_format="wav",
        transcript_text="Agent hello.",
        speaker_segments=[
            TranscriptionSegment(
                start_seconds=0.0,
                end_seconds=0.8,
                speaker="A",
                text="hello",
            )
        ],
    )

    assert "## Speaker Transcript" in markdown
    assert "[00:00:00] Speaker A: hello" in markdown


def test_generate_subtitle_filename_uses_source_stem_and_timestamp() -> None:
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    filename = generate_subtitle_filename(source_stem="meeting", timestamp=generated_at)

    assert filename == "meeting_2026-04-07T12-30-11.srt"


def test_render_subtitles_srt_numbers_cues_and_keeps_unicode() -> None:
    rendered = render_subtitles_srt(
        [
            TranscriptionSegment(0.0, 1.25, "", " привет\nмир "),
            TranscriptionSegment(1.25, 2.5, "", "hello"),
        ]
    )

    assert rendered == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "привет мир\n"
        "\n"
        "2\n"
        "00:00:01,250 --> 00:00:02,500\n"
        "hello\n"
    )


def test_render_subtitles_srt_rolls_milliseconds_into_minutes_and_hours() -> None:
    rendered = render_subtitles_srt(
        [
            TranscriptionSegment(59.0, 59.9996, "", "minute"),
            TranscriptionSegment(3599.0, 3599.9996, "", "hour"),
        ]
    )

    assert "00:00:59,000 --> 00:01:00,000\nminute\n" in rendered
    assert "00:59:59,000 --> 01:00:00,000\nhour\n" in rendered


def test_render_subtitles_srt_rejects_interval_collapsed_by_rounding() -> None:
    with pytest.raises(TranscriptionError, match="interval"):
        render_subtitles_srt([TranscriptionSegment(1.0001, 1.0004, "", "too short")])


def test_render_subtitles_srt_rejects_empty_input_and_blank_text() -> None:
    with pytest.raises(TranscriptionError, match="segment"):
        render_subtitles_srt([])
    with pytest.raises(TranscriptionError, match="text"):
        render_subtitles_srt([TranscriptionSegment(0.0, 1.0, "", "  ")])
