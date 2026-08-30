from datetime import datetime, timezone

from audio_transcriber.openai_client import TranscriptionSegment
from audio_transcriber.writer import (
    generate_markdown_filename,
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
