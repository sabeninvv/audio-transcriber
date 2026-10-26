from pathlib import Path

import pytest

import main as cli_module
from audio_transcriber.exceptions import TranscriptionError


def test_build_parser_describes_supported_media_types() -> None:
    parser = cli_module.build_parser()
    help_text = parser.format_help()

    assert "mp4" in help_text
    assert "avi" in help_text
    assert "directory" in help_text


def test_main_runs_transcription_and_prints_output(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    output_dir = tmp_path / "transcripts"
    result_path = output_dir / "result.md"
    captured: dict[str, tuple[Path, Path]] = {}

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        captured["args"] = (input_path, output_dir)
        return result_path

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_path), "--output", str(output_dir)]
    )

    assert exit_code == 0
    assert captured["args"] == (input_path, output_dir)
    assert capsys.readouterr().out == f"{result_path}\n"


def test_main_transcribes_supported_media_files_from_directory_and_skips_unsupported(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "meeting.wav").write_bytes(b"audio")
    (input_dir / "podcast.mp3").write_bytes(b"audio")
    (input_dir / "notes.txt").write_text("text", encoding="utf-8")
    nested_dir = input_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "ignored.wav").write_bytes(b"audio")
    output_dir = tmp_path / "transcripts"
    calls: list[tuple[Path, Path]] = []

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        calls.append((input_path, output_dir))
        return output_dir / f"{input_path.stem}.md"

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_dir), "--output", str(output_dir)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        (input_dir / "meeting.wav", output_dir),
        (input_dir / "podcast.mp3", output_dir),
    ]
    assert captured.out == f"{output_dir / 'meeting.md'}\n{output_dir / 'podcast.md'}\n"
    assert captured.err == "Skipping unsupported file: notes.txt\n"


def test_main_continues_after_supported_file_failure_in_directory_mode(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "broken.wav").write_bytes(b"audio")
    (input_dir / "meeting.mp3").write_bytes(b"audio")
    (input_dir / "notes.txt").write_text("text", encoding="utf-8")
    output_dir = tmp_path / "transcripts"
    calls: list[tuple[Path, Path]] = []

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        calls.append((input_path, output_dir))
        if input_path.name == "broken.wav":
            raise TranscriptionError("boom")
        return output_dir / f"{input_path.stem}.md"

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_dir), "--output", str(output_dir)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert calls == [
        (input_dir / "broken.wav", output_dir),
        (input_dir / "meeting.mp3", output_dir),
    ]
    assert captured.out == f"{output_dir / 'meeting.md'}\n"
    assert captured.err == (
        "Error transcribing broken.wav: boom\n"
        "Skipping unsupported file: notes.txt\n"
    )


def test_main_returns_error_when_directory_has_no_supported_media_files(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.txt").write_text("text", encoding="utf-8")
    output_dir = tmp_path / "transcripts"
    called: list[tuple[Path, Path]] = []

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        called.append((input_path, output_dir))
        return output_dir / f"{input_path.stem}.md"

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_dir), "--output", str(output_dir)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert called == []
    assert captured.out == ""
    assert captured.err == (
        "Skipping unsupported file: notes.txt\n"
        f"Error: no supported media files found in directory: {input_dir}\n"
    )


def test_main_continues_after_unexpected_file_error_in_directory_mode(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "broken.wav").write_bytes(b"audio")
    (input_dir / "meeting.mp3").write_bytes(b"audio")
    output_dir = tmp_path / "transcripts"
    calls: list[tuple[Path, Path]] = []
    exit_code: int | None = None
    raised: Exception | None = None

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        calls.append((input_path, output_dir))
        if input_path.name == "broken.wav":
            raise RuntimeError("boom")
        return output_dir / f"{input_path.stem}.md"

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    try:
        exit_code = cli_module.main(
            ["--input", str(input_dir), "--output", str(output_dir)]
        )
    except Exception as error:
        raised = error
    captured = capsys.readouterr()

    assert raised is None
    assert exit_code == 1
    assert calls == [
        (input_dir / "broken.wav", output_dir),
        (input_dir / "meeting.mp3", output_dir),
    ]
    assert captured.out == f"{output_dir / 'meeting.md'}\n"
    assert captured.err == "Error transcribing broken.wav: boom\n"


def test_main_returns_error_code_for_transcription_errors(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    output_dir = tmp_path / "transcripts"

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        raise TranscriptionError("boom")

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_path), "--output", str(output_dir)]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == "Error: boom\n"


def test_main_prints_root_cause_for_transcription_errors(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    output_dir = tmp_path / "transcripts"

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str = "md",
    ) -> Path:
        raise TranscriptionError("boom") from ValueError("bad detail")

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    exit_code = cli_module.main(
        ["--input", str(input_path), "--output", str(output_dir)]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == "Error: boom\nCause: bad detail\n"


def test_main_passes_srt_format_for_file_and_directory(
    monkeypatch: object,
    capsys: object,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    input_dir = tmp_path / "media"
    input_dir.mkdir()
    (input_dir / "meeting.mp3").write_bytes(b"audio")
    (input_dir / "notes.txt").write_text("text", encoding="utf-8")
    output_dir = tmp_path / "subtitles"
    formats: list[str] = []

    async def fake_transcribe_audio(
        *,
        input_path: Path,
        output_dir: Path,
        output_format: str,
    ) -> Path:
        formats.append(output_format)
        return output_dir / f"{input_path.stem}.srt"

    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe_audio)

    file_exit = cli_module.main(
        ["--input", str(input_path), "--output", str(output_dir), "--format", "srt"]
    )
    directory_exit = cli_module.main(
        ["--input", str(input_dir), "--output", str(output_dir), "--format", "srt"]
    )
    captured = capsys.readouterr()

    assert file_exit == 0
    assert directory_exit == 0
    assert formats == ["srt", "srt"]
    assert captured.out == f"{output_dir / 'input.srt'}\n{output_dir / 'meeting.srt'}\n"
    assert captured.err == "Skipping unsupported file: notes.txt\n"


def test_main_rejects_unknown_subtitle_format(capsys: object, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli_module.main(
            [
                "--input",
                str(tmp_path / "input.wav"),
                "--output",
                str(tmp_path / "out"),
                "--format",
                "vtt",
            ]
        )

    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
