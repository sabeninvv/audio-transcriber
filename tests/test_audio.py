from pathlib import Path

import pytest

import audio_transcriber.audio as audio_module
from audio_transcriber.audio import (
    calculate_chunk_duration_seconds,
    ensure_ffmpeg_available,
    normalize_to_mp3,
    probe_audio_metadata,
    validate_audio_path,
)
from audio_transcriber.exceptions import (
    FfmpegNotAvailableError,
    UnsupportedAudioFormatError,
)


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".mp4", ".avi", ".webm", ".ogg", ".oga", ".m4a"])
def test_validate_audio_path_accepts_supported_extensions(
    tmp_path: Path,
    extension: str,
) -> None:
    input_path = tmp_path / f"meeting{extension}"
    input_path.write_bytes(b"audio")

    assert validate_audio_path(input_path) == input_path


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".mp4", ".avi", ".webm", ".ogg", ".oga", ".m4a"])
def test_is_supported_audio_path_accepts_supported_extensions(
    tmp_path: Path,
    extension: str,
) -> None:
    input_path = tmp_path / f"meeting{extension}"
    input_path.write_bytes(b"audio")

    assert hasattr(audio_module, "is_supported_audio_path")
    assert audio_module.is_supported_audio_path(input_path) is True


def test_is_supported_audio_path_rejects_unsupported_extensions(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.txt"
    input_path.write_bytes(b"audio")

    assert hasattr(audio_module, "is_supported_audio_path")
    assert audio_module.is_supported_audio_path(input_path) is False


def test_validate_audio_path_rejects_unsupported_extensions(tmp_path: Path) -> None:
    input_path = tmp_path / "meeting.txt"
    input_path.write_bytes(b"audio")

    with pytest.raises(UnsupportedAudioFormatError, match="txt"):
        validate_audio_path(input_path)


def test_ensure_ffmpeg_available_requires_ffmpeg_and_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binaries = {
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffprobe": None,
    }
    monkeypatch.setattr(audio_module, "which", lambda name: binaries.get(name))

    with pytest.raises(FfmpegNotAvailableError, match="ffprobe"):
        ensure_ffmpeg_available()


def test_probe_audio_metadata_maps_missing_binary_to_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "meeting.mp3"
    input_path.write_bytes(b"audio")

    def raise_missing_binary(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(audio_module.subprocess, "run", raise_missing_binary)

    with pytest.raises(FfmpegNotAvailableError, match="ffprobe"):
        probe_audio_metadata(input_path)


def test_calculate_chunk_duration_seconds_uses_bitrate_and_target_size() -> None:
    duration = calculate_chunk_duration_seconds(
        bitrate_bps=128_000,
        max_chunk_bytes=24 * 1024 * 1024,
    )

    assert duration == 1_572


def test_calculate_chunk_duration_seconds_never_returns_less_than_one_second() -> None:
    duration = calculate_chunk_duration_seconds(
        bitrate_bps=512_000_000,
        max_chunk_bytes=1,
    )

    assert duration == 1


def test_calculate_chunk_duration_seconds_caps_result_by_max_duration() -> None:
    duration = calculate_chunk_duration_seconds(
        bitrate_bps=64_000,
        max_chunk_bytes=24 * 1024 * 1024,
        max_duration_seconds=1_300,
    )

    assert duration == 1_300


def test_normalize_to_mp3_uses_common_ffmpeg_call_for_webm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "meeting.webm"
    output_path = tmp_path / "normalized.mp3"
    input_path.write_bytes(b"audio")

    captured_command: list[str] = []

    def fake_run_command(command: list[str]) -> object:
        captured_command.extend(command)
        return object()

    monkeypatch.setattr(audio_module, "_run_command", fake_run_command)

    result = normalize_to_mp3(input_path, output_path)

    assert result == output_path
    assert captured_command == [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-b:a",
        audio_module.DEFAULT_MP3_BITRATE,
        str(output_path),
    ]
