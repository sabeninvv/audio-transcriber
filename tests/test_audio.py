import subprocess
from pathlib import Path

import pytest

import audio_transcriber.audio as audio_module
from audio_transcriber.audio import (
    calculate_chunk_duration_seconds,
    ensure_ffmpeg_available,
    normalize_to_mp3,
    probe_audio_metadata,
    split_mp3_into_chunks,
    validate_audio_path,
)
from audio_transcriber.exceptions import (
    AudioChunkingError,
    FfmpegNotAvailableError,
    UnsupportedAudioFormatError,
)


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".mp4", ".mov", ".avi", ".webm", ".ogg", ".oga", ".m4a"])
def test_validate_audio_path_accepts_supported_extensions(
    tmp_path: Path,
    extension: str,
) -> None:
    input_path = tmp_path / f"meeting{extension}"
    input_path.write_bytes(b"audio")

    assert validate_audio_path(input_path) == input_path


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".mp4", ".mov", ".avi", ".webm", ".ogg", ".oga", ".m4a"])
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


def test_split_mp3_chunks_stay_on_parent_timeline(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-ac", "1", str(source)],
        check=True,
        capture_output=True,
    )
    normalized = normalize_to_mp3(source, tmp_path / "norm.mp3")
    chunks = split_mp3_into_chunks(normalized, tmp_path / "chunks", chunk_duration_seconds=2)

    durations = [probe_audio_metadata(chunk).duration_seconds for chunk in chunks]
    parent_duration = probe_audio_metadata(normalized).duration_seconds
    frame_seconds = 1152 / 44100
    covered = sum(durations)

    assert covered <= parent_duration + 0.001
    assert parent_duration - covered <= frame_seconds + 0.001
    rows = [
        line.split(",")
        for line in (tmp_path / "chunks" / "segments.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert abs(float(rows[0][1])) < 0.001
    assert abs(sum(durations) - float(rows[len(chunks) - 1][2])) < 0.001


def test_unreadable_partial_frame_tail_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_transcriber.audio import _without_undecodable_tail

    good = tmp_path / "chunk_0001.mp3"
    tail = tmp_path / "chunk_0002.mp3"
    good.write_bytes(b"good")
    tail.write_bytes(b"tail")

    def fake_probe(path: str | Path) -> audio_module.AudioMetadata:
        if Path(path).name == "chunk_0002.mp3":
            raise AudioChunkingError("invalid frame")
        return audio_module.AudioMetadata(duration_seconds=2.0, bitrate_bps=64_000)

    monkeypatch.setattr(audio_module, "probe_audio_metadata", fake_probe)

    kept = _without_undecodable_tail(
        [good, tail],
        [("chunk_0001.mp3", 0.0, 2.0), ("chunk_0002.mp3", 2.0, 2.026122)],
        sample_rate=44_100,
    )

    assert kept == [good]
    assert not tail.exists()


def test_unreadable_chunk_longer_than_one_frame_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_transcriber.audio import _without_undecodable_tail

    good = tmp_path / "chunk_0001.mp3"
    broken = tmp_path / "chunk_0002.mp3"
    good.write_bytes(b"good")
    broken.write_bytes(b"broken")

    def fake_probe(path: str | Path) -> audio_module.AudioMetadata:
        if Path(path).name == "chunk_0002.mp3":
            raise AudioChunkingError("invalid frame")
        return audio_module.AudioMetadata(duration_seconds=2.0, bitrate_bps=64_000)

    monkeypatch.setattr(audio_module, "probe_audio_metadata", fake_probe)

    with pytest.raises(AudioChunkingError, match="chunk_0002.mp3"):
        _without_undecodable_tail(
            [good, broken],
            [("chunk_0001.mp3", 0.0, 2.0), ("chunk_0002.mp3", 2.0, 4.0)],
            sample_rate=44_100,
        )

    assert broken.exists()
