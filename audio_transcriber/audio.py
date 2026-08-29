import json
import subprocess
from dataclasses import dataclass
from math import floor
from pathlib import Path
from shutil import which

from audio_transcriber.exceptions import (
    AudioChunkingError,
    FfmpegNotAvailableError,
    UnsupportedAudioFormatError,
)

DEFAULT_MP3_BITRATE = "64k"
SUPPORTED_AUDIO_EXTENSIONS = {
    ".avi",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".wav",
    ".webm",
}


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_seconds: float
    bitrate_bps: int


def is_supported_audio_path(input_path: str | Path) -> bool:
    path = Path(input_path)
    return path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def validate_audio_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not is_supported_audio_path(path):
        supported = ", ".join(ext.lstrip(".") for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format: {path.suffix or '<none>'}. "
            f"Only {supported} are supported."
        )
    return path


def ensure_ffmpeg_available() -> None:
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if which(tool) is None]
    if missing_tools:
        missing = ", ".join(missing_tools)
        raise FfmpegNotAvailableError(
            f"Required audio tools are missing: {missing}. Install ffmpeg and ffprobe."
        )


def calculate_chunk_duration_seconds(
    *,
    bitrate_bps: int,
    max_chunk_bytes: int,
    max_duration_seconds: int | None = None,
) -> int:
    if bitrate_bps <= 0:
        raise AudioChunkingError("Audio bitrate must be greater than zero.")
    if max_chunk_bytes <= 0:
        raise AudioChunkingError("Chunk size must be greater than zero.")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise AudioChunkingError("Chunk duration limit must be greater than zero.")

    duration_seconds = floor((max_chunk_bytes * 8) / bitrate_bps)
    if max_duration_seconds is not None:
        duration_seconds = min(duration_seconds, max_duration_seconds)
    return max(1, duration_seconds)


def probe_audio_metadata(input_path: str | Path) -> AudioMetadata:
    path = Path(input_path)
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
        format_info = payload["format"]
        duration_seconds = float(format_info["duration"])
        bitrate_bps = int(format_info["bit_rate"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioChunkingError("Unable to parse ffprobe metadata output.") from error

    return AudioMetadata(duration_seconds=duration_seconds, bitrate_bps=bitrate_bps)


def normalize_to_mp3(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bitrate: str = DEFAULT_MP3_BITRATE,
) -> Path:
    source = Path(input_path)
    target = Path(output_path)
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-b:a",
            bitrate,
            str(target),
        ]
    )
    return target


def split_mp3_into_chunks(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_duration_seconds: int,
) -> list[Path]:
    if chunk_duration_seconds <= 0:
        raise AudioChunkingError("Chunk duration must be greater than zero.")

    source = Path(input_path)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_pattern = output_directory / "chunk_%04d.mp3"

    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_duration_seconds),
            "-segment_start_number",
            "1",
            "-c",
            "copy",
            str(output_pattern),
        ]
    )

    chunk_paths = sorted(output_directory.glob("chunk_*.mp3"))
    if not chunk_paths:
        raise AudioChunkingError("ffmpeg did not produce any chunk files.")
    return chunk_paths


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        tool_name = command[0] if command else "ffmpeg"
        raise FfmpegNotAvailableError(
            f"Required audio tool is missing: {tool_name}. Install ffmpeg and ffprobe."
        ) from error

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "Unknown error."
        raise AudioChunkingError(stderr)
    return result
