import json
import logging
import subprocess
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path
from shutil import which

from audio_transcriber.exceptions import (
    AudioChunkingError,
    FfmpegNotAvailableError,
    UnsupportedAudioFormatError,
)

logger = logging.getLogger(__name__)

DEFAULT_MP3_BITRATE = "64k"
MP3_SAMPLES_PER_FRAME = 1152
SUPPORTED_AUDIO_EXTENSIONS = {
    ".avi",
    ".m4a",
    ".mp3",
    ".mov",
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
    segment_list = output_directory / "segments.csv"
    sample_rate = _audio_sample_rate(source)

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
            "-segment_list",
            str(segment_list),
            "-segment_list_type",
            "csv",
            "-c",
            "copy",
            str(output_pattern),
        ]
    )

    chunk_paths = sorted(output_directory.glob("chunk_*.mp3"))
    if not chunk_paths:
        raise AudioChunkingError("ffmpeg did not produce any chunk files.")
    return _without_undecodable_tail(
        chunk_paths,
        _read_segment_rows(segment_list),
        sample_rate=sample_rate,
    )


def _audio_sample_rate(path: Path) -> int:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        sample_rate = int(json.loads(result.stdout)["streams"][0]["sample_rate"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioChunkingError("Unable to parse ffprobe sample rate.") from error
    if sample_rate <= 0:
        raise AudioChunkingError("Audio sample rate must be greater than zero.")
    return sample_rate


def _read_segment_rows(path: Path) -> list[tuple[str, float, float]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AudioChunkingError("ffmpeg did not write a segment list.") from error

    rows: list[tuple[str, float, float]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 3:
            raise AudioChunkingError(f"Unable to parse ffmpeg segment list line {line_number}.")
        name, raw_start, raw_end = parts
        try:
            start = float(raw_start)
            end = float(raw_end)
        except ValueError as error:
            raise AudioChunkingError(
                f"Unable to parse ffmpeg segment list line {line_number}."
            ) from error
        if not isfinite(start) or not isfinite(end) or end < start:
            raise AudioChunkingError(f"Invalid ffmpeg segment range on line {line_number}.")
        rows.append((Path(name).name, start, end))
    if not rows:
        raise AudioChunkingError("ffmpeg segment list is empty.")
    return rows


def _without_undecodable_tail(
    chunk_paths: list[Path],
    rows: list[tuple[str, float, float]],
    *,
    sample_rate: int,
) -> list[Path]:
    spans = {name: end - start for name, start, end in rows}
    kept: list[Path] = []
    for index, chunk_path in enumerate(chunk_paths):
        try:
            probe_audio_metadata(chunk_path)
        except AudioChunkingError as error:
            span = spans.get(chunk_path.name)
            # ponytail: drop only a final partial MP3 frame; re-mux a longer tail if it must be kept
            span_samples = round(span * sample_rate) if span is not None else None
            is_partial_tail = (
                index == len(chunk_paths) - 1
                and index > 0
                and span_samples is not None
                and span_samples <= MP3_SAMPLES_PER_FRAME
            )
            if not is_partial_tail:
                raise AudioChunkingError(f"Chunk {chunk_path.name} is not readable.") from error
            chunk_path.unlink()
            logger.info(
                "Omitting unreadable trailing MP3 frame %s span_samples=%s.",
                chunk_path.name,
                span_samples,
            )
            continue
        kept.append(chunk_path)
    if not kept:
        raise AudioChunkingError("ffmpeg did not produce any readable chunk files.")
    return kept


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
