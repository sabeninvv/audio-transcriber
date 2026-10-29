from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from audio_transcriber.audio import (
    calculate_chunk_duration_seconds,
    ensure_ffmpeg_available,
    normalize_to_mp3,
    probe_audio_metadata,
    split_mp3_into_chunks,
    validate_audio_path,
)
from audio_transcriber.config import (
    SUBTITLE_CHUNK_MAX_DURATION_SECONDS,
    TranscriptionConfig,
    TranscriptionProvider,
    load_config,
)
from audio_transcriber.elevenlabs_client import ElevenLabsTranscriptionClient
from audio_transcriber.exceptions import (
    AudioChunkingError,
    TranscriptionError,
)
from audio_transcriber.gemini_client import GeminiTranscriptionClient
from audio_transcriber.models import (
    OutputFormat,
    TranscriptionResult,
    TranscriptionSegment,
    ensure_subtitles_allowed,
    parse_output_format,
)
from audio_transcriber.openai_client import OpenAITranscriptionClient
from audio_transcriber.writer import (
    generate_markdown_filename,
    generate_subtitle_filename,
    render_subtitles_srt,
    render_transcript_markdown,
)


class TranscriptionClient(Protocol):
    async def transcribe_file(
        self,
        path: Path,
        *,
        prompt_suffix: str | None = None,
    ) -> TranscriptionResult: ...


class TranscriptionService:
    def __init__(
        self,
        client: TranscriptionClient,
        *,
        now_provider: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._client = client
        self._now_provider = now_provider

    async def transcribe_audio(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        config: TranscriptionConfig,
        output_format: OutputFormat | str = OutputFormat.MARKDOWN,
    ) -> Path:
        selected_format = parse_output_format(output_format)
        if selected_format is OutputFormat.SRT:
            ensure_subtitles_allowed(diarization_enabled=config.diarization_enabled)
        source_path = validate_audio_path(input_path)
        ensure_ffmpeg_available()

        with TemporaryDirectory(dir=config.temp_dir) as temp_dir_name:
            working_dir = Path(temp_dir_name)
            normalized_path = await asyncio.to_thread(
                normalize_to_mp3,
                source_path,
                working_dir / "normalized.mp3",
            )
            metadata = await asyncio.to_thread(probe_audio_metadata, normalized_path)
            chunk_max_duration_seconds = config.chunk_max_duration_seconds
            if (
                selected_format is OutputFormat.SRT
                and config.provider is TranscriptionProvider.OPENAI
            ):
                # whisper-1 repeats its previous window on long requests.
                chunk_max_duration_seconds = min(
                    chunk_max_duration_seconds,
                    SUBTITLE_CHUNK_MAX_DURATION_SECONDS,
                )
            chunk_paths = await self._prepare_chunks(
                normalized_path=normalized_path,
                working_dir=working_dir,
                duration_seconds=metadata.duration_seconds,
                bitrate_bps=metadata.bitrate_bps,
                chunk_max_bytes=config.chunk_max_bytes,
                chunk_max_duration_seconds=chunk_max_duration_seconds,
            )

            transcript_chunks: list[TranscriptionResult] = []
            chunk_offsets = await self._build_chunk_offsets(
                chunk_paths=chunk_paths,
                include_offsets=(
                    config.diarization_enabled or selected_format is OutputFormat.SRT
                ),
            )
            prompt_suffix: str | None = None
            for chunk_index, chunk_path in enumerate(chunk_paths):
                try:
                    chunk_result = await self._client.transcribe_file(
                        chunk_path,
                        prompt_suffix=prompt_suffix,
                    )
                except TranscriptionError:
                    raise
                except Exception as error:
                    raise TranscriptionError("Transcription request failed.") from error
                transcript_chunks.append(
                    TranscriptionResult(
                        text=chunk_result.text,
                        segments=_offset_segments(
                            chunk_result.segments,
                            offset_seconds=chunk_offsets[chunk_index],
                        ),
                    )
                )
                prompt_suffix = _tail_context(chunk_result.text, config.chunk_context_chars)

            generated_at = self._now_provider()
            cues = tuple(
                segment
                for chunk in transcript_chunks
                for segment in chunk.segments
            )
            if selected_format is OutputFormat.SRT:
                payload = render_subtitles_srt(cues)
                output_name = generate_subtitle_filename(
                    source_stem=source_path.stem,
                    timestamp=generated_at,
                )
            else:
                payload = render_transcript_markdown(
                    source_name=source_path.name,
                    generated_at=generated_at,
                    chunk_count=len(chunk_paths),
                    original_format=source_path.suffix.lstrip(".").lower(),
                    transcript_text="\n\n".join(chunk.text for chunk in transcript_chunks),
                    speaker_segments=cues or None,
                )
                output_name = generate_markdown_filename(
                    source_stem=source_path.stem,
                    timestamp=generated_at,
                )
            output_directory = Path(output_dir)
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / output_name
            await asyncio.to_thread(output_path.write_text, payload, "utf-8")
            return output_path

    async def _prepare_chunks(
        self,
        *,
        normalized_path: Path,
        working_dir: Path,
        duration_seconds: float,
        bitrate_bps: int,
        chunk_max_bytes: int,
        chunk_max_duration_seconds: int,
    ) -> list[Path]:
        if (
            normalized_path.stat().st_size <= chunk_max_bytes
            and duration_seconds <= chunk_max_duration_seconds
        ):
            return [normalized_path]

        chunk_duration_seconds = calculate_chunk_duration_seconds(
            bitrate_bps=bitrate_bps,
            max_chunk_bytes=chunk_max_bytes,
            max_duration_seconds=chunk_max_duration_seconds,
        )
        chunk_paths = await asyncio.to_thread(
            split_mp3_into_chunks,
            normalized_path,
            working_dir / "chunks",
            chunk_duration_seconds=chunk_duration_seconds,
        )
        return await self._ensure_chunk_size_limit(
            chunk_paths=chunk_paths,
            working_dir=working_dir,
            chunk_max_bytes=chunk_max_bytes,
            chunk_duration_seconds=chunk_duration_seconds,
        )

    async def _ensure_chunk_size_limit(
        self,
        *,
        chunk_paths: list[Path],
        working_dir: Path,
        chunk_max_bytes: int,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        adjusted_chunks: list[Path] = []
        for chunk_path in chunk_paths:
            if chunk_path.stat().st_size <= chunk_max_bytes:
                adjusted_chunks.append(chunk_path)
                continue

            if chunk_duration_seconds <= 1:
                raise AudioChunkingError(
                    f"Chunk {chunk_path.name} still exceeds the maximum upload size."
                )

            next_duration = max(1, chunk_duration_seconds // 2)
            resplit_directory = working_dir / f"{chunk_path.stem}_resplit"
            resplit_chunks = await asyncio.to_thread(
                split_mp3_into_chunks,
                chunk_path,
                resplit_directory,
                chunk_duration_seconds=next_duration,
            )
            adjusted_chunks.extend(
                await self._ensure_chunk_size_limit(
                    chunk_paths=resplit_chunks,
                    working_dir=resplit_directory,
                    chunk_max_bytes=chunk_max_bytes,
                    chunk_duration_seconds=next_duration,
                )
            )

        return adjusted_chunks

    async def _build_chunk_offsets(
        self,
        *,
        chunk_paths: list[Path],
        include_offsets: bool,
    ) -> list[float]:
        if not include_offsets:
            return [0.0] * len(chunk_paths)

        offsets: list[float] = []
        accumulated_seconds = 0.0
        for chunk_path in chunk_paths:
            offsets.append(accumulated_seconds)
            chunk_metadata = await asyncio.to_thread(probe_audio_metadata, chunk_path)
            accumulated_seconds += chunk_metadata.duration_seconds
        return offsets


async def transcribe_audio(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    output_format: OutputFormat | str = OutputFormat.MARKDOWN,
) -> Path:
    selected_format = parse_output_format(output_format)
    config = load_config()
    if selected_format is OutputFormat.SRT:
        ensure_subtitles_allowed(diarization_enabled=config.diarization_enabled)
    timestamps_enabled = selected_format is OutputFormat.SRT
    client: TranscriptionClient
    match config.provider:
        case TranscriptionProvider.OPENAI:
            client = OpenAITranscriptionClient(
                api_key=config.api_key,
                model=config.model,
                prompt=config.prompt,
                languages=config.languages,
                keywords=config.keywords,
                diarization_enabled=config.diarization_enabled,
                diarization_model=config.diarization_model,
                subtitle_model=config.subtitle_model,
                known_speaker_names=config.known_speaker_names,
                known_speaker_references=config.known_speaker_references,
                timestamps_enabled=timestamps_enabled,
            )
        case TranscriptionProvider.ELEVENLABS:
            client = ElevenLabsTranscriptionClient(
                api_key=config.api_key,
                model=config.model,
                languages=config.languages,
                keywords=config.keywords,
                diarization_enabled=config.diarization_enabled,
                timestamps_enabled=timestamps_enabled,
            )
        case TranscriptionProvider.GEMINI:
            if timestamps_enabled:
                # Сегментных таймкодов Gemini не отдаёт.
                raise TranscriptionError("SRT subtitles are not supported by gemini.")
            client = GeminiTranscriptionClient(
                api_key=config.api_key,
                model=config.model,
                prompt=config.prompt,
                languages=config.languages,
                keywords=config.keywords,
            )
    service = TranscriptionService(client)
    return await service.transcribe_audio(
        input_path,
        output_dir,
        config=config,
        output_format=selected_format,
    )


def _tail_context(text: str, max_chars: int) -> str | None:
    if max_chars <= 0:
        return None
    normalized = text.strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def _offset_segments(
    segments: tuple[TranscriptionSegment, ...],
    *,
    offset_seconds: float,
) -> tuple[TranscriptionSegment, ...]:
    if not segments or offset_seconds <= 0:
        return segments
    return tuple(
        TranscriptionSegment(
            start_seconds=segment.start_seconds + offset_seconds,
            end_seconds=segment.end_seconds + offset_seconds,
            speaker=segment.speaker,
            text=segment.text,
        )
        for segment in segments
    )
