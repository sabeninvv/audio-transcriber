from datetime import datetime, timezone
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import OpenAIError

import audio_transcriber.service as service_module
from audio_transcriber.audio import AudioMetadata, calculate_chunk_duration_seconds
from audio_transcriber.config import TranscriptionConfig
from audio_transcriber.exceptions import OpenAITranscriptionError
from audio_transcriber.openai_client import (
    OpenAITranscriptionClient,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.service import TranscriptionService


class FakeTranscriptionsEndpoint:
    def __init__(self, *, response: object | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response
        self._error = error

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeOpenAISdkClient:
    def __init__(self, endpoint: FakeTranscriptionsEndpoint) -> None:
        self.audio = SimpleNamespace(transcriptions=endpoint)


class FakeChunkTranscriber:
    def __init__(
        self,
        *,
        responses: list[TranscriptionResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[Path, str | None]] = []
        self._responses = list(responses or [])
        self._error = error

    async def transcribe_file(
        self,
        path: Path,
        *,
        prompt_suffix: str | None = None,
    ) -> TranscriptionResult:
        self.calls.append((path, prompt_suffix))
        if self._error is not None:
            raise self._error
        return self._responses.pop(0)


class FakeLoggedOpenAIError(OpenAIError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400
        self.body = {
            "error": {
                "message": "audio duration exceeds limit",
                "code": "invalid_value",
            }
        }


@pytest.mark.asyncio
async def test_public_transcribe_audio_loads_config_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key", model="gpt-transcribe")
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

    class FakeService:
        def __init__(self, client: object) -> None:
            captured["client"] = client

        async def transcribe_audio(
            self,
            source_path: str | Path,
            target_dir: str | Path,
            *,
            config: TranscriptionConfig,
        ) -> Path:
            captured["service_args"] = (Path(source_path), Path(target_dir), config)
            return Path(target_dir) / "wrapped.md"

    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "OpenAITranscriptionClient", FakeClient)
    monkeypatch.setattr(service_module, "TranscriptionService", FakeService)

    result = await service_module.transcribe_audio(input_path, output_dir)

    assert captured["service_args"] == (input_path, output_dir, config)
    assert result == output_dir / "wrapped.md"
    assert captured["client_kwargs"] == {
        "api_key": "test-key",
        "model": config.model,
        "prompt": config.prompt,
        "languages": config.languages,
        "keywords": config.keywords,
        "diarization_enabled": config.diarization_enabled,
        "diarization_model": config.diarization_model,
        "known_speaker_names": config.known_speaker_names,
        "known_speaker_references": config.known_speaker_references,
    }


@pytest.mark.asyncio
async def test_openai_transcription_client_returns_response_text(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(response=SimpleNamespace(text="hello world"))
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(api_key="test-key", client=sdk_client)

    result = await client.transcribe_file(chunk_path)

    assert result == TranscriptionResult(text="hello world")
    assert endpoint.calls[0]["model"] == "gpt-transcribe"
    assert endpoint.calls[0]["response_format"] == "json"
    assert endpoint.calls[0]["chunking_strategy"] == "auto"
    assert endpoint.calls[0]["extra_body"]["languages"] == ["ru", "en"]
    assert "keywords" in endpoint.calls[0]["extra_body"]


@pytest.mark.asyncio
async def test_openai_transcription_client_uses_single_language_hint_for_legacy_model(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(response=SimpleNamespace(text="hello world"))
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(
        api_key="test-key",
        model="gpt-4o-transcribe",
        languages=("ru", "en"),
        keywords=("scheduler",),
        client=sdk_client,
    )

    await client.transcribe_file(chunk_path)

    assert endpoint.calls[0]["model"] == "gpt-4o-transcribe"
    assert endpoint.calls[0]["language"] == "ru"
    assert "extra_body" not in endpoint.calls[0]


@pytest.mark.asyncio
async def test_openai_transcription_client_includes_prompt_suffix(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(response=SimpleNamespace(text="hello world"))
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(
        api_key="test-key",
        prompt="Main context",
        client=sdk_client,
    )

    await client.transcribe_file(chunk_path, prompt_suffix="tail text")

    prompt = endpoint.calls[0]["prompt"]
    assert isinstance(prompt, str)
    assert "Main context" in prompt
    assert "tail text" in prompt


@pytest.mark.asyncio
async def test_openai_transcription_client_maps_diarized_segments(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "Agent: hello",
            "segments": [
                {"start": 0.0, "end": 1.8, "speaker": "agent", "text": "hello"},
                {"start": 2.0, "end": 3.1, "speaker": "A", "text": "hi"},
            ],
        }
    )
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(
        api_key="test-key",
        diarization_enabled=True,
        known_speaker_names=("agent",),
        known_speaker_references=("data:audio/wav;base64,AAAA",),
        client=sdk_client,
    )

    result = await client.transcribe_file(chunk_path)

    assert result.text == "Agent: hello"
    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.8, speaker="agent", text="hello"),
        TranscriptionSegment(start_seconds=2.0, end_seconds=3.1, speaker="A", text="hi"),
    )
    assert endpoint.calls[0]["model"] == "gpt-4o-transcribe-diarize"
    assert endpoint.calls[0]["response_format"] == "diarized_json"
    assert endpoint.calls[0]["known_speaker_names"] == ["agent"]
    assert endpoint.calls[0]["known_speaker_references"] == ["data:audio/wav;base64,AAAA"]


@pytest.mark.asyncio
async def test_openai_transcription_client_wraps_sdk_errors(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(error=OpenAIError("boom"))
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(api_key="test-key", client=sdk_client)

    with pytest.raises(OpenAITranscriptionError, match="OpenAI transcription request failed"):
        await client.transcribe_file(chunk_path)


@pytest.mark.asyncio
async def test_openai_transcription_client_logs_api_error_details(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(error=FakeLoggedOpenAIError("boom"))
    sdk_client = FakeOpenAISdkClient(endpoint)
    client = OpenAITranscriptionClient(api_key="test-key", client=sdk_client)

    caplog.set_level(logging.ERROR, logger="audio_transcriber.openai_client")

    with pytest.raises(OpenAITranscriptionError, match="OpenAI transcription request failed"):
        await client.transcribe_file(chunk_path)

    assert "OpenAI transcription request failed" in caplog.text
    assert "chunk.mp3" in caplog.text
    assert "gpt-transcribe" in caplog.text
    assert "400" in caplog.text
    assert "audio duration exceeds limit" in caplog.text


@pytest.mark.asyncio
async def test_service_transcribes_single_file_and_writes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key")
    fake_client = FakeChunkTranscriber(responses=[TranscriptionResult(text="Single transcript")])
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)
    probed_paths: list[Path] = []

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        probed_paths.append(Path(path))
        return AudioMetadata(duration_seconds=12.0, bitrate_bps=64_000)

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "single.md",
    )

    service = TranscriptionService(fake_client, now_provider=lambda: generated_at)
    result = await service.transcribe_audio(input_path, output_dir, config=config)

    assert len(probed_paths) == 1
    assert probed_paths[0].name == "normalized.mp3"
    assert result == output_dir / "single.md"
    assert result.read_text(encoding="utf-8") == (
        "# Audio Transcription\n\n"
        "- Source file: meeting.wav\n"
        "- Generated at: 2026-04-07T12:30:11+00:00\n"
        "- Chunks: 1\n"
        "- Original format: wav\n"
        "- Working format: mp3\n\n"
        "## Transcript\n\n"
        "Single transcript\n"
    )
    assert [path.name for path, _ in fake_client.calls] == ["normalized.mp3"]
    assert fake_client.calls[0][1] is None


@pytest.mark.asyncio
async def test_service_names_output_file_using_source_stem_and_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key")
    fake_client = FakeChunkTranscriber(responses=[TranscriptionResult(text="Single transcript")])
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(
        input_audio: str | Path,
        output_audio: str | Path,
        *,
        bitrate: str = "64k",
    ) -> Path:
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(
        service_module,
        "probe_audio_metadata",
        lambda path: AudioMetadata(duration_seconds=12.0, bitrate_bps=64_000),
    )

    service = TranscriptionService(fake_client, now_provider=lambda: generated_at)
    result = await service.transcribe_audio(input_path, output_dir, config=config)

    assert result == output_dir / "meeting_2026-04-07T12-30-11.md"


@pytest.mark.asyncio
async def test_service_transcribes_multiple_chunks_in_order_and_passes_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key", chunk_max_bytes=10)
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(text="First chunk"),
            TranscriptionResult(text="Second chunk"),
        ]
    )
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)
    split_durations: list[int] = []

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"0123456789abcdef")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        return AudioMetadata(duration_seconds=120.0, bitrate_bps=64_000)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        split_durations.append(chunk_duration_seconds)
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        return [first, second]

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "multi.md",
    )

    service = TranscriptionService(fake_client, now_provider=lambda: generated_at)
    result = await service.transcribe_audio(input_path, output_dir, config=config)
    content = result.read_text(encoding="utf-8")

    assert split_durations == [
        calculate_chunk_duration_seconds(
            bitrate_bps=64_000,
            max_chunk_bytes=config.chunk_max_bytes,
        )
    ]
    assert content.index("First chunk") < content.index("Second chunk")
    assert [path.name for path, _ in fake_client.calls] == ["chunk_0001.mp3", "chunk_0002.mp3"]
    assert fake_client.calls[0][1] is None
    assert fake_client.calls[1][1] == "First chunk"


@pytest.mark.asyncio
async def test_service_limits_chunk_context_before_passing_to_next_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(
        openai_api_key="test-key",
        chunk_max_bytes=10,
        chunk_context_chars=5,
    )
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(text="123456789"),
            TranscriptionResult(text="ok"),
        ]
    )

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(
        input_audio: str | Path,
        output_audio: str | Path,
        *,
        bitrate: str = "64k",
    ) -> Path:
        del input_audio, bitrate
        target = Path(output_audio)
        target.write_bytes(b"0123456789abcdef")
        return target

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(
        service_module,
        "probe_audio_metadata",
        lambda path: AudioMetadata(duration_seconds=120.0, bitrate_bps=64_000),
    )
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "context.md",
    )

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        del input_audio, chunk_duration_seconds
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        return [first, second]

    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)

    service = TranscriptionService(fake_client)
    await service.transcribe_audio(input_path, output_dir, config=config)

    assert fake_client.calls[1][1] == "56789"


@pytest.mark.asyncio
async def test_service_splits_when_duration_exceeds_limit_even_if_size_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.mp3"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(
        openai_api_key="test-key",
        chunk_max_bytes=24 * 1024 * 1024,
        chunk_max_duration_seconds=1_300,
    )
    fake_client = FakeChunkTranscriber(
        responses=[TranscriptionResult(text="First chunk"), TranscriptionResult(text="Second chunk")]
    )
    split_durations: list[int] = []

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        return AudioMetadata(duration_seconds=2_156.784, bitrate_bps=64_000)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        split_durations.append(chunk_duration_seconds)
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        return [first, second]

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "duration_only.md",
    )

    service = TranscriptionService(fake_client)
    result = await service.transcribe_audio(input_path, output_dir, config=config)
    content = result.read_text(encoding="utf-8")

    assert split_durations == [1_300]
    assert content.index("First chunk") < content.index("Second chunk")
    assert [path.name for path, _ in fake_client.calls] == ["chunk_0001.mp3", "chunk_0002.mp3"]


@pytest.mark.asyncio
async def test_service_re_splits_oversized_chunks_until_they_fit_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key", chunk_max_bytes=10)
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(text="First split"),
            TranscriptionResult(text="Second split"),
            TranscriptionResult(text="Final chunk"),
        ]
    )
    split_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"0123456789abcdef")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        return AudioMetadata(duration_seconds=120.0, bitrate_bps=32)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        source = Path(input_audio)
        split_calls.append((source.name, chunk_duration_seconds))
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        if source.name == "normalized.mp3":
            oversized = output_directory / "chunk_0001.mp3"
            normal = output_directory / "chunk_0002.mp3"
            oversized.write_bytes(b"0123456789ab")
            normal.write_bytes(b"12345678")
            return [oversized, normal]

        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"123456")
        second.write_bytes(b"abcdef")
        return [first, second]

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "resplit.md",
    )

    service = TranscriptionService(fake_client)
    result = await service.transcribe_audio(input_path, output_dir, config=config)
    content = result.read_text(encoding="utf-8")

    assert split_calls == [("normalized.mp3", 2), ("chunk_0001.mp3", 1)]
    assert content.index("First split") < content.index("Second split") < content.index("Final chunk")


@pytest.mark.asyncio
async def test_service_maps_client_failures_to_openai_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key")
    fake_client = FakeChunkTranscriber(error=RuntimeError("boom"))

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(
        service_module,
        "probe_audio_metadata",
        lambda path: AudioMetadata(duration_seconds=10.0, bitrate_bps=64_000),
    )

    service = TranscriptionService(fake_client)
    with pytest.raises(OpenAITranscriptionError, match="OpenAI transcription request failed"):
        await service.transcribe_audio(input_path, output_dir, config=config)


@pytest.mark.asyncio
async def test_service_cleans_up_temporary_files_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(openai_api_key="test-key")
    fake_client = FakeChunkTranscriber(error=RuntimeError("boom"))
    working_directory: dict[str, Path] = {}

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        working_directory["path"] = target.parent
        target.write_bytes(b"normalized")
        return target

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(
        service_module,
        "probe_audio_metadata",
        lambda path: AudioMetadata(duration_seconds=10.0, bitrate_bps=64_000),
    )

    service = TranscriptionService(fake_client)
    with pytest.raises(OpenAITranscriptionError):
        await service.transcribe_audio(input_path, output_dir, config=config)

    assert "path" in working_directory
    assert not working_directory["path"].exists()


@pytest.mark.asyncio
async def test_service_offsets_diarized_segments_and_renders_speaker_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(
        openai_api_key="test-key",
        chunk_max_bytes=10,
        diarization_enabled=True,
    )
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(
                text="A says hello",
                segments=(TranscriptionSegment(0.0, 1.0, "A", "hello"),),
            ),
            TranscriptionResult(
                text="B replies",
                segments=(TranscriptionSegment(1.0, 2.0, "B", "reply"),),
            ),
        ]
    )

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"0123456789abcdef")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        name = Path(path).name
        if name == "normalized.mp3":
            return AudioMetadata(duration_seconds=120.0, bitrate_bps=64_000)
        if name == "chunk_0001.mp3":
            return AudioMetadata(duration_seconds=5.0, bitrate_bps=64_000)
        return AudioMetadata(duration_seconds=6.0, bitrate_bps=64_000)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        del input_audio, chunk_duration_seconds
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        return [first, second]

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)
    monkeypatch.setattr(
        service_module,
        "generate_markdown_filename",
        lambda *, source_stem, timestamp: "diarized.md",
    )

    service = TranscriptionService(fake_client)
    result = await service.transcribe_audio(input_path, output_dir, config=config)
    content = result.read_text(encoding="utf-8")

    assert "## Speaker Transcript" in content
    assert "[00:00:00] Speaker A: hello" in content
    assert "[00:00:06] Speaker B: reply" in content
