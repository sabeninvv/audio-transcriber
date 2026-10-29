from datetime import datetime, timezone
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import OpenAIError

import audio_transcriber.service as service_module
from audio_transcriber.audio import AudioMetadata, calculate_chunk_duration_seconds
from audio_transcriber.config import (
    DEFAULT_SUBTITLE_MODEL,
    TranscriptionConfig,
    TranscriptionProvider,
)
from audio_transcriber.elevenlabs_client import ElevenLabsTranscriptionClient
from audio_transcriber.gemini_client import GeminiTranscriptionClient

OPENROUTER_TRANSCRIPTION_MODEL = "openai/gpt-transcribe"
OPENROUTER_SUBTITLE_MODEL = "openai/whisper-1"
from audio_transcriber.exceptions import (
    ElevenLabsTranscriptionError,
    OpenAITranscriptionError,
    TranscriptionError,
)
from audio_transcriber.openai_client import (
    OpenAITranscriptionClient,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.service import TranscriptionService
from tests.fakes import (
    FakeChunkTranscriber,
    FakeLoggedOpenAIError,
    FakeOpenAISdkClient,
    FakeTranscriptionService,
    FakeTranscriptionsEndpoint,
)


@pytest.mark.asyncio
async def test_public_transcribe_audio_loads_config_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key", model="gpt-transcribe")
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
            output_format: object,
        ) -> Path:
            captured["service_args"] = (Path(source_path), Path(target_dir), config, output_format)
            return Path(target_dir) / "wrapped.md"

    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "OpenAITranscriptionClient", FakeClient)
    monkeypatch.setattr(service_module, "TranscriptionService", FakeService)

    result = await service_module.transcribe_audio(input_path, output_dir)

    assert captured["service_args"] == (input_path, output_dir, config, "md")
    assert result == output_dir / "wrapped.md"
    assert captured["client_kwargs"] == {
        "api_key": "test-key",
        "model": config.model,
        "prompt": config.prompt,
        "languages": config.languages,
        "keywords": config.keywords,
        "diarization_enabled": config.diarization_enabled,
        "diarization_model": config.diarization_model,
        "subtitle_model": config.subtitle_model,
        "known_speaker_names": config.known_speaker_names,
        "known_speaker_references": config.known_speaker_references,
        "timestamps_enabled": False,
    }


@pytest.mark.asyncio
async def test_public_transcribe_audio_uses_whisper_timestamps_for_srt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SRT -> клиент получает модель субтитров из конфига и включённые таймкоды."""
    input_path = tmp_path / "meeting.wav"
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(
        api_key="test-key",
        model="gpt-transcribe",
        subtitle_model=OPENROUTER_SUBTITLE_MODEL,
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

    class FakeService:
        def __init__(self, client: object) -> None:
            del client

        async def transcribe_audio(
            self,
            source_path: str | Path,
            target_dir: str | Path,
            *,
            config: TranscriptionConfig,
            output_format: object,
        ) -> Path:
            del source_path, config
            captured["output_format"] = output_format
            return Path(target_dir) / "wrapped.srt"

    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "OpenAITranscriptionClient", FakeClient)
    monkeypatch.setattr(service_module, "TranscriptionService", FakeService)

    result = await service_module.transcribe_audio(input_path, output_dir, output_format="srt")

    assert result == output_dir / "wrapped.srt"
    assert captured["output_format"] == "srt"
    client_kwargs = captured["client_kwargs"]
    assert isinstance(client_kwargs, dict)
    # Модель субтитров берётся из конфига отдельным аргументом.
    assert client_kwargs["subtitle_model"] == OPENROUTER_SUBTITLE_MODEL
    assert client_kwargs["timestamps_enabled"] is True
    assert client_kwargs["diarization_enabled"] is False


@pytest.mark.asyncio
async def test_public_transcribe_audio_builds_elevenlabs_client_for_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Провайдер elevenlabs -> сервис получает клиента ElevenLabs."""
    config = TranscriptionConfig(api_key="test-key", provider=TranscriptionProvider.ELEVENLABS)
    fake_service = FakeTranscriptionService()
    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "TranscriptionService", fake_service)

    await service_module.transcribe_audio("meeting.wav", "out")

    assert len(fake_service.clients) == 1
    assert isinstance(fake_service.clients[0], ElevenLabsTranscriptionClient)


@pytest.mark.asyncio
async def test_public_transcribe_audio_builds_gemini_client_for_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Провайдер gemini -> сервис получает клиента Gemini."""
    config = TranscriptionConfig(api_key="test-key", provider=TranscriptionProvider.GEMINI)
    fake_service = FakeTranscriptionService()
    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "TranscriptionService", fake_service)

    await service_module.transcribe_audio("meeting.wav", "out")

    assert len(fake_service.clients) == 1
    assert isinstance(fake_service.clients[0], GeminiTranscriptionClient)


@pytest.mark.asyncio
async def test_public_transcribe_audio_rejects_srt_for_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini + SRT -> ошибка до сервиса: сегментных таймкодов у Gemini нет."""
    config = TranscriptionConfig(api_key="test-key", provider=TranscriptionProvider.GEMINI)
    fake_service = FakeTranscriptionService()
    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "TranscriptionService", fake_service)

    with pytest.raises(TranscriptionError, match="gemini"):
        await service_module.transcribe_audio("meeting.wav", "out", output_format="srt")

    # Медиа не обрабатывалось: сервис не создан.
    assert fake_service.clients == []


@pytest.mark.asyncio
async def test_service_keeps_elevenlabs_error_from_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка ElevenLabs из клиента -> пробрасывается как есть, без обёртки OpenAI."""
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    client_error = ElevenLabsTranscriptionError("ElevenLabs transcription request failed.")
    config = TranscriptionConfig(api_key="test-key", provider=TranscriptionProvider.ELEVENLABS)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        del input_audio, bitrate
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)
    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(
        service_module,
        "probe_audio_metadata",
        lambda path: AudioMetadata(duration_seconds=10.0, bitrate_bps=64_000),
    )

    service = TranscriptionService(FakeChunkTranscriber(error=client_error))
    with pytest.raises(ElevenLabsTranscriptionError) as error_info:
        await service.transcribe_audio(input_path, tmp_path / "out", config=config)

    assert error_info.value is client_error


@pytest.mark.asyncio
async def test_public_transcribe_audio_rejects_srt_with_diarization_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TranscriptionConfig(api_key="test-key", diarization_enabled=True)

    def fail_client(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("client constructed")

    monkeypatch.setattr(service_module, "load_config", lambda: config)
    monkeypatch.setattr(service_module, "OpenAITranscriptionClient", fail_client)

    with pytest.raises(TranscriptionError, match="diarization"):
        await service_module.transcribe_audio("meeting.wav", "out", output_format="srt")


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
async def test_openai_transcription_client_sends_hints_for_provider_prefixed_model(
    tmp_path: Path,
) -> None:
    """openai/gpt-transcribe (OpenRouter) -> languages и keywords идут через extra_body."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(response=SimpleNamespace(text="hello world"))
    client = OpenAITranscriptionClient(
        api_key="test-key",
        model=OPENROUTER_TRANSCRIPTION_MODEL,
        languages=("ru", "en"),
        keywords=("scheduler",),
        client=FakeOpenAISdkClient(endpoint),
    )

    await client.transcribe_file(chunk_path)

    call = endpoint.calls[0]
    # Модель уходит с префиксом провайдера, подсказки не теряются на legacy-ветке.
    assert call["model"] == OPENROUTER_TRANSCRIPTION_MODEL
    assert call["extra_body"] == {"languages": ["ru", "en"], "keywords": ["scheduler"]}
    assert "language" not in call


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
async def test_openai_transcription_client_requests_segment_timestamps(
    tmp_path: Path,
) -> None:
    """Таймкоды -> verbose_json на модели субтитров без промпта и подсказок."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "привет hello",
            "segments": [
                {"start": 0.0, "end": 1.25, "text": " привет "},
                {"start": 1.25, "end": 2.5, "text": "hello"},
            ],
        }
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        subtitle_model=OPENROUTER_SUBTITLE_MODEL,
        languages=("ru", "en"),
        keywords=("scheduler",),
        prompt="Main context",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    result = await client.transcribe_file(chunk_path, prompt_suffix="хвост")

    call = endpoint.calls[0]
    # Запрос уходит на модель субтитров, а не на основную модель транскрибации.
    assert call["model"] == OPENROUTER_SUBTITLE_MODEL
    assert call["response_format"] == "verbose_json"
    assert call["timestamp_granularities"] == ["segment"]
    assert "chunking_strategy" not in call
    assert "extra_body" not in call
    assert "language" not in call
    assert "keywords" not in call
    assert "prompt" not in call
    assert result.text == "привет hello"
    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.25, speaker="", text="привет"),
        TranscriptionSegment(start_seconds=1.25, end_seconds=2.5, speaker="", text="hello"),
    )


@pytest.mark.asyncio
async def test_openai_transcription_client_defaults_timestamps_to_whisper(tmp_path: Path) -> None:
    """Таймкоды без явных моделей -> запрос на whisper-1, а не на gpt-transcribe."""
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={"text": "hello", "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    await client.transcribe_file(chunk_path)

    # Клиент сам гарантирует whisper-модель для SRT.
    assert endpoint.calls[0]["model"] == DEFAULT_SUBTITLE_MODEL


@pytest.mark.asyncio
async def test_openai_transcription_client_reads_timestamp_objects_and_one_language(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response=SimpleNamespace(
            text="hello",
            segments=[SimpleNamespace(start=0, end=1, text=" hello ")],
        )
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        languages=("ru",),
        keywords=("scheduler",),
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    result = await client.transcribe_file(chunk_path)

    assert endpoint.calls[0]["language"] == "ru"
    assert "extra_body" not in endpoint.calls[0]
    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.0, speaker="", text="hello"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "segments",
    [
        None,
        [],
        [{"start": 0.0, "end": 1.0}],
        [{"start": "0", "end": 1.0, "text": "hello"}],
        [{"start": True, "end": 1.0, "text": "hello"}],
        [{"start": float("nan"), "end": 1.0, "text": "hello"}],
        [{"start": float("inf"), "end": 2.0, "text": "hello"}],
        [{"start": -0.01, "end": 1.0, "text": "hello"}],
        [{"start": 1.0, "end": 1.0, "text": "hello"}],
        [{"start": 1.5, "end": 1.0, "text": "hello"}],
        [{"start": 0.0, "end": 1.0, "text": "   "}],
        [
            {"start": 1.0, "end": 2.0, "text": "hello"},
            {"start": 0.4, "end": 1.2, "text": "back"},
        ],
    ],
)
async def test_openai_transcription_client_rejects_unusable_timestamp_segments(
    tmp_path: Path,
    segments: object,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(response={"text": "hello", "segments": segments})
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    with pytest.raises(OpenAITranscriptionError):
        await client.transcribe_file(chunk_path)


@pytest.mark.asyncio
async def test_openai_transcription_client_skips_segments_with_no_duration(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 1.0, "text": ""},
                {"start": 1.5, "end": 1.2, "text": "nope"},
                {"start": 2.0, "end": 3.0, "text": "world"},
            ],
        }
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    result = await client.transcribe_file(chunk_path)

    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.0, speaker="", text="hello"),
        TranscriptionSegment(start_seconds=2.0, end_seconds=3.0, speaker="", text="world"),
    )


@pytest.mark.asyncio
async def test_openai_transcription_client_skips_blank_segments_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "   "},
                {"start": 2.0, "end": 3.0, "text": None},
                {"start": 3.0, "end": 4.0, "text": "world"},
            ],
        }
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    result = await client.transcribe_file(chunk_path)

    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.0, speaker="", text="hello"),
        TranscriptionSegment(start_seconds=3.0, end_seconds=4.0, speaker="", text="world"),
    )


@pytest.mark.asyncio
async def test_openai_transcription_client_omits_repeated_whisper_segments(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "hello",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "compression_ratio": 1.1},
                {"start": 1.0, "end": 2.0, "text": "hello", "compression_ratio": 2.8},
            ],
        }
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    result = await client.transcribe_file(chunk_path)

    assert result.segments == (
        TranscriptionSegment(start_seconds=0.0, end_seconds=1.0, speaker="", text="hello"),
    )


@pytest.mark.asyncio
async def test_openai_transcription_client_rejects_only_repeated_whisper_segments(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk.mp3"
    chunk_path.write_bytes(b"audio")
    endpoint = FakeTranscriptionsEndpoint(
        response={
            "text": "hello",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "compression_ratio": 3.0},
            ],
        }
    )
    client = OpenAITranscriptionClient(
        api_key="test-key",
        timestamps_enabled=True,
        client=FakeOpenAISdkClient(endpoint),
    )

    with pytest.raises(OpenAITranscriptionError, match="segments"):
        await client.transcribe_file(chunk_path)


def test_openai_transcription_client_rejects_timestamps_with_diarization() -> None:
    with pytest.raises(TranscriptionError, match="diarization"):
        OpenAITranscriptionClient(
            api_key="test-key",
            diarization_enabled=True,
            timestamps_enabled=True,
        )


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
    config = TranscriptionConfig(api_key="test-key")
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
    config = TranscriptionConfig(api_key="test-key")
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
    config = TranscriptionConfig(api_key="test-key", chunk_max_bytes=10)
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
        api_key="test-key",
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
        api_key="test-key",
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
    config = TranscriptionConfig(api_key="test-key", chunk_max_bytes=10)
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
async def test_service_maps_client_failures_to_transcription_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Непредвиденный сбой клиента -> нейтральная TranscriptionError без привязки к провайдеру."""
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key")
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
    with pytest.raises(TranscriptionError, match="Transcription request failed"):
        await service.transcribe_audio(input_path, output_dir, config=config)


@pytest.mark.asyncio
async def test_service_cleans_up_temporary_files_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key")
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
    with pytest.raises(TranscriptionError):
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
        api_key="test-key",
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


@pytest.mark.asyncio
async def test_service_writes_srt_on_one_timeline_across_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key", chunk_max_bytes=10)
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(
                text="hello",
                segments=(TranscriptionSegment(0.0, 1.0, "", "hello"),),
            ),
            TranscriptionResult(
                text="мир",
                segments=(TranscriptionSegment(0.5, 1.25, "", "мир"),),
            ),
        ]
    )
    generated_at = datetime(2026, 4, 7, 12, 30, 11, tzinfo=timezone.utc)

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

    service = TranscriptionService(fake_client, now_provider=lambda: generated_at)
    result = await service.transcribe_audio(
        input_path,
        output_dir,
        config=config,
        output_format="srt",
    )

    assert result == output_dir / "meeting_2026-04-07T12-30-11.srt"
    assert result.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello\n"
        "\n"
        "2\n"
        "00:00:05,500 --> 00:00:06,250\n"
        "мир\n"
    )


@pytest.mark.asyncio
async def test_service_offsets_srt_after_resplitting_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "meeting.wav"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key", chunk_max_bytes=10)
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(text="a", segments=(TranscriptionSegment(0.2, 0.5, "", "a"),)),
            TranscriptionResult(text="b", segments=(TranscriptionSegment(0.2, 0.5, "", "b"),)),
            TranscriptionResult(text="c", segments=(TranscriptionSegment(0.2, 0.5, "", "c"),)),
        ]
    )

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        target = Path(output_audio)
        target.write_bytes(b"0123456789abcdef")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        probed = Path(path)
        if probed.name == "normalized.mp3":
            return AudioMetadata(duration_seconds=120.0, bitrate_bps=32)
        if probed.parent.name.endswith("_resplit") and probed.name == "chunk_0001.mp3":
            return AudioMetadata(duration_seconds=1.0, bitrate_bps=32)
        if probed.parent.name.endswith("_resplit"):
            return AudioMetadata(duration_seconds=2.0, bitrate_bps=32)
        return AudioMetadata(duration_seconds=3.0, bitrate_bps=32)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        del chunk_duration_seconds
        source = Path(input_audio)
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

    service = TranscriptionService(fake_client)
    result = await service.transcribe_audio(
        input_path,
        output_dir,
        config=config,
        output_format="srt",
    )
    content = result.read_text(encoding="utf-8")

    assert "00:00:00,200 --> 00:00:00,500\na\n" in content
    assert "00:00:01,200 --> 00:00:01,500\nb\n" in content
    assert "00:00:03,200 --> 00:00:03,500\nc\n" in content


@pytest.mark.parametrize(
    ("provider", "expected_split_durations"),
    [(TranscriptionProvider.OPENAI, [30]), (TranscriptionProvider.ELEVENLABS, [])],
)
@pytest.mark.asyncio
async def test_service_splits_subtitles_into_whisper_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: TranscriptionProvider,
    expected_split_durations: list[int],
) -> None:
    """SRT -> 30-секундные окна только для whisper; ElevenLabs шлёт файл без нарезки."""
    input_path = tmp_path / "lesson.mp3"
    input_path.write_bytes(b"source")
    output_dir = tmp_path / "out"
    config = TranscriptionConfig(api_key="test-key", provider=provider)
    fake_client = FakeChunkTranscriber(
        responses=[
            TranscriptionResult(text="one", segments=(TranscriptionSegment(0.0, 1.0, "", "one"),)),
            TranscriptionResult(text="two", segments=(TranscriptionSegment(0.0, 1.0, "", "two"),)),
        ]
    )
    split_durations: list[int] = []

    monkeypatch.setattr(service_module, "validate_audio_path", lambda path: Path(path))
    monkeypatch.setattr(service_module, "ensure_ffmpeg_available", lambda: None)

    def fake_normalize(input_audio: str | Path, output_audio: str | Path, *, bitrate: str = "64k") -> Path:
        del input_audio, bitrate
        target = Path(output_audio)
        target.write_bytes(b"normalized")
        return target

    def fake_probe(path: str | Path) -> AudioMetadata:
        if Path(path).name == "normalized.mp3":
            return AudioMetadata(duration_seconds=120.0, bitrate_bps=64_000)
        return AudioMetadata(duration_seconds=30.0, bitrate_bps=64_000)

    def fake_split(
        input_audio: str | Path,
        chunk_dir: str | Path,
        *,
        chunk_duration_seconds: int,
    ) -> list[Path]:
        del input_audio
        split_durations.append(chunk_duration_seconds)
        output_directory = Path(chunk_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        first = output_directory / "chunk_0001.mp3"
        second = output_directory / "chunk_0002.mp3"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        return [first, second]

    monkeypatch.setattr(service_module, "normalize_to_mp3", fake_normalize)
    monkeypatch.setattr(service_module, "probe_audio_metadata", fake_probe)
    monkeypatch.setattr(service_module, "split_mp3_into_chunks", fake_split)

    service = TranscriptionService(fake_client)
    await service.transcribe_audio(input_path, output_dir, config=config, output_format="srt")

    assert split_durations == expected_split_durations


@pytest.mark.asyncio
async def test_service_rejects_srt_with_diarization_before_media_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_normalize(*args: object, **kwargs: object) -> Path:
        raise AssertionError("media processing started")

    monkeypatch.setattr(service_module, "normalize_to_mp3", fail_normalize)
    service = TranscriptionService(FakeChunkTranscriber())

    with pytest.raises(TranscriptionError, match="diarization"):
        await service.transcribe_audio(
            tmp_path / "missing.wav",
            tmp_path / "out",
            config=TranscriptionConfig(api_key="test-key", diarization_enabled=True),
            output_format="srt",
        )


@pytest.mark.asyncio
async def test_service_rejects_unknown_output_format_before_media_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_normalize(*args: object, **kwargs: object) -> Path:
        raise AssertionError("media processing started")

    monkeypatch.setattr(service_module, "normalize_to_mp3", fail_normalize)
    service = TranscriptionService(FakeChunkTranscriber())

    with pytest.raises(TranscriptionError, match="output format"):
        await service.transcribe_audio(
            tmp_path / "missing.wav",
            tmp_path / "out",
            config=TranscriptionConfig(api_key="test-key"),
            output_format="vtt",
        )
