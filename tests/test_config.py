import pytest

from audio_transcriber.exceptions import TranscriptionError
from audio_transcriber.config import (
    DEFAULT_CHUNK_MAX_BYTES,
    DEFAULT_CHUNK_CONTEXT_CHARS,
    DEFAULT_DIARIZATION_ENABLED,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_SUBTITLE_MODEL,
    DEFAULT_TRANSCRIPTION_LANGUAGES,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSCRIPTION_PROMPT,
    OPENAI_ONLY_ENV_NAMES,
    TranscriptionProvider,
    load_config,
)

ELEVENLABS_KEY_ENV = "API_ELEVENLABS_API_KEY"
ELEVENLABS_KEY = "elevenlabs-key"


def test_load_config_reads_openai_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Провайдер не задан -> OpenAI с ключом из OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TRANSCRIPTION_PROVIDER", raising=False)

    config = load_config()

    assert config.api_key == "test-key"
    assert config.provider is TranscriptionProvider.OPENAI


def test_load_config_reads_elevenlabs_key_and_default_model() -> None:
    """Провайдер elevenlabs -> ключ из своей переменной и модель scribe_v2 по умолчанию."""
    config = load_config({"TRANSCRIPTION_PROVIDER": " ElevenLabs ", ELEVENLABS_KEY_ENV: ELEVENLABS_KEY})

    assert config.provider is TranscriptionProvider.ELEVENLABS
    assert config.api_key == ELEVENLABS_KEY
    assert config.model == DEFAULT_ELEVENLABS_MODEL == "scribe_v2"
    # Дефолтный глоссарий не уходит в платные keyterms.
    assert config.keywords == ()


def test_load_config_keeps_explicit_keywords_for_elevenlabs() -> None:
    """Явный TRANSCRIPTION_KEYWORDS у ElevenLabs -> попадает в конфиг."""
    config = load_config(
        {
            "TRANSCRIPTION_PROVIDER": "elevenlabs",
            ELEVENLABS_KEY_ENV: ELEVENLABS_KEY,
            "TRANSCRIPTION_KEYWORDS": "ReelsMaker, ContentLoader",
        }
    )

    assert config.keywords == ("ReelsMaker", "ContentLoader")


def test_load_config_requires_elevenlabs_key_even_with_openai_key() -> None:
    """ElevenLabs без своего ключа -> ошибка, ключ OpenAI не подставляется."""
    with pytest.raises(TranscriptionError, match=ELEVENLABS_KEY_ENV):
        load_config({"TRANSCRIPTION_PROVIDER": "elevenlabs", "OPENAI_API_KEY": "test-key"})


def test_load_config_rejects_unknown_provider() -> None:
    """Неизвестный провайдер -> ошибка со списком допустимых значений."""
    with pytest.raises(TranscriptionError, match="openai, elevenlabs"):
        load_config({"TRANSCRIPTION_PROVIDER": "gemini", "OPENAI_API_KEY": "test-key"})


@pytest.mark.parametrize("env_name", OPENAI_ONLY_ENV_NAMES)
def test_load_config_rejects_openai_only_settings_for_elevenlabs(env_name: str) -> None:
    """Явно заданная OpenAI-настройка у ElevenLabs -> ошибка вместо молчаливого игнора."""
    with pytest.raises(TranscriptionError, match=env_name):
        load_config(
            {
                "TRANSCRIPTION_PROVIDER": "elevenlabs",
                ELEVENLABS_KEY_ENV: ELEVENLABS_KEY,
                env_name: "value",
            }
        )


def test_load_config_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустое окружение кроме ключа -> все значения по умолчанию."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = load_config()

    assert config.model == DEFAULT_TRANSCRIPTION_MODEL
    # SRT по умолчанию идёт на whisper-1 OpenAI.
    assert config.subtitle_model == DEFAULT_SUBTITLE_MODEL == "whisper-1"
    assert config.languages == DEFAULT_TRANSCRIPTION_LANGUAGES
    assert config.prompt == DEFAULT_TRANSCRIPTION_PROMPT
    assert config.chunk_context_chars == DEFAULT_CHUNK_CONTEXT_CHARS
    assert config.keywords
    assert config.diarization_enabled is DEFAULT_DIARIZATION_ENABLED
    assert config.chunk_max_bytes == DEFAULT_CHUNK_MAX_BYTES
    assert config.chunk_max_bytes < 25 * 1024 * 1024


def test_load_config_reads_provider_prefixed_models() -> None:
    """Модели OpenRouter в ENV -> попадают в конфиг без изменений."""
    transcription_model = "openai/gpt-transcribe"
    subtitle_model = "openai/whisper-1"

    config = load_config(
        {
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPTION_MODEL": transcription_model,
            "TRANSCRIPTION_SUBTITLE_MODEL": f" {subtitle_model} ",
        }
    )

    assert config.model == transcription_model
    # Пробелы вокруг значения обрезаются, как у остальных моделей.
    assert config.subtitle_model == subtitle_model


@pytest.mark.parametrize("subtitle_model", ["", "  ", "gpt-transcribe", "openai/gpt-4o-transcribe"])
def test_load_config_rejects_non_whisper_subtitle_model(subtitle_model: str) -> None:
    """Пустая или не-whisper модель субтитров -> ошибка при загрузке конфига."""
    with pytest.raises(TranscriptionError, match="TRANSCRIPTION_SUBTITLE_MODEL"):
        load_config(
            {
                "OPENAI_API_KEY": "test-key",
                "TRANSCRIPTION_SUBTITLE_MODEL": subtitle_model,
            }
        )


def test_load_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(TranscriptionError, match="OPENAI_API_KEY"):
        load_config()


def test_load_config_reads_languages_keywords_and_prompt() -> None:
    config = load_config(
        {
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPTION_LANGUAGES": "ru, en, de",
            "TRANSCRIPTION_KEYWORDS": "ReelsMaker, ContentLoader",
            "TRANSCRIPTION_PROMPT": "Team standup call",
            "TRANSCRIPTION_CHUNK_CONTEXT_CHARS": "123",
            "TRANSCRIPTION_MODEL": "gpt-4o-transcribe",
        }
    )

    assert config.model == "gpt-4o-transcribe"
    assert config.languages == ("ru", "en", "de")
    assert config.keywords == ("ReelsMaker", "ContentLoader")
    assert config.prompt == "Team standup call"
    assert config.chunk_context_chars == 123


def test_load_config_rejects_forbidden_keyword_characters() -> None:
    with pytest.raises(TranscriptionError, match="TRANSCRIPTION_KEYWORDS"):
        load_config(
            {
                "OPENAI_API_KEY": "test-key",
                "TRANSCRIPTION_KEYWORDS": "ok,<bad>",
            }
        )


def test_load_config_requires_diarization_flag_for_known_speakers() -> None:
    with pytest.raises(TranscriptionError, match="TRANSCRIPTION_DIARIZATION_ENABLED"):
        load_config(
            {
                "OPENAI_API_KEY": "test-key",
                "TRANSCRIPTION_KNOWN_SPEAKER_NAMES": "agent",
                "TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES": "data:audio/wav;base64,AAAA",
            }
        )


def test_load_config_requires_equal_known_speaker_lists() -> None:
    with pytest.raises(TranscriptionError, match="same number"):
        load_config(
            {
                "OPENAI_API_KEY": "test-key",
                "TRANSCRIPTION_DIARIZATION_ENABLED": "true",
                "TRANSCRIPTION_KNOWN_SPEAKER_NAMES": "agent,user",
                "TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES": "data:audio/wav;base64,AAAA",
            }
        )
