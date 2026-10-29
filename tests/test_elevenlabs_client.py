from pathlib import Path

import httpx
import pytest

from audio_transcriber.config import DEFAULT_ELEVENLABS_MODEL
from audio_transcriber.elevenlabs_client import ElevenLabsTranscriptionClient, _group_words, _Word
from audio_transcriber.exceptions import ElevenLabsTranscriptionError, TranscriptionError
from audio_transcriber.models import TranscriptionSegment
from tests.conftest import get_elevenlabs_response, get_elevenlabs_word
from tests.fakes import FakeElevenLabsApi

API_KEY = "test-key"
TRANSCRIPT_TEXT = "Привет. Как дела?"
LANGUAGE = "ru"
KEYWORDS = ("ReelsMaker", "ContentLoader")
FIRST_SPEAKER = "speaker_0"
SECOND_SPEAKER = "speaker_1"
PREVIOUS_CHUNK_TAIL = "хвост прошлого чанка"


@pytest.mark.asyncio
async def test_transcribe_file_sends_scribe_form_and_returns_text_only(chunk_path: Path) -> None:
    """Обычный режим -> форма Scribe с ключом, языком и keyterms; в ответе только текст."""
    api = FakeElevenLabsApi(
        response=get_elevenlabs_response(TRANSCRIPT_TEXT, [get_elevenlabs_word("Привет.", 0.0, 0.5)])
    )
    client = ElevenLabsTranscriptionClient(
        api_key=API_KEY,
        languages=(LANGUAGE,),
        keywords=KEYWORDS,
        transport=api.transport(),
    )

    result = await client.transcribe_file(chunk_path, prompt_suffix=PREVIOUS_CHUNK_TAIL)

    assert result.text == TRANSCRIPT_TEXT
    # Без диаризации и SRT сегменты не нужны writer-у.
    assert result.segments == ()
    assert api.requests[0].headers["xi-api-key"] == API_KEY
    assert api.sent_fields() == {
        "model_id": [DEFAULT_ELEVENLABS_MODEL],
        "diarize": ["false"],
        "tag_audio_events": ["false"],
        "timestamps_granularity": ["word"],
        "language_code": [LANGUAGE],
        "keyterms": list(KEYWORDS),
    }
    # Scribe не принимает промпт: контекст прошлого чанка не уходит в запрос.
    assert PREVIOUS_CHUNK_TAIL.encode() not in api.requests[0].content


@pytest.mark.asyncio
async def test_transcribe_file_leaves_language_auto_for_several_languages(chunk_path: Path) -> None:
    """Несколько языков и ключевые слова по умолчанию -> без language_code и keyterms."""
    api = FakeElevenLabsApi(response=get_elevenlabs_response(TRANSCRIPT_TEXT))
    client = ElevenLabsTranscriptionClient(
        api_key=API_KEY,
        languages=(LANGUAGE, "en"),
        transport=api.transport(),
    )

    await client.transcribe_file(chunk_path)

    sent_fields = api.sent_fields()
    # Scribe сам определит язык, если их больше одного.
    assert "language_code" not in sent_fields
    assert "keyterms" not in sent_fields


@pytest.mark.asyncio
async def test_transcribe_file_groups_words_by_speaker_when_diarized(chunk_path: Path) -> None:
    """Диаризация -> diarize=true и реплики по спикерам без пробелов и звуковых событий."""
    api = FakeElevenLabsApi(
        response=get_elevenlabs_response(
            TRANSCRIPT_TEXT,
            [
                get_elevenlabs_word("Привет", 0.0, 0.4, speaker_id=FIRST_SPEAKER),
                get_elevenlabs_word(" ", 0.4, 0.5, speaker_id=FIRST_SPEAKER, word_type="spacing"),
                get_elevenlabs_word("всем", 0.5, 0.9, speaker_id=FIRST_SPEAKER),
                get_elevenlabs_word("(смех)", 0.9, 1.2, word_type="audio_event"),
                get_elevenlabs_word("Как", 1.5, 1.7, speaker_id=SECOND_SPEAKER),
                get_elevenlabs_word("дела?", 1.7, 2.0, speaker_id=SECOND_SPEAKER),
            ],
        )
    )
    client = ElevenLabsTranscriptionClient(
        api_key=API_KEY,
        diarization_enabled=True,
        transport=api.transport(),
    )

    result = await client.transcribe_file(chunk_path)

    assert api.sent_fields()["diarize"] == ["true"]
    assert result.segments == (
        TranscriptionSegment(0.0, 0.9, FIRST_SPEAKER, "Привет всем"),
        TranscriptionSegment(1.5, 2.0, SECOND_SPEAKER, "Как дела?"),
    )


@pytest.mark.asyncio
async def test_transcribe_file_returns_cues_for_subtitles(chunk_path: Path) -> None:
    """Таймкоды без диаризации -> реплики с пустым спикером."""
    api = FakeElevenLabsApi(
        response=get_elevenlabs_response(
            TRANSCRIPT_TEXT,
            [
                get_elevenlabs_word("Привет.", 0.0, 0.5),
                get_elevenlabs_word("Как", 0.6, 0.8),
                get_elevenlabs_word("дела?", 0.8, 1.1),
            ],
        )
    )
    client = ElevenLabsTranscriptionClient(
        api_key=API_KEY,
        timestamps_enabled=True,
        transport=api.transport(),
    )

    result = await client.transcribe_file(chunk_path)

    # Конец предложения закрывает реплику субтитра.
    assert result.segments == (
        TranscriptionSegment(0.0, 0.5, "", "Привет."),
        TranscriptionSegment(0.6, 1.1, "", "Как дела?"),
    )


def test_group_words_closes_segment_longer_than_limit() -> None:
    """Длинная фраза без точки -> новая реплика, когда старая превысила 7 секунд."""
    words = [
        _Word(text="раз", type="word", start=0.0, end=3.0),
        _Word(text="два", type="word", start=3.0, end=6.5),
        _Word(text="три", type="word", start=6.5, end=7.5),
    ]

    segments = _group_words(words)

    assert segments == (
        TranscriptionSegment(0.0, 6.5, "", "раз два"),
        TranscriptionSegment(6.5, 7.5, "", "три"),
    )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"detail": {"status": "invalid_api_key"}}),
        httpx.Response(200, json={"detail": "accepted"}),
        get_elevenlabs_response("   "),
    ],
    ids=["http-error", "unexpected-shape", "blank-text"],
)
@pytest.mark.asyncio
async def test_transcribe_file_rejects_failed_or_empty_response(
    chunk_path: Path,
    response: httpx.Response,
) -> None:
    """Ошибка API, неожиданная форма или пустой текст -> ElevenLabsTranscriptionError."""
    api = FakeElevenLabsApi(response=response)
    client = ElevenLabsTranscriptionClient(api_key=API_KEY, transport=api.transport())

    with pytest.raises(ElevenLabsTranscriptionError):
        await client.transcribe_file(chunk_path)


@pytest.mark.asyncio
async def test_transcribe_file_wraps_transport_errors(chunk_path: Path) -> None:
    """Сетевой сбой httpx -> ElevenLabsTranscriptionError с исходной причиной."""
    transport_error = httpx.ConnectError("connection refused")
    api = FakeElevenLabsApi(error=transport_error)
    client = ElevenLabsTranscriptionClient(api_key=API_KEY, transport=api.transport())

    with pytest.raises(ElevenLabsTranscriptionError) as error_info:
        await client.transcribe_file(chunk_path)

    assert error_info.value.__cause__ is transport_error


def test_init_rejects_subtitles_with_diarization() -> None:
    """SRT вместе с диаризацией -> ошибка ещё до запроса."""
    with pytest.raises(TranscriptionError, match="diarization"):
        ElevenLabsTranscriptionClient(
            api_key=API_KEY,
            diarization_enabled=True,
            timestamps_enabled=True,
        )
