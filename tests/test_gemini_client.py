import base64
import json
from pathlib import Path

import httpx
import pytest

from audio_transcriber.config import DEFAULT_GEMINI_MODEL, DEFAULT_TRANSCRIPTION_PROMPT
from audio_transcriber.exceptions import GeminiTranscriptionError
from audio_transcriber.gemini_client import GeminiTranscriptionClient
from tests.conftest import get_gemini_response
from tests.fakes import FakeHttpApi

API_KEY = "test-key"
TRANSCRIPT_TEXT = "Привет. Как дела?"
LANGUAGES = ("ru", "en")
KEYWORDS = ("ReelsMaker", "ContentLoader")
PREVIOUS_CHUNK_TAIL = "хвост прошлого чанка"
BLOCK_REASON = "SAFETY"
GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{DEFAULT_GEMINI_MODEL}:generateContent"
)


@pytest.mark.asyncio
async def test_transcribe_file_sends_inline_audio_with_instruction(chunk_path: Path) -> None:
    """Обычный вызов -> generateContent с ключом, mp3 в base64 и инструкцией; текст из всех parts."""
    api = FakeHttpApi(response=get_gemini_response(TRANSCRIPT_TEXT))
    client = GeminiTranscriptionClient(
        api_key=API_KEY,
        languages=LANGUAGES,
        keywords=KEYWORDS,
        transport=api.transport(),
    )

    result = await client.transcribe_file(chunk_path, prompt_suffix=PREVIOUS_CHUNK_TAIL)

    # Части ответа склеиваются в один текст без сегментов.
    assert result.text == TRANSCRIPT_TEXT
    assert result.segments == ()
    request = api.requests[0]
    assert str(request.url) == GENERATE_CONTENT_URL
    assert request.headers["x-goog-api-key"] == API_KEY
    body = json.loads(request.content)
    text_part, audio_part = body["contents"][0]["parts"]
    assert audio_part["inlineData"] == {
        "mimeType": "audio/mp3",
        "data": base64.b64encode(chunk_path.read_bytes()).decode("ascii"),
    }
    # Инструкция несёт промпт, языки, глоссарий и контекст прошлого чанка.
    assert DEFAULT_TRANSCRIPTION_PROMPT in text_part["text"]
    assert "Languages: ru, en." in text_part["text"]
    assert "Glossary: ReelsMaker, ContentLoader." in text_part["text"]
    assert text_part["text"].endswith(PREVIOUS_CHUNK_TAIL)
    assert body["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "low"}}


@pytest.mark.asyncio
async def test_transcribe_file_skips_empty_hints(chunk_path: Path) -> None:
    """Пустые промпт, языки и глоссарий -> в инструкции только фиксированная просьба расшифровать."""
    api = FakeHttpApi(response=get_gemini_response(TRANSCRIPT_TEXT))
    client = GeminiTranscriptionClient(
        api_key=API_KEY,
        prompt="",
        languages=(),
        keywords=(),
        transport=api.transport(),
    )

    await client.transcribe_file(chunk_path)

    text = json.loads(api.requests[0].content)["contents"][0]["parts"][0]["text"]
    assert text.startswith("Transcribe this audio verbatim.")
    assert "Languages:" not in text
    assert "Glossary:" not in text


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": {"message": "invalid argument"}}),
        httpx.Response(200, json={"candidates": "broken"}),
        httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}),
        get_gemini_response(TRANSCRIPT_TEXT, finish_reason="MAX_TOKENS"),
        get_gemini_response("   "),
    ],
    ids=["http-error", "unexpected-shape", "no-candidates", "max-tokens", "blank-text"],
)
@pytest.mark.asyncio
async def test_transcribe_file_rejects_failed_or_incomplete_response(
    chunk_path: Path,
    response: httpx.Response,
) -> None:
    """Ошибка API, неожиданная форма, блокировка, обрезка или пустой текст -> GeminiTranscriptionError."""
    api = FakeHttpApi(response=response)
    client = GeminiTranscriptionClient(api_key=API_KEY, transport=api.transport())

    with pytest.raises(GeminiTranscriptionError):
        await client.transcribe_file(chunk_path)


@pytest.mark.asyncio
async def test_transcribe_file_wraps_transport_errors(chunk_path: Path) -> None:
    """Сетевой сбой httpx -> GeminiTranscriptionError с исходной причиной."""
    transport_error = httpx.ConnectError("connection refused")
    api = FakeHttpApi(error=transport_error)
    client = GeminiTranscriptionClient(api_key=API_KEY, transport=api.transport())

    with pytest.raises(GeminiTranscriptionError) as error_info:
        await client.transcribe_file(chunk_path)

    assert error_info.value.__cause__ is transport_error


@pytest.mark.asyncio
async def test_transcribe_file_reports_block_reason(chunk_path: Path) -> None:
    """Промпт заблокирован фильтром -> причина блокировки в тексте ошибки."""
    api = FakeHttpApi(
        response=httpx.Response(200, json={"promptFeedback": {"blockReason": BLOCK_REASON}})
    )
    client = GeminiTranscriptionClient(api_key=API_KEY, transport=api.transport())

    with pytest.raises(GeminiTranscriptionError, match=f"block_reason={BLOCK_REASON}"):
        await client.transcribe_file(chunk_path)
