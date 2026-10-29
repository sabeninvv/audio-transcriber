from pathlib import Path

import httpx
import pytest


@pytest.fixture
def chunk_path(tmp_path: Path) -> Path:
    """Чанк mp3 на диске для клиентов транскрибации."""
    path = tmp_path / "chunk.mp3"
    path.write_bytes(b"audio")
    return path


def get_elevenlabs_word(
    text: str,
    start: float,
    end: float,
    *,
    speaker_id: str | None = None,
    word_type: str = "word",
) -> dict[str, object]:
    """Элемент words ответа ElevenLabs Speech-to-Text."""
    return {
        "text": text,
        "type": word_type,
        "logprob": 0.0,
        "start": start,
        "end": end,
        "speaker_id": speaker_id,
    }


def get_elevenlabs_response(text: str, words: list[dict[str, object]] | None = None) -> httpx.Response:
    """Успешный ответ ElevenLabs Speech-to-Text."""
    return httpx.Response(
        200,
        json={
            "language_code": "rus",
            "language_probability": 1.0,
            "text": text,
            "words": words or [],
        },
    )


def get_gemini_response(text: str, *, finish_reason: str = "STOP") -> httpx.Response:
    """Ответ Gemini generateContent с одним кандидатом, текст разбит на две части."""
    middle = len(text) // 2
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": text[:middle]}, {"text": text[middle:]}],
                    },
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        },
    )
