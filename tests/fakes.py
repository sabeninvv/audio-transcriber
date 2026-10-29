from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import OpenAIError

from audio_transcriber.openai_client import TranscriptionResult


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


class FakeHttpApi:
    """Обработчик httpx.MockTransport: запоминает запросы и отдаёт заданный ответ или ошибку."""

    def __init__(
        self,
        *,
        response: httpx.Response | None = None,
        error: httpx.HTTPError | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response
        self._error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def sent_fields(self) -> dict[str, list[str]]:
        """Текстовые поля multipart последнего запроса, без файла."""
        request = self.requests[-1]
        raw = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        message = BytesParser(policy=default_email_policy).parsebytes(raw)
        fields: dict[str, list[str]] = {}
        for part in message.iter_parts():
            if part.get_filename():
                continue
            name = part.get_param("name", header="content-disposition")
            fields.setdefault(str(name), []).append(part.get_content())
        return fields


class FakeTranscriptionService:
    """Подмена класса TranscriptionService: вызов запоминает клиента, медиа не трогается."""

    def __init__(self) -> None:
        self.clients: list[object] = []

    def __call__(self, client: object) -> "FakeTranscriptionService":
        self.clients.append(client)
        return self

    async def transcribe_audio(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        config: object,
        output_format: object,
    ) -> Path:
        del input_path, config, output_format
        return Path(output_dir) / "result.md"
