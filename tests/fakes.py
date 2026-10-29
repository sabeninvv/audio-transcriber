from pathlib import Path
from types import SimpleNamespace

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
