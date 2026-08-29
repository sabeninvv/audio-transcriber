from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from bot_app.config import BotConfig  # noqa: E402
from bot_app.session_store import FileSessionStore  # noqa: E402
from bot_app.telethon_bot import TelethonTranscriberBot  # noqa: E402


class FakeMessage:
    def __init__(self, *, file_name: str, payload: bytes = b"audio") -> None:
        self.media = object()
        self.file = SimpleNamespace(name=file_name, ext=Path(file_name).suffix, size=len(payload))
        self._payload = payload
        self.download_paths: list[Path] = []

    async def download_media(self, *, file: str) -> str:
        destination = Path(file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._payload)
        self.download_paths.append(destination)
        return str(destination)


class FakeEvent:
    def __init__(self, *, sender_id: int, message: FakeMessage) -> None:
        self.sender_id = sender_id
        self.message = message
        self.raw_text = ""
        self.replies: list[str] = []
        self.reply_files: list[str | None] = []

    async def reply(self, message: str, *, file: str | None = None, **kwargs: object) -> None:
        del kwargs
        self.replies.append(message)
        self.reply_files.append(file)


def _build_config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        telegram_bot_token="test-token",
        telegram_api_id=12345,
        telegram_api_hash="hash-value",
        access_code="secret-code",
        sessions_dir=tmp_path / "DB",
        downloads_dir=tmp_path / "media",
        transcripts_dir=tmp_path / "transcripts",
    )


@pytest.mark.asyncio
async def test_media_requires_authorization(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)
    event = FakeEvent(sender_id=7, message=FakeMessage(file_name="meeting.mp3"))

    await bot.handle_media_message(event)

    assert event.replies == ["Сначала выполните /start и введите код доступа."]


@pytest.mark.asyncio
async def test_media_rejects_unsupported_format(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    session_store.authorize(7)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)
    event = FakeEvent(sender_id=7, message=FakeMessage(file_name="notes.txt"))

    await bot.handle_media_message(event)

    assert len(event.replies) == 1
    assert "Неподдерживаемый формат" in event.replies[0]


@pytest.mark.asyncio
async def test_media_transcribes_and_replies_with_text(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    session_store.authorize(100)
    transcript_path = tmp_path / "transcripts" / "result.md"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("ready transcript", encoding="utf-8")
    captured: dict[str, Path] = {}

    async def fake_transcribe(*, input_path: str | Path, output_dir: str | Path) -> Path:
        captured["input_path"] = Path(input_path)
        captured["output_dir"] = Path(output_dir)
        return transcript_path

    bot = TelethonTranscriberBot(
        config=config,
        session_store=session_store,
        transcribe_func=fake_transcribe,
    )
    event = FakeEvent(sender_id=100, message=FakeMessage(file_name="meeting.mp3"))

    await bot.handle_media_message(event)

    assert captured["output_dir"] == config.transcripts_dir
    assert event.replies == [
        "Файл получен. Начинаю скачивание...",
        "Скачивание завершено. Начинаю транскрибацию...",
        "ready transcript",
    ]
    assert event.reply_files == [None, None, None]
    assert event.message.download_paths[0].exists() is False


@pytest.mark.asyncio
async def test_media_transcription_error_returns_human_message(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    session_store.authorize(200)

    async def fail_transcribe(*, input_path: str | Path, output_dir: str | Path) -> Path:
        del input_path, output_dir
        raise RuntimeError("boom")

    bot = TelethonTranscriberBot(
        config=config,
        session_store=session_store,
        transcribe_func=fail_transcribe,
    )
    event = FakeEvent(sender_id=200, message=FakeMessage(file_name="meeting.mp3"))

    await bot.handle_media_message(event)

    assert event.replies == [
        "Файл получен. Начинаю скачивание...",
        "Скачивание завершено. Начинаю транскрибацию...",
        "Не удалось выполнить транскрибацию. Проверьте формат файла и попробуйте снова."
    ]
