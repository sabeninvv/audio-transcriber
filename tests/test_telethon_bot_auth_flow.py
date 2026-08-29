from pathlib import Path
import sys

import pytest


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from bot_app.config import BotConfig  # noqa: E402
from bot_app.session_store import FileSessionStore  # noqa: E402
from bot_app.telethon_bot import TelethonTranscriberBot  # noqa: E402


class FakeEvent:
    def __init__(self, *, sender_id: int, raw_text: str, has_media: bool = False) -> None:
        self.sender_id = sender_id
        self.raw_text = raw_text
        self.message = type("Message", (), {"media": object() if has_media else None})()
        self.replies: list[str] = []

    async def reply(self, message: str, **kwargs: object) -> None:
        del kwargs
        self.replies.append(message)


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
async def test_start_requests_access_code_for_unauthorized_user(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)
    event = FakeEvent(sender_id=1, raw_text="/start")

    await bot.handle_start(event)

    assert event.replies == ["Введите код доступа для активации бота."]


@pytest.mark.asyncio
async def test_text_message_authorizes_user_with_valid_code(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)
    event = FakeEvent(sender_id=123, raw_text="secret-code")

    await bot.handle_text_message(event)

    assert session_store.is_authorized(123) is True
    assert event.replies == ["Доступ подтвержден. Теперь отправьте медиафайл для транскрибации."]


@pytest.mark.asyncio
async def test_text_message_rejects_invalid_code(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)
    event = FakeEvent(sender_id=321, raw_text="wrong-code")

    await bot.handle_text_message(event)

    assert session_store.is_authorized(321) is False
    assert event.replies == ["Неверный код доступа. Попробуйте снова."]
