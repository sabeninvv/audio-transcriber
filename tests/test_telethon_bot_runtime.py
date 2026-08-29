from pathlib import Path
import sys
from types import SimpleNamespace


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from bot_app.config import BotConfig  # noqa: E402
from bot_app.session_store import FileSessionStore  # noqa: E402
from bot_app.telethon_bot import TelethonTranscriberBot  # noqa: E402


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


def test_run_starts_client_without_manual_run_until_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _build_config(tmp_path)
    session_store = FileSessionStore(config.sessions_dir)
    bot = TelethonTranscriberBot(config=config, session_store=session_store)

    class FakeLoop:
        def run_until_complete(self, awaitable) -> None:
            del awaitable
            raise AssertionError("run_until_complete must not be called here")

    class FakeClient:
        def __init__(self, session: str, api_id: int, api_hash: str) -> None:
            self.session = session
            self.api_id = api_id
            self.api_hash = api_hash
            self.loop = FakeLoop()
            self.handlers: list[tuple[object, object]] = []
            self.started_with: str | None = None
            self.disconnected = False

        def add_event_handler(self, handler, event) -> None:
            self.handlers.append((handler, event))

        def start(self, *, bot_token: str) -> "FakeClient":
            self.started_with = bot_token
            return self

        def run_until_disconnected(self) -> None:
            self.disconnected = True

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    created_clients: list[FakeClient] = []

    def fake_client_factory(session: str, api_id: int, api_hash: str) -> FakeClient:
        client = FakeClient(session=session, api_id=api_id, api_hash=api_hash)
        created_clients.append(client)
        return client

    fake_events = SimpleNamespace(NewMessage=lambda **kwargs: kwargs)
    fake_telethon = SimpleNamespace(TelegramClient=fake_client_factory, events=fake_events)
    monkeypatch.setitem(sys.modules, "telethon", fake_telethon)

    bot.run()

    assert len(created_clients) == 1
    client = created_clients[0]
    assert client.started_with == "test-token"
    assert client.disconnected is True
    assert len(client.handlers) == 3
