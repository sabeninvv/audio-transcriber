from pathlib import Path
import sys


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from bot_app.session_store import FileSessionStore  # noqa: E402


def test_file_session_store_persists_authorized_user(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path / "DB")
    user_id = 123

    assert store.is_authorized(user_id) is False

    store.authorize(user_id)

    assert store.is_authorized(user_id) is True

    reloaded_store = FileSessionStore(tmp_path / "DB")
    assert reloaded_store.is_authorized(user_id) is True


def test_file_session_store_handles_corrupted_session_file(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "DB"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "1001.json").write_text("{", encoding="utf-8")

    store = FileSessionStore(sessions_dir)

    assert store.is_authorized(1001) is False
