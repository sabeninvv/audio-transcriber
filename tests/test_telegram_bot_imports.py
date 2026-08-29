from pathlib import Path
import sys


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))


def test_bot_module_imports_successfully() -> None:
    from bot_app.telethon_bot import TelethonTranscriberBot

    assert TelethonTranscriberBot is not None
