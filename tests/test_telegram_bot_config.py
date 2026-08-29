from pathlib import Path
import sys

import pytest


BOT_ROOT = Path(__file__).resolve().parents[1] / "s-family-audio-transcriber-bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from bot_app.config import DEFAULT_ACCESS_CODE, load_bot_config  # noqa: E402


def test_load_bot_config_uses_default_access_code_and_db_directory() -> None:
    config = load_bot_config(
        {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
        },
        bot_root=BOT_ROOT,
    )

    assert config.telegram_bot_token == "test-token"
    assert config.telegram_api_id == 12345
    assert config.telegram_api_hash == "hash-value"
    assert config.access_code == DEFAULT_ACCESS_CODE
    assert config.sessions_dir == BOT_ROOT / "DB"


def test_load_bot_config_reads_access_code_from_environment() -> None:
    config = load_bot_config(
        {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_ACCESS_CODE": "my-secret",
        },
        bot_root=BOT_ROOT,
    )

    assert config.access_code == "my-secret"


def test_load_bot_config_requires_telegram_bot_token() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        load_bot_config({}, bot_root=BOT_ROOT)


def test_load_bot_config_requires_telegram_api_id() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_API_ID"):
        load_bot_config(
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "TELEGRAM_API_HASH": "hash-value",
            },
            bot_root=BOT_ROOT,
        )


def test_load_bot_config_requires_telegram_api_hash() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_API_HASH"):
        load_bot_config(
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "TELEGRAM_API_ID": "12345",
            },
            bot_root=BOT_ROOT,
        )


def test_load_bot_config_requires_telegram_api_id_as_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        load_bot_config(
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "TELEGRAM_API_ID": "abc",
                "TELEGRAM_API_HASH": "hash-value",
            },
            bot_root=BOT_ROOT,
        )
