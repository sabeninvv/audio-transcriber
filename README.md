# audio-transcriber

Async Python module for transcribing local media files with the OpenAI Audio API.

## Requirements

- Python `3.12`
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` available on `PATH`
- `OPENAI_API_KEY` in the environment

## Install Dependencies

```bash
uv sync
```

## Public API

```python
from pathlib import Path

from audio_transcriber import transcribe_audio


result_path = await transcribe_audio(
    input_path=Path("/tmp/meeting.wav"),
    output_dir=Path("/tmp/transcripts"),
)
```

The public Python API accepts a single media file per call.

## Supported Formats

- `mp3`
- `wav`
- `mp4`
- `avi`
- `webm`
- `m4a`
- `ogg`
- `oga`

The module:

- normalizes accepted media to temporary `mp3`
- splits large files into smaller `mp3` chunks before upload
- sends chunk continuity context between chunk requests
- calls OpenAI transcription API with `gpt-transcribe` by default
- writes merged output to `{INPUT_FILE_NAME}_{timestamp}.md`

## Configuration

Required:

- `OPENAI_API_KEY`

Optional quality controls:

- `TRANSCRIPTION_MODEL` (default: `gpt-transcribe`)
- `TRANSCRIPTION_PROMPT`
- `TRANSCRIPTION_LANGUAGES` (comma-separated, default: `ru,en`)
- `TRANSCRIPTION_KEYWORDS` (comma-separated glossary)
- `TRANSCRIPTION_CHUNK_CONTEXT_CHARS` (default: `220`)

Diarization controls:

- `TRANSCRIPTION_DIARIZATION_ENABLED` (`true`/`false`, default: `false`)
- `TRANSCRIPTION_DIARIZATION_MODEL` (default: `gpt-4o-transcribe-diarize`)
- `TRANSCRIPTION_KNOWN_SPEAKER_NAMES` (comma-separated, up to 4)
- `TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES` (`|`-separated or newline-separated data URLs, same count as names)

Notes:

- `TRANSCRIPTION_KEYWORDS` entries cannot include `<`, `>`, carriage return, or newline.
- Set `TRANSCRIPTION_DIARIZATION_ENABLED=true` to add speaker labels. This is slower than the default transcript-only path.
- `gpt-transcribe` uses `languages` and `keywords` through request `extra_body`.
- `gpt-4o-transcribe-diarize` does not support prompt/keywords and is intended only when speaker labeling is required.

## Telegram Bot Usage

```python
from pathlib import Path

from audio_transcriber import transcribe_audio


async def handle_audio_message(local_audio_path: Path, transcripts_dir: Path) -> Path:
    return await transcribe_audio(
        input_path=local_audio_path,
        output_dir=transcripts_dir,
    )
```

## CLI Usage

```bash
export OPENAI_API_KEY="your_openai_api_key"


uv run --env-file ~/.envs/openai.env main.py --input media/audio/ --output transcripts/

TRANSCRIPTION_DIARIZATION_ENABLED=true \
uv run --env-file ~/.envs/openai.env main.py --input media/audio/ --output transcripts/
```

The CLI also accepts a directory path for `--input`.

When a directory is provided, the CLI:

- scans only top-level files in that directory
- transcribes supported media files
- skips unsupported files and reports each skip to stderr
- continues processing after per-file transcription failures and reports each failure to stderr
- exits with non-zero status if any supported file fails or if no supported media files are found

## Output Format

Generated Markdown files include:

- source file name
- generation timestamp
- chunk count
- original format
- working format (`mp3`)
- merged transcript text
- speaker timeline section when diarization mode is enabled
