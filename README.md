# audio-transcriber

Async Python module for transcribing local media files with the OpenAI Audio API.

## Requirements

- Python `3.14` (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` available on `PATH`
- `OPENAI_API_KEY` in the environment

## Install Dependencies

Install `ffmpeg` (it ships with `ffprobe`):

```bash
# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Then install Python dependencies:

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

subtitle_path = await transcribe_audio(
    input_path=Path("/tmp/meeting.wav"),
    output_dir=Path("/tmp/subtitles"),
    output_format="srt",
)
```

The public Python API accepts a single media file per call. `output_format` is `md` by default and `srt` for YouTube-compatible subtitles.

## Supported Formats

- `mp3`
- `wav`
- `mp4`
- `mov`
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
- writes merged output to `{INPUT_FILE_NAME}_{timestamp}.md` by default
- writes `{INPUT_FILE_NAME}_{timestamp}.srt` when `output_format="srt"`

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
- SRT output uses `whisper-1` segment timestamps and splits audio into 30-second requests so the model does not repeat an earlier phrase. OpenAI plans to shut that model down on 2027-02-26.
- SRT output cannot be combined with `TRANSCRIPTION_DIARIZATION_ENABLED=true`.

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

uv run --env-file ~/.envs/openai.env main.py --input media/audio/ --output subtitles/ --format srt

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

Generated SRT files contain numbered cues, `HH:MM:SS,mmm` ranges, and cue text. They do not include speaker labels.
