# audio-transcriber

Async Python module for transcribing local media files with the OpenAI Audio API, ElevenLabs Speech-to-Text, or Gemini.

## Requirements

- Python `3.14` (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` available on `PATH`
- `OPENAI_API_KEY` (an OpenAI or OpenRouter key) `API_ELEVENLABS_API_KEY`, or `GEMINI_API_KEY` in the environment

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

- `OPENAI_API_KEY` for `TRANSCRIPTION_PROVIDER=openai` (default)
- `API_ELEVENLABS_API_KEY` for `TRANSCRIPTION_PROVIDER=elevenlabs`
- `GEMINI_API_KEY` for `TRANSCRIPTION_PROVIDER=gemini`

Optional quality controls:

- `TRANSCRIPTION_PROVIDER` (`openai`, `elevenlabs`, or `gemini`, default: `openai`)
- `TRANSCRIPTION_MODEL` (default: `gpt-transcribe`, `scribe_v2` for ElevenLabs, `gemini-3.8-flash` for Gemini)
- `TRANSCRIPTION_SUBTITLE_MODEL` (default: `whisper-1`, used for SRT output; only `whisper-1` or `openai/whisper-1`)
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

## OpenRouter

The OpenAI SDK reads `OPENAI_BASE_URL`, so OpenRouter needs only environment variables:

```bash
OPENAI_API_KEY=<openrouter key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
TRANSCRIPTION_MODEL=openai/gpt-transcribe
TRANSCRIPTION_SUBTITLE_MODEL=openai/whisper-1
```

Limitations on OpenRouter:

- `prompt` is accepted but ignored, so `TRANSCRIPTION_PROMPT` and previous-chunk context have no effect.
- There is no diarization model, so keep `TRANSCRIPTION_DIARIZATION_ENABLED=false`.
- Upstream requests time out after 60 seconds.

To switch back to OpenAI, unset `OPENAI_BASE_URL`, put an OpenAI key into `OPENAI_API_KEY`, and drop the `openai/` prefix from the models (or unset them to use the defaults). Keep one env file per provider and pick it with `uv run --env-file ...`; a value exported in the shell overrides the file.

## ElevenLabs

```bash
TRANSCRIPTION_PROVIDER=elevenlabs
API_ELEVENLABS_API_KEY=<elevenlabs key>
```

The module sends each chunk to `POST https://api.elevenlabs.io/v1/speech-to-text` with `scribe_v2`. Markdown, SRT and diarization work the same way as with OpenAI.

Differences from OpenAI:

- `TRANSCRIPTION_LANGUAGES` with one value is sent as `language_code`; several values leave language auto-detection on.
- `TRANSCRIPTION_KEYWORDS` are sent as `keyterms` only when set explicitly: ElevenLabs charges 20% more for requests with keyterms, so the default keyword list is not used.
- Scribe has no prompt, so previous-chunk context is not sent.
- `TRANSCRIPTION_PROMPT`, `TRANSCRIPTION_SUBTITLE_MODEL`, `TRANSCRIPTION_DIARIZATION_MODEL`, `TRANSCRIPTION_KNOWN_SPEAKER_NAMES` and `TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES` are OpenAI-only; setting any of them with ElevenLabs fails at startup.
- SRT cues are built from word timestamps without 30-second windows.

## Gemini

```bash
TRANSCRIPTION_PROVIDER=gemini
GEMINI_API_KEY=<gemini key>
```

The module sends each chunk inline as `audio/mp3` to `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` with a "transcribe verbatim" instruction. Only Markdown output without speaker labels is supported.

Differences from OpenAI:

- `TRANSCRIPTION_PROMPT`, `TRANSCRIPTION_LANGUAGES`, `TRANSCRIPTION_KEYWORDS` and previous-chunk context go into the instruction text.
- Thinking is set to `low`. `thinkingLevel` works on Gemini 3 models; other models may reject the request.
- A response cut by `MAX_TOKENS` or blocked by safety filters fails the file instead of writing a partial transcript.
- SRT output and `TRANSCRIPTION_DIARIZATION_ENABLED=true` fail at startup: Gemini returns no segment timestamps or speaker labels.
- `TRANSCRIPTION_SUBTITLE_MODEL`, `TRANSCRIPTION_DIARIZATION_MODEL`, `TRANSCRIPTION_KNOWN_SPEAKER_NAMES` and `TRANSCRIPTION_KNOWN_SPEAKER_REFERENCES` are OpenAI-only; setting any of them with Gemini fails at startup.
- Inline requests are limited to 20 MB; the default 1300-second chunk is about 14 MB after base64.

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
