from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from audio_transcriber import transcribe_audio
from audio_transcriber.audio import is_supported_audio_path
from audio_transcriber.exceptions import TranscriptionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a local media file (mp3/wav/mp4/avi/webm/m4a/ogg/oga) "
            "or a directory of media "
            "files into markdown transcripts."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the input media file or directory "
            "(.mp3, .wav, .mp4, .avi, .webm, .m4a, .ogg, or .oga)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the generated markdown file will be stored.",
    )
    return parser


async def transcribe_directory(input_dir: Path, output_dir: Path) -> tuple[list[Path], list[str], bool, bool]:
    result_paths: list[Path] = []
    stderr_lines: list[str] = []
    found_supported_media = False
    had_failures = False

    for candidate_path in sorted(input_dir.iterdir()):
        if not candidate_path.is_file():
            continue
        if not is_supported_audio_path(candidate_path):
            stderr_lines.append(f"Skipping unsupported file: {candidate_path.name}")
            continue

        found_supported_media = True
        try:
            result_paths.append(
                await transcribe_audio(input_path=candidate_path, output_dir=output_dir)
            )
        except Exception as error:
            had_failures = True
            stderr_lines.append(f"Error transcribing {candidate_path.name}: {error}")
            if error.__cause__ is not None:
                stderr_lines.append(f"Cause: {error.__cause__}")

    return result_paths, stderr_lines, found_supported_media, had_failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_dir():
        result_paths, stderr_lines, found_supported_media, had_failures = asyncio.run(
            transcribe_directory(input_path, output_dir)
        )
        for result_path in result_paths:
            print(result_path)
        for stderr_line in stderr_lines:
            print(stderr_line, file=sys.stderr)
        if not found_supported_media:
            print(
                f"Error: no supported media files found in directory: {input_path}",
                file=sys.stderr,
            )
            return 1
        return 1 if had_failures else 0

    try:
        result = asyncio.run(
            transcribe_audio(
                input_path=input_path,
                output_dir=output_dir,
            )
        )
    except TranscriptionError as error:
        print(f"Error: {error}", file=sys.stderr)
        if error.__cause__ is not None:
            print(f"Cause: {error.__cause__}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
