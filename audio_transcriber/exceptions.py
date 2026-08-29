class TranscriptionError(Exception):
    """Base exception for transcription workflow failures."""


class UnsupportedAudioFormatError(TranscriptionError):
    """Raised when the input audio format is not supported."""


class FfmpegNotAvailableError(TranscriptionError):
    """Raised when ffmpeg or ffprobe is unavailable."""


class AudioChunkingError(TranscriptionError):
    """Raised when audio preparation or chunking fails."""


class OpenAITranscriptionError(TranscriptionError):
    """Raised when the OpenAI transcription request fails."""
