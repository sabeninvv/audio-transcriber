from audio_transcriber import transcribe_audio


def test_package_exports_transcribe_audio() -> None:
    assert callable(transcribe_audio)
