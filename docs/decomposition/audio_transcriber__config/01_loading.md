### Загрузка конфигурации
Собирает `TranscriptionConfig` из окружения: ключ API, модели, промпт, подсказки языка и спикеров. Провайдер (OpenAI или OpenRouter) выбирается вне модуля через `OPENAI_BASE_URL` SDK.

* `load_config` — читает ENV (или переданный mapping), требует `OPENAI_API_KEY`, берёт `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_DIARIZATION_MODEL`, `TRANSCRIPTION_SUBTITLE_MODEL` (по умолчанию `whisper-1`; допускается только `whisper-1` или `openai/whisper-1`, иначе `TranscriptionError`) и проверяет согласованность known speakers с диаризацией.
