### Запрос транскрибации
Клиент Gemini `generateContent`: отправляет чанк mp3 inline вместе с текстовой инструкцией и превращает ответ в `TranscriptionResult` без сегментов.

* `GeminiTranscriptionClient.__init__` — сохраняет ключ, модель (по умолчанию `gemini-3.8-flash`), промпт, языки, ключевые слова и httpx-транспорт для подмены в тестах.
* `GeminiTranscriptionClient.transcribe_file` — на каждый вызов открывает `httpx.AsyncClient` с таймаутом 600 с и шлёт `POST .../models/{model}:generateContent` с заголовком `x-goog-api-key`, текстом инструкции (с контекстом предыдущего чанка через `models.build_prompt`), файлом в base64 как `audio/mp3` и `thinkingLevel=low`. Сетевую ошибку, статус 4xx/5xx, неожиданную форму ответа (тело пишется в лог), отсутствие `candidates` (в тексте ошибки `blockReason` из `promptFeedback`), `finishReason` не `STOP` (например, `MAX_TOKENS`) и пустой текст превращает в `GeminiTranscriptionError`. Текст склеивает из всех `parts` первого кандидата.
* `_encode_audio` — читает чанк и кодирует его в base64; вызывается через `asyncio.to_thread`, чтобы не блокировать event loop.
