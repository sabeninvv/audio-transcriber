### Запрос транскрибации
Клиент ElevenLabs Speech-to-Text (Scribe): собирает multipart-запрос и превращает ответ в `TranscriptionResult`.

* `ElevenLabsTranscriptionClient.__init__` — сохраняет ключ, модель (по умолчанию `scribe_v2`), языки, ключевые слова (по умолчанию нет) и режимы; при `timestamps_enabled` запрещает диаризацию; принимает httpx-транспорт для подмены в тестах.
* `ElevenLabsTranscriptionClient.transcribe_file` — на каждый вызов открывает и закрывает `httpx.AsyncClient` с таймаутом 600 с, отправляет файл с `model_id`, `diarize`, `timestamps_granularity=word`, `tag_audio_events=false`, `language_code` только при одном языке и `keyterms` из ключевых слов; `prompt_suffix` игнорирует. Сетевую ошибку, статус 4xx/5xx, неожиданную форму ответа и пустой `text` превращает в `ElevenLabsTranscriptionError`. Сегменты возвращает только при диаризации или таймкодах.
