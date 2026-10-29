### Запросы транскрибации
Клиент поверх OpenAI-совместимого Audio API (OpenAI или OpenRouter через base URL SDK): выбирает режим запроса и собирает параметры.

* `OpenAITranscriptionClient.__init__` — сохраняет основную модель, модель диаризации, модель субтитров (по умолчанию `whisper-1`) и подсказки; при `timestamps_enabled` запрещает диаризацию; создаёт `AsyncOpenAI`, если клиент не передан.
* `OpenAITranscriptionClient.transcribe_file` — открывает файл, вызывает timestamped, diarized или стандартный запрос и возвращает `TranscriptionResult`; ошибки SDK логирует и оборачивает в `OpenAITranscriptionError`.
* `OpenAITranscriptionClient._create_standard_transcription` — запрос с `self._model`, промптом и контекстом чанка; для `gpt-transcribe` (в том числе с префиксом `OPENAI_MODEL_PREFIX`) передаёт `languages`/`keywords` через `extra_body`, иначе один `language`.
* `OpenAITranscriptionClient._create_timestamped_transcription` — `verbose_json` с сегментами для SRT на модели субтитров, без промпта; язык только если он один.
* `OpenAITranscriptionClient._create_diarized_transcription` — запрос `diarized_json` на модели диаризации с known speakers.
* `OpenAITranscriptionClient._active_model` — модель для логов ошибки: субтитров при таймкодах, диаризационная при диаризации, иначе основная.
