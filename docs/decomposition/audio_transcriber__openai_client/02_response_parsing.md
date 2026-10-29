### Разбор ответа
Извлекает текст и сегменты из ответа SDK (объект или dict) и отбрасывает некорректные сегменты.

* `_extract_transcript_text` — возвращает непустой `text`, иначе `OpenAITranscriptionError`.
* `_extract_diarized_segments` — собирает сегменты со спикером, пропуская неполные.
* `_extract_timestamped_segments` — собирает сегменты субтитров: пропускает неверные интервалы, пустой текст и повторы whisper; падает на нарушении порядка или пустом результате.
* `_whisper_segment_repeated` — признак повтора whisper по `compression_ratio` выше 2.4.
* `_require_timestamp` — достаёт конечную числовую метку времени или падает с номером сегмента.
* `_coerce_response_value` — читает поле из объекта SDK или dict.
