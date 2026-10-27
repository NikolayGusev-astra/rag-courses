# Курс RAG Engineering: от мини-RAG до production

Практический курс по поиску по своим документам: 11 глав, каждый скрипт
запускается оффлайн на чистом Python, качество измеряется метриками, а не
ощущениями.

Читается на https://hermes-agent.ru/courses/rag-engineering/

## Чему научишься

- Решать, нужен ли RAG в задаче вообще (и когда хватает длинного контекста).
- Строить мини-RAG руками и понимать каждый слой.
- Гибридный поиск: BM25 + dense + RRF, реранкинг cross-encoder.
- Оценка качества: recall@k, MRR, nDCG, золотой набор, eval-гейт в CI.
- Паттерны HyDE / multi-hop / Corrective RAG / GraphRAG с честной оценкой цены.
- Agentic RAG и рабочий MCP-сервер базы знаний.
- Production: FastAPI, стриминг, кэш, права доступа, деградация, p95.

## Структура

```
rag-engineering/
  chapters/        тексты глав (markdown, как на сайте)
  code/            runnable-скрипты каждой главы
  illustrations/   SVG-диаграммы
SUBSCRIBE.md       как следить за обновлениями
CHANGELOG.md       что менялось между версиями
```

## Как проходить

Строго линейно: главы опираются на код предыдущих. Лабы обязательны, у каждой
главы есть «Критерий готовности» — не двигайся дальше, пока чекбоксы не
закрыты.

## Запуск

```bash
python code/01-mini-rag.py   # оффлайн, чистый stdlib
```

Исключения: глава 05 опционально использует transformers, глава 08 требует
`pip install mcp`, глава 09 — `pip install fastapi uvicorn`. Это написано
внутри каждой главы.

## Лицензия

Текст: CC BY-SA 4.0. Код: MIT.

## Обновления

Подписка: [SUBSCRIBE.md](SUBSCRIBE.md). История: [CHANGELOG.md](CHANGELOG.md).
Сайт курса: https://hermes-agent.ru/courses/rag-engineering/
