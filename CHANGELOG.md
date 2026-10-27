# Changelog

Формат: семантическое версионирование. Минор — новое содержание курса
(новая глава, обновлённые рекомендации), патч — исправления кода и фактов.

## [1.0.0] - 2026-10-27

### Первый публичный релиз

- 11 глав: от «нужен ли вам RAG» до production-сервиса на FastAPI.
- Гибридный поиск BM25 + dense + RRF, реранкинг cross-encoder.
- Свой eval-раннер (recall@k, MRR, nDCG) и гейт в CI.
- Паттерны HyDE / multi-hop / Corrective RAG / GraphRAG с честной оценкой цены.
- Рабочий MCP-сервер базы знаний (FastMCP).
- Весь код запускается оффлайн на чистом stdlib, кроме глав 05 (опциональный
  transformers), 08 (pip install mcp) и 09 (pip install fastapi uvicorn).
