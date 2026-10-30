# Спека курса LLM Inference Systems (для субагентов-авторов)

## Контекст
Пишешь модуль русскоязычного курса «LLM Inference Systems» для сайта hermes-agent.ru.
Аудитория: системные инженеры, знают Python и немного C++; пишут код руками.
Ядро курса — github.com/skyzh/tiny-llm (MLX, Apple Silicon, Qwen3), расширенный
CUDA-путём (llm.c, FlashAttention), llama.cpp (GGUF, CPU), vLLM (PagedAttention),
«How to Scale Your Model» (JAX/Google).

## Формат модуля (markdown, файл modules/<slug>.md)
```
# LLM Inference Systems. Модуль NN: <заголовок>

<1-2 абзаца: что делаем в модуле и зачем, сразу к делу>

## <разделы H2/H3>
<теория: формулы, блок-схемы текстом, таблицы сравнения>

## Задание <NN.1>...
<конкретное задание с интерфейсом функции, что подать на вход, что ожидать>

## Код
<runnable-код: python (основной) и при необходимости C++/Metal-наброски;
код рабочий, с комментариями, не псевдокод>

## Как проверить
<команда запуска, ожидаемый вывод, чеклист>

## Сводка
<5-7 буллетов, что унесли>

---
**Навигация по курсу:** [Хаб](/courses/llm-inference/) · [Следующий: <title>](/courses/llm-inference/<next-slug>/)
```

## Требования к стилю
- Русский технический язык, без канцелярита. Терминология: «параметры», не «ручки».
- Каждое утверждение про устройство модели/памяти — с числом (например: W4-веса 7B-модели
  занимают ~3.5 ГБ, fp16 KV-кэш на 4k токенов — X МБ при d_model=4096, n_layers=32 —
  числа считай честно по формуле в тексте).
- Код полный и runnable на CPU (torch без CUDA или numpy; Metal-фрагменты допустимы
  как иллюстрация, помечены «только Apple Silicon»).
- Никаких «в этом разделе мы рассмотрим» — сразу к делу.
- Объём модуля: 3000-6000 слов.

## Источники (первоисточники, не пересказы)
- tiny-llm: https://github.com/skyzh/tiny-llm (README + структура недель)
- llm.c (Karpathy), nanoGPT
- Qwen3 architecture blog / HF transformers llama-family
- FlashAttention-1/2 (Dao et al.)
- vLLM: Efficient Memory Management for LLM Serving with PagedAttention (SOSP'23)
- Continuous batching (Orca, OSDI'22)
- llama.cpp / GGUF / k-quants README
- How to Scale Your Model (JAX team, 2025-2026): арифметика памяти, TP/PP/EP
- GPU-mode (YouTube), Karpathy «Intro to LLMs» — как опциональные ссылки

Пиши файл в modules/<slug>.md. Не запускай git, не коммить. Код внутри модуля
сам не проверяй — это делает отдельный верификатор.
