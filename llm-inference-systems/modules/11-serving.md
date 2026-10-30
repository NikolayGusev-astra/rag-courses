---
chapter: 11
slug: 11-serving
title: "Модуль 11: Прод-сервинг: метрики и движки"
summary: "TTFT, TPOT, p95 против средних, llama-server и vLLM, prefix caching, стриминг и наблюдаемость."
---

# LLM Inference Systems. Модуль 11: Прод-сервинг, метрики и движки

Модель, которая отвечает 40 tok/s в блокноте, и модель, которая обслуживает 200 пользователей с бюджетом TTFT p95 < 500 мс — это два разных инженерных объекта. Между ними лежит слой, который в этой части курса мы до сих пор игнорировали: сервер, его метрики, его политика батчинга и кэширования, его экономика.


В модуле мы разберём, что именно измерять (и почему средние значения бесполезны), как считать стоимость миллиона токенов из пропускной способности, чем отличаются флаги `llama-server` и vLLM, что происходит с KV-кэшем при отмене запроса и как выглядит минимальный нагрузочный тест на 20 запросов, который уже показывает очередь. Всё код — на stdlib Python, работает на CPU: для бенчмарка мы поднимем mock-сервер, совместимый с OpenAI SSE, а затем те же скрипты направим на настоящий `llama-server`.

## 11.1 Три метрики, которых достаточно

Сервинг LLM описывается тремя числами на запрос и их распределениями.

**TTFT (time to first token)** — от отправки запроса до первого токена ответа. Внутри: время в очереди + префилл промпта. Это то, что пользователь воспринимает как «скорость отклика».

**TPOT (time per output token)**, он же ITL (inter-token latency) — средний интервал между последующими токенами. Для стриминга это скорость «печатания». Считается честно:

```
TPOT = (t_last_token − t_first_token) / (n_tokens − 1)
```

Знаменатель `n_tokens − 1`, а не `n_tokens`: между первым и последним токеном ровно `n−1` интервалов.

**Throughput** — агрегированная пропускная способность сервера, токенов/с в целом, а не на запрос. Растёт с размером батча почти линейно, пока не упрётся в compute.

Четвёртая, неявная метрика — **goodput**: сколько токенов в секунду сервер отдаёт, укладываясь в SLO. Если 30% запросов пробивают TTFT-бюджет, их токены в goodput не идут, и это меняет расчёт стоимости. К осени 2026 goodput стал штатной метрикой, а не концепцией: `vllm bench serve` принимает её прямо флагом — `--goodput ttft:1000 tpot:50` означает «засчитывать только запросы с TTFT ≤ 1000 мс и TPOT ≤ 50 мс».

### TPOT, ITL и stall: три разные вещи, которые нельзя называть одним словом

Методическая поправка к определению выше: TPOT — это **среднее**, и среднее скрывает tail-события. Разводите три понятия:

- **TPOT** — среднее время на токен за запрос: одна скалярная цифра, удобная для SLO-сравнения, но слепая к провалам внутри запроса.
- **ITL** (inter-token latency) — распределение интервалов между последовательными токенами: показывает, есть ли внутри «ровного» потока паузы, которые среднее усреднило.
- **Stall** — tail-событие: отдельный интервал, на порядок превышающий типичный (вытеснение KV, chunked prefill соседа, своп). Один stall на 500 мс посреди потока по 30 мс остаётся невидимым в TPOT, но виден в p99 ITL — и именно его замечает пользователь как «подвисание».

Практическое правило: TPOT — для контракта с продуктом, ITL-распределение — для отладки, stall-счётчик — для алертинга.

### Почему p95, а не среднее

Распределение задержек в сервинге LLM правоскошенное и мультимодальное: быстрые запросы (короткий промпт, пустая очередь) и медленные (длинный промпт, попал в очередь за большим батчем) образуют два горба. Среднее попадает между ними и не описывает ни один из режимов.

Практическое правило: **среднее скрывает хвост, хвост и есть продукт**. Пользователь, у которого 1 запрос из 20 ждёт 4 секунды, помнит именно его. Поэтому в SLO пишут p95 и p99, а среднее оставляют для дашборда.

Перцентиль считаем по определению с линейной интерполяцией:

```
rank = (p/100) * (n − 1)
p_p = x[floor(rank)] + (rank − floor(rank)) * (x[floor(rank)+1] − x[floor(rank)])
```

На 20 запросах p95 — это фактически второй худший результат; статистически это шум. Для p95 нужны сотни запросов, для p99 — тысячи. В задании 11.2 у нас 20 запросов, поэтому мы смотрим на p50/p95 как на индикатор формы распределения, а не как на контрактное число.

## 11.2 SLO-бюджеты: чат и batch — разные сервисы

| Сценарий | TTFT | TPOT | Throughput | Что оптимизируем |
|---|---|---|---|---|
| Интерактивный чат | p95 < 500 мс | p95 < 50 мс (20 tok/s) | вторично | хвост TTFT, prefix caching |
| Чат с длинным контекстом (RAG, 8k промпт) | p95 < 1.5 с | p95 < 50 мс | вторично | chunked prefill, кэш префикса |
| Code completion (inline) | p95 < 200 мс | p95 < 30 мс | вторично | маленькая модель, спекуляция |
| Batch-обработка (ночная) | не важно | не важно | максимум tok/s | большой батч, отказ от стриминга |
| Агент с tool-calls | p95 < 800 мс | p95 < 40 мс | средне | prefix caching (системный промпт + tools) |

Ключевой конфликт: **TTFT и throughput тянут в разные стороны**. Хотите маленький TTFT — не ставьте запрос в очередь, значит батч маленький, значит throughput низкий и USD/1M токенов высокий. Хотите throughput — собирайте батч побольше, TTFT растёт на время ожидания «попутчиков».

Формула компромисса для непрерывного батчинга: если сервер ждёт `w` секунд, чтобы набрать батч, TTFT ≥ `w + prefill`. Классический continuous batching (Orca, OSDI'22) убирает это ожидание почти полностью: новая последовательность входит в батч на следующем шаге декодирования, не дожидаясь, пока батч опустеет. Отсюда правило: **включайте continuous batching и не ждите полного батча** — в llama.cpp это `--cont-batching` (включён по умолчанию), в vLLM это единственный режим.

## 11.3 Экономика: USD/1M токенов из throughput

Стоимость считается в одну строчку:

```
USD/1M токенов = цена_часа / (throughput_tok_s * 3600 / 1e6)
```

Пример. Инстанс за 2.50 USD/час, агрегированная пропускная способность 3000 tok/s:

```
3000 tok/s * 3600 = 10.8M токенов/час
2.50 USD / 10.8 = 0.23 USD за 1M токенов
```

Если батч недогружен и реальная пропускная способность 400 tok/s — те же `2.50/час дают`1.74/1M, в 7.5 раза дороже. **Экономика сервинга — это на 90% утилизация батча, а не выбор модели.**

### Предел decode: memory-bound

Декодирование одного токена требует прочитать из памяти все веса модели (каждый параметр участвует ровно в одном MAC на токен). Значит потолок скорости на батче 1:

```
tok/s_max ≈ пропускная_способность_памяти / размер_весов
```

Честные числа:

| Носитель | BW | 7B Q4 (4.1 ГБ) | 8B fp16 (16 ГБ) |
|---|---|---|---|
| DDR5-5600 dual channel | ~80 ГБ/с | 19.5 tok/s | 5 tok/s |
| Apple M2 Pro (unified) | ~200 ГБ/с | 48 tok/s | 12.5 tok/s |
| A100 80GB HBM2e | ~2039 ГБ/с | 497 tok/s | 127 tok/s |
| H100 SXM HBM3 | ~3350 ГБ/с | 817 tok/s | 209 tok/s |

Реальные числа ниже теоретических на 30–50%: KV-кэш, накладные расходы ядер, неидеальная утилизация. M2 Pro на Qwen3-8B Q4_K_M выдаёт ~25–30 tok/s при потолке 48 — это нормально.

Из формулы видно два рычага: **квантование** (уменьшает числитель-веса) и **батчинг** (амортизирует чтение весов на B запросов: при батче B веса читаются один раз, токенов генерируется B). Пока батч не вышел в compute-bound режим, throughput ≈ B × tok/s_батч1.

## 11.4 llama.cpp server: флаги, которые реально меняют метрики

`llama-server` — HTTP-обёртка над `llama.cpp` с OpenAI-совместимым API. Работает на CPU (это наш основной путь для упражнений), на Metal, CUDA, Vulkan.

```bash
llama-server \
  -m qwen3-8b-q4_k_m.gguf \
  -c 8192 \
  -ngl 0 \
  --parallel 4 \
  --cont-batching \
  --metrics \
  --host 127.0.0.1 --port 8080
```

Разбор:

- `-c 8192` — общий размер контекста. **Важно:** при `--parallel N` llama.cpp делит `-c` между слотами: `--parallel 4 -c 8192` даёт по 2048 токенов на слот. Хотите 4 слота по 4k — ставьте `-c 16384`. Это самая частая ошибка при конфигурации.
- `-ngl 0` — ноль слоёв на GPU (чистый CPU). На Apple Silicon ставят `-ngl 99` — все слои на Metal. На CPU с 8 потоками Qwen3-8B Q4_K_M даёт ~10–15 tok/s, на M2 Pro с Metal — ~25–30.
- `--parallel 4` — четыре слота, то есть до 4 одновременных последовательностей в батче. Это главный рычаг throughput и главный источник роста TTFT (очередь).
- `--cont-batching` — непрерывный батчинг, новые запросы входят в текущий батч. По умолчанию включён, но лучше указать явно.
- `--metrics` — отдаёт Prometheus-метрики на `/metrics`.
- `-ctk q8_0 -ctv q8_0` — квантование KV-кэша. Для Qwen3-8B (36 слоёв, 8 KV-голов, head_dim 128) fp16 KV-кэш стоит `2 * 36 * 8 * 128 * 2 = 147 456` байт = **144 КиБ на токен**. При контексте 32k это 4.5 ГиБ — больше, чем сами веса в Q4. `q8_0` сжимает примерно вдвое (8 бит + scale на 32 элемента ≈ 8.5 бит против 16), то есть ~77 КиБ/токен и 2.4 ГиБ на 32k.

Проверка живости и метрик:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics | head -40
```

Стриминговый запрос (эндпоинт `/v1/chat/completions`):

```bash
curl -N -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"max_tokens":64,
       "messages":[{"role":"user","content":"Объясни, что такое KV-кэш, в двух предложениях."}]}'
```

`-N` отключает буферизацию curl, иначе вы не увидите поток. Ответ — SSE: строки `data: {...}` и терминальная `data: [DONE]`.

Эндпоинт `/completion` — «сырой», без chat-шаблона:

```bash
curl -N -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"n_predict":32,"prompt":"KV-кэш — это"}'
```

## 11.5 vLLM: те же метрики, другая машинерия

vLLM реализует PagedAttention (SOSP'23): KV-кэш нарезан на блоки фиксированного размера (по умолчанию 16 токенов), блоки лежат в общем пуле и выдаются последовательностям по мере надобности через таблицу блоков — как страницы в ОС. Следствия: фрагментация KV падает с 60–80% до единиц процентов, в батч влезает в 2–4 раза больше последовательностей, throughput растёт пропорционально.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8000
```

- `--gpu-memory-utilization 0.90` — доля VRAM под веса + KV-пул. Остаток 10% нужен CUDA-графам и активациям; ставить 0.98 — верный способ поймать OOM при первом длинном запросе.
- `--max-model-len` — верхняя граница контекста. Определяет размер KV-пула: чем меньше, тем больше блоков влезает в память и тем больше батч.
- `--enable-chunked-prefill` — префилл длинного промпта режется на куски и перемешивается с декодированием других последовательностей. Убирает «стопор» TTFT у соседей: без него промпт на 8k блокирует весь батч на время своего префилла.
- `--enable-prefix-caching` — общий префикс (системный промпт, tools, few-shot) считается один раз и переиспользуется всеми запросами через хеш блоков.
- `--max-num-seqs 64` — потолок одновременных последовательностей. Это ручной ограничитель очереди: больше — выше throughput, хуже TTFT.

Замер: `vllm bench serve --model ... --dataset-name random --num-prompts 200 --request-rate 8`. Всё это требует CUDA-GPU; на Windows CPU-сборка vLLM либо отсутствует, либо сильно урезана — для CPU-экспериментов остаётся llama.cpp. Наш бенчмарк ниже работает с любым OpenAI-совместимым SSE-эндпоинтом, поэтому переключается флагом `--url`.

**Версии и ловушка совместимости (актуально на осень 2026).** Актуальный релиз — vLLM 0.29.0 от 09.09.2026. Две ловушки, в которые регулярно попадают читатели: во-первых, новые сборки vLLM выбросили архитектуру sm_70 — V100 на них просто не стартует, «свежий» pip install на старом железе даёт нерабочий движок; во-вторых, наоборот, старые сборки не знают архитектуру Qwen3.5/3.6 (Gated DeltaNet + MTP) — модель не загрузится или загрузится без MTP-голов. Правило: версии движка и архитектуры модели сверяются до покупки железа, а не после — таблица поддержки в vLLM-репозитории меняется быстрее, чем документация туториалов.

llama.cpp, со своей стороны, поддерживает четыре способа спекуляции: draft model (внешняя маленькая модель), n-gram/prompt-lookup, MTP (нативные головы, Qwen3.6/3.8, EXAONE MoE) и DFlash/DSpark (параллельный драфт блоков). Выбор метода — флагом запуска, без пересборки.

Сравнение движков:

| | llama.cpp | vLLM | SGLang | MLX (tiny-llm) |
|---|---|---|---|---|
| Железо | CPU, Metal, CUDA, Vulkan | CUDA, ROCm | CUDA, ROCm | Apple Silicon |
| Формат весов | GGUF (k-quants) | safetensors fp16/awq/gptq | safetensors | MLX |
| Пейджинг KV | нет (слоты) | PagedAttention | RadixAttention | нет |
| Prefix caching | через `--prompt-cache` (слабо) | да | да, radix tree | нет |
| Роль в курсе | CPU-база, локальный сервинг | прод-эталон | прод-эталон | учебный движок |

## 11.6 Prefix caching: сколько именно экономит

**NVFP4 как отдельная категория артефактов.** Квантование перестало быть «настройкой при загрузке» и оформилось в отдельный класс публикуемых весов: репозитории вида unsloth/Qwen3.8-27B-NVFP4 — это готовые FP4-артефакты под Blackwell, а не конверсия на месте. Ориентировочная раскладка для Qwen3.8-27B на vLLM 0.17+ (цифры авторов замера, не независимые): NVFP4 помещается на одном GPU Blackwell; FP8 требует ~48 ГБ; BF16 — ~80 ГБ. То есть одна и та же модель живёт на одной карте, на 48-гигабайтной карте или на карте 80 ГБ в зависимости от формата — и выбор формата определяется не «желанием квантовать», а классом имеющегося железа.

Отрезвляющая цифра для ожиданий от FP4: DGX Spark даёт FP4-петафлопс в маркетинговых единицах, но 35B MoE-модель реально выдаёт на нём ~40 tok/s. Пиковые терафлопсы низкоточного формата ≠ throughput на реальной MoE-нагрузке: узкое место в памяти и маршрутизации, а не в умножителях.

Считаем честно. Системный промпт агента — 1500 токенов. Префилл на 8B-модели — это `2 * N_params * T` FLOP: `2 * 8e9 * 1500 = 2.4e13` = 24 TFLOP. На ускорителе с реальными 50 TFLOPS fp16 (половина пика) это **~480 мс только на префилл**, до первого токена. Именно поэтому TTFT агентных систем без кэша префикса живёт в районе секунды.

С кэшем: префикс посчитан один раз, дальше попадание в кэш стоит чтения KV-блоков из HBM — для 1500 токенов это `1500 * 144 КиБ ≈ 211 МиБ` чтения, на 2 ТБ/с — **~0.1 мс** плюс накладные. TTFT падает с ~500 мс до ~20–40 мс, то есть **в 10–20 раз**, при условии, что префикс побайтово совпадает (тот же системный промпт, тот же порядок tools).

Ограничение: кэш живёт в KV-пуле и вытесняется LRU. При `--max-model-len 8192` и 64 слотах в пуле держится ограниченное число блоков; если трафик разнородный, hit rate падает. Поэтому порядок промпта важен: **статичное — в начало, динамичное (вопрос пользователя, timestamp) — в конец**. Любой байт, изменившийся в начале, обнуляет весь кэш после него.

## 11.7 Стриминг и отмена запроса

SSE-поток — это `Content-Type: text/event-stream`, кадры `data: <json>\n\n`, терминатор `data: [DONE]`. Каждый кадр в OpenAI-схеме несёт `choices[0].delta.content` (в llama.cpp `/completion` — поле `content`).

Отмена (пользователь закрыл вкладку → TCP RST или клиент закрыл поток) — это не «остановка генерации», а сигнал серверу:

1. Сервер обнаруживает разрыв соединения на очередной записи в сокет (EPIPE).
2. Последовательность помечается отменённой и выкидывается из батча на следующем шаге.
3. **KV-кэш освобождается**: в vLLM блоки возвращаются в пул (и остаются в prefix cache, если он включён и блоки помечены как кэшируемые); в llama.cpp освобождается слот и его участок KV.
4. Уже потраченный префилл **не возвращается** — это чистая потеря compute. Если 30% запросов отменяют на 5-м токене, вы всё равно заплатили за префилл всех 30%.

Практический вывод: обрыв на длинном промпте дороже, чем обрыв на длинной генерации (префилл — самая дорогая фаза). Поэтому в агентных системах ставят таймаут на «время до первого токена» и не отменяют запросы, которые уже прошли префилл.

## 11.8 Наблюдаемость: что писать в лог

Минимальный набор на запрос:

```json
{"ts":"2026-02-11T10:12:03.441Z","req_id":"a91f","model":"qwen3-8b",
 "prompt_tokens":1523,"completion_tokens":214,
 "queue_ms":38,"ttft_ms":612,"tpot_ms":31.4,"e2e_ms":7320,
 "finish_reason":"stop","cache_hit_tokens":1500,"status":"ok"}
```

`queue_ms` отдельно от `ttft_ms` — иначе невозможно отличить «медленный префилл» от «медленной очереди», а лечатся они противоположными действиями (chunked prefill / prefix caching против увеличения числа слотов).

Метрики, которые смотрят на дашборде:

- гистограммы `ttft`, `tpot`, `e2e` (не средние);
- `requests_running` / `requests_waiting` — длина очереди; устойчивый рост `waiting` = исчерпан батч;
- `kv_cache_usage` (в vLLM `vllm:gpu_cache_usage_perc`) — выше 0.9 начинается вытеснение и падение hit rate;
- `tokens_in` / `tokens_out` в секунду, раздельно: prefill-bound и decode-bound профили лечатся по-разному;
- `cache_hit_tokens / prompt_tokens` — здоровье prefix caching.

Троттлинг по квоте — прикладной слой: считаете токены на API-ключ, при превышении отдаёте 429 до того, как запрос попал в очередь. Троттлинг после очереди бессмысленен: ресурс уже занят.

## Задание 11.1 (обязательное): три запроса, TTFT и TPOT из потока

Реализуйте функцию, которая делает один стриминговый запрос и возвращает измеренные метрики.

```python
def measure(url: str, prompt: str, max_tokens: int = 64, timeout: float = 120.0) -> dict:
    """Один SSE-запрос к OpenAI-совместимому /v1/chat/completions.

    Возвращает dict с ключами:
      ttft_ms: float   — от отправки запроса до первого непустого delta.content
      tpot_ms: float   — (t_last - t_first) / (n_tokens - 1); 0.0 если токен один
      e2e_ms:  float   — полное время до [DONE]
      tokens:  int     — число полученных токенов (непустых delta)
      text:    str     — склеенный ответ
    """
```

Запустите на `llama-server` три запроса с одинаковым промптом и `max_tokens=128`, сведите в таблицу:

| Прогон | TTFT, мс | TPOT, мс | Токенов | E2E, мс |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

Затем перезапустите сервер с `--parallel 4` (не забыв `-c 16384`, чтобы на слот осталось 4k) и повторите. Ожидаемо: на одиночных последовательных запросах метрики почти не изменятся — слоты влияют на конкурентный сценарий, а не на одиночный. Это и есть первый вывод упражнения.

## Задание 11.2: нагрузочный мини-тест с пуассоновским потоком

20 запросов, пуассоновское поступление (экспоненциальные интервалы), два режима: `rate=1 rps` (сервер недогружен) и `rate=8 rps` (сервер за пределом при 2 слотах). Соберите TTFT p50/p95, TPOT p50/p95, агрегированный throughput и посчитайте `/1M токенов при цене`0.50/час (аренда CPU-инстанса).

Ожидаемая картина: при `rate=1` TTFT ≈ одиночному TTFT (очереди нет), throughput = `rate × tokens`. При `rate=8` TTFT p95 вырастает в разы (очередь), throughput упирается в потолок `slots × tok/s_на_слот`, а TPOT остаётся почти неизменным — это признак того, что вы упёрлись в **очередь, а не в деградацию декодирования**. Различать эти два состояния — главный навык модуля.

## Код

Три файла в каталоге `bench/`. Только stdlib, Python 3.9+.

### `bench/mock_server.py` — OpenAI-совместимый SSE-сервер с искусственными задержками

```python
#!/usr/bin/env python3
"""Mock OpenAI /v1/chat/completions (stream=true) на stdlib.

Нужен, чтобы прогнать бенчмарк без llama.cpp и увидеть очередь.
Задержки искусственные, но модель задержек та же, что у настоящего сервера:
  TTFT = очередь_за_слот + базовая_задержка + префилл(длина_промпта)
  TPOT = 1 / TOK_S

Переменные окружения:
  MOCK_SLOTS        — число слотов (аналог llama.cpp --parallel), по умолчанию 4
  MOCK_TTFT         — базовая задержка до первого токена, с (0.08)
  MOCK_TOK_S        — токенов/с на слот (40)
  MOCK_PREFILL_TOK_S— скорость префилла, токенов/с (1200)
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SLOTS = int(os.environ.get("MOCK_SLOTS", "4"))
TTFT_BASE_S = float(os.environ.get("MOCK_TTFT", "0.08"))
TOK_S = float(os.environ.get("MOCK_TOK_S", "40"))
PREFILL_TOK_S = float(os.environ.get("MOCK_PREFILL_TOK_S", "1200"))

_slots = threading.Semaphore(SLOTS)   # ожидание слота попадает в TTFT


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 + отсутствие Content-Length => тело до закрытия соединения,
    # клиент читает поток построчно по мере поступления.
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):   # тишина в stderr
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/completion"):
            self.send_error(404)
            return

        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return

        if "messages" in req:
            prompt = " ".join(str(m.get("content", "")) for m in req["messages"])
        else:
            prompt = str(req.get("prompt", ""))
        n_prompt = max(1, len(prompt) // 4)          # грубая оценка токенов
        max_tokens = int(req.get("max_tokens", req.get("n_predict", 64)))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            with _slots:                             # <= здесь копится очередь
                time.sleep(TTFT_BASE_S + n_prompt / PREFILL_TOK_S)
                for i in range(max_tokens):
                    self.wfile.write(_sse({"choices": [{"delta": {"content": f"tok{i} "}}]}))
                    self.wfile.flush()
                    if i + 1 < max_tokens:
                        time.sleep(1.0 / TOK_S)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # клиент отменил запрос — слот освобождается выходом из with
            pass


def main():
    host, port = "127.0.0.1", int(os.environ.get("MOCK_PORT", "8080"))
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"mock server on http://{host}:{port}  slots={SLOTS} tok_s={TOK_S}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
```

### `bench/sse_client.py` — измерение TTFT/TPOT + CLI

```python
#!/usr/bin/env python3
"""Стриминговый клиент к OpenAI-совместимому серверу. Только stdlib."""
import argparse
import json
import time
import urllib.request


def measure(url: str, prompt: str, max_tokens: int = 64,
            timeout: float = 120.0, model: str = "local") -> dict:
    """Один SSE-запрос. Возвращает ttft_ms, tpot_ms, e2e_ms, tokens, text."""
    body = json.dumps({
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.perf_counter()
    t_first = None
    t_last = None
    tokens = 0
    parts = []

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:                      # построчно, по мере поступления
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            delta = ""
            if "choices" in obj:              # OpenAI-схема
                ch = obj["choices"][0]
                delta = (ch.get("delta") or {}).get("content") or ""
            elif "content" in obj:            # llama.cpp /completion
                delta = obj["content"] or ""
            if not delta:
                continue

            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            tokens += 1
            parts.append(delta)

    t_end = time.perf_counter()
    if t_first is None:                       # сервер не отдал ни одного токена
        return {"ttft_ms": float("nan"), "tpot_ms": float("nan"),
                "e2e_ms": (t_end - t0) * 1000, "tokens": 0, "text": ""}

    ttft_ms = (t_first - t0) * 1000
    tpot_ms = ((t_last - t_first) / (tokens - 1) * 1000) if tokens > 1 else 0.0
    return {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "e2e_ms": (t_end - t0) * 1000, "tokens": tokens, "text": "".join(parts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--prompt", default="Перечисли три причины, почему decode memory-bound.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=3, help="сколько запросов подряд")
    args = ap.parse_args()

    print(f"{'run':>3} {'ttft_ms':>9} {'tpot_ms':>9} {'tokens':>7} {'e2e_ms':>9}")
    for i in range(1, args.n + 1):
        r = measure(args.url, args.prompt, args.max_tokens)
        print(f"{i:>3} {r['ttft_ms']:>9.1f} {r['tpot_ms']:>9.1f} "
              f"{r['tokens']:>7} {r['e2e_ms']:>9.1f}")


if __name__ == "__main__":
    main()
```

### `bench/load_test.py` — пуассоновский нагрузочный тест

```python
#!/usr/bin/env python3
"""20 запросов с пуассоновским поступлением. Считает перцентили и USD/1M токенов."""
import argparse
import asyncio
import math
import random
import time

from sse_client import measure


def pct(xs, p: float) -> float:
    """Перцентиль с линейной интерполяцией. xs — неотсортированный список."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (rank - lo) * (s[hi] - s[lo])


async def fire(url, prompt, max_tokens, out, sem):
    async with sem:
        r = await asyncio.to_thread(measure, url, prompt, max_tokens)
        out.append(r)


async def run(url, prompt, max_tokens, n, rate, concurrency):
    out = []
    sem = asyncio.Semaphore(concurrency)
    t_start = time.perf_counter()
    tasks = []
    for _ in range(n):
        await asyncio.sleep(random.expovariate(rate))   # пуассоновский поток
        tasks.append(asyncio.create_task(
            fire(url, prompt, max_tokens, out, sem)))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start
    return out, wall


def report(out, wall, price_per_hour):
    ttft = [r["ttft_ms"] for r in out if r["tokens"] > 0]
    tpot = [r["tpot_ms"] for r in out if r["tokens"] > 1]
    e2e = [r["e2e_ms"] for r in out]
    total_tokens = sum(r["tokens"] for r in out)
    thr = total_tokens / wall if wall > 0 else 0.0
    cost = price_per_hour / (thr * 3600 / 1e6) if thr > 0 else float("inf")

    print(f"запросов: {len(out)}   wall: {wall:.2f} с   токенов: {total_tokens}")
    print(f"throughput: {thr:8.1f} tok/s   USD/1M: {cost:6.2f} "
          f"(при {price_per_hour:.2f} USD/час)")
    print(f"{'метрика':>8} {'p50':>9} {'p95':>9} {'max':>9}")
    for name, xs in (("ttft_ms", ttft), ("tpot_ms", tpot), ("e2e_ms", e2e)):
        print(f"{name:>8} {pct(xs,50):>9.1f} {pct(xs,95):>9.1f} {max(xs):>9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--prompt", default="Объясни PagedAttention в трёх предложениях.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rate", type=float, default=1.0, help="запросов/с (пуассон)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--price", type=float, default=0.50, help="USD/час за инстанс")
    args = ap.parse_args()

    out, wall = asyncio.run(
        run(args.url, args.prompt, args.max_tokens, args.n, args.rate, args.concurrency))
    report(out, wall, args.price)


if __name__ == "__main__":
    main()
```

## Как проверить

**Быстрый прогон без llama.cpp** (два терминала):

```bash
# терминал 1
MOCK_SLOTS=2 MOCK_TOK_S=40 python bench/mock_server.py
# терминал 2
python bench/sse_client.py --n 3 --max-tokens 64
```

Ожидаемый вывод (числа зависят от машины, важны соотношения):

```
run  ttft_ms  tpot_ms  tokens   e2e_ms
  1    210.4     25.1      64   1785.9
  2    208.7     25.0      64   1783.2
  3    211.9     25.2      64   1794.0
```

`tpot_ms ≈ 1000/40 = 25`, `ttft_ms ≈ 80 + len(prompt)/4/1200*1000`. Разброс между прогонами < 5%.

**Нагрузочный тест, недогруз:**

```bash
python bench/load_test.py --n 20 --rate 1.0 --max-tokens 64 --price 0.50
```

Ожидаемо: `ttft_ms p95` близок к одиночному TTFT (очередь не растёт, 1 rps < 2 слота × 40 tok/s), throughput ≈ `20 × 64 / wall`.

**Нагрузочный тест, перегруз:**

```bash
python bench/load_test.py --n 20 --rate 8.0 --max-tokens 64 --price 0.50
```

Ожидаемо: `ttft_ms p95` в 3–10 раз выше, `tpot_ms` практически не изменился, throughput упирается в `2 × 40 = 80 tok/s`. Это и есть подпись «очередь, а не деградация декодирования».

**Повтор с 4 слотами:**

```bash
MOCK_SLOTS=4 python bench/mock_server.py     # перезапустить
python bench/load_test.py --n 20 --rate 8.0 --max-tokens 64
```

Ожидаемо: throughput ~160 tok/s, `ttft p95` примерно вдвое ниже.

**На настоящем llama.cpp:**

```bash
llama-server -m qwen3-8b-q4_k_m.gguf -c 16384 -ngl 0 --parallel 4 --metrics --port 8080
python bench/sse_client.py --n 3 --max-tokens 128
python bench/load_test.py --n 20 --rate 1.0 --max-tokens 128
python bench/load_test.py --n 20 --rate 8.0 --max-tokens 128
```

Чеклист:

- [ ] `measure()` возвращает `ttft_ms` от отправки запроса, а не от соединения;
- [ ] `tpot_ms` считается по `n−1` интервалам; при одном токене — `0.0`, а не деление на ноль;
- [ ] перцентили считаются с интерполяцией, p95 на 20 точках не равен максимуму;
- [ ] при перегрузе растёт TTFT, а TPOT стоит на месте;
- [ ] `--parallel 4` повышает throughput и снижает TTFT p95 при том же `rate`;
- [ ] `-c` при `--parallel N` делится на N — контекст на слот не меньше нужного;
- [ ] в таблице есть `USD/1M`, посчитанный из throughput, а не из цены за токен;
- [ ] в логе запроса `queue_ms` отделён от `ttft_ms`.

## Сводка

- Метрики сервинга — TTFT, TPOT/ITL, агрегированный throughput; смотреть надо на p95/p99, среднее скрывает хвост, а хвост — это и есть пользовательский опыт. Goodput — штатная метрика (`vllm bench serve --goodput ttft:1000 tpot:50`); TPOT — среднее, ITL — распределение, stall — tail-событие: не смешивать.
- SLO чата и batch-обработки противоположны: чат покупает малый TTFT ценой throughput, batch — наоборот. Continuous batching позволяет не выбирать.
- `USD/1M токенов = цена_часа / (tok/s × 3600 / 1e6)`; экономика сервинга — это утилизация батча, а не выбор модели.
- Decode memory-bound: потолок `tok/s ≈ BW / размер_весов`. 7B в Q4 (4.1 ГБ) на DDR5-80 ГБ/с — 19.5 tok/s теоретически, 13–15 реально; на M2 Pro (200 ГБ/с) — до 48 теоретически, 25–30 реально.
- KV-кэш Qwen3-8B (36 слоёв, 8 KV-голов, head_dim 128) = 144 КиБ/токен; 32k контекста = 4.5 ГиБ, больше самих весов в Q4. `-ctk/-ctv q8_0` режет это примерно вдвое.
- Prefix caching экономит не «немного», а 10–20× TTFT: префилл 1500 токенов на 8B — это 24 TFLOP и ~500 мс, попадание в кэш — чтение 211 МиБ KV и ~0.1 мс.
- При перегрузе растёт TTFT (очередь), а TPOT остаётся ровным; если поплыл TPOT — проблема в декодировании (троттлинг, вытеснение KV), а не в очередности.


---


