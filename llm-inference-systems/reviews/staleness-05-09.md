# Отчёт об устаревании: модули 05-09

Дата: 2026-10-30. Проверка на фоне конца 2026. Метод: web-поиск (arxiv 2026 приоритетно) + синтез через deepseek-v4.1-flash.

## Сводка приоритетов

| Модуль | Приоритет | Главный пробел |
|---|---|---|
| 05-quantization | Критично | Нет ternary/1.58-bit, FP4 (NVFP4/MXFP4), KV-квант нового поколения |
| 06-kernels | Высокий | Нет FlashAttention-4/TMEM, Triton как стандарт, MLX/Metal-стек 2026 |
| 07-fused | Высокий | FA-2 уже не вершина; MLA-мегакернелы (FlashMLA/ThunderMLA) |
| 08-batching | Средний | Дефолты шедулеров изменились (SGLang fcfs); chunked prefill стал стандартом |
| 09-paged-kv | Высокий | RadixAttention vs PagedAttention, 2-bit KV-квант, attention backends vLLM |

---

## 05-quantization

**Сейчас в модуле:** групповое симметричное int4, пакетинг 2 int4/байт, GGUF Q4_K, llama.cpp CPU, GPTQ/AWQ.

**Устарело:**
- Тезис «int4 — практический пол» — снят: тернарные веса (BitNet b1.58 + bitnet.cpp, запуск 2B-моделей на слабом железе) в рабочем инструментарии.
- Пяти-trit packing (1.625 бит/вес) как лучшая схема упаковки — устарел.
- Отсутствие FP4-трека: NVFP4 (NVIDIA Blackwell, tensor cores) и MXFP4 — продакшн-форматы 2026.
- KV-квантование в модуле отсутствует как отдельная тема, хотя baseline-набор уже KIVI и TurboQuant (кодбук, turbo2/turbo3/turbo4 в полях llama.cpp/vLLM).

**Добавить:**
- BITCOS, «Breaking the 1.58-bit Barrier for Ternary LLMs», arXiv **2609.16338** (Intel, сент. 2026): presence bitmap + compacted sign vector; нули до **51.5%** весов в **29** ternary-моделях; **1.485 бит/вес** на самых разреженных; unpacking под AVX-512/AVX2/Xe2; до **1.28x** на тернарных matvec-кернелах, decode throughput до **1.18x** CPU / **1.27x** GPU.
- FP4-блок: NVFP4 + MXFP4; vLLM Humming kernel — MXFP4-веса распаковываются при загрузке, счёт в FP8 на tensor cores, **~4.2x на H20** и **~4.5x на H200** против DeepGEMM W8A8 на малых/средних батчах; FreeToken-движок держит MXFP4/NVFP4/FP8/BF16.
- KVarN, arXiv **2606.03458** (Huawei, июнь 2026): calibration-free, Hadamard rotation + dual-scaling variance normalization по обеим осям K и V; ключевой механизм деградации — накопление ошибки по шагам декодинга из-за неверных per-token scales; SOTA на 2-bit KV при длинном reasoning-декодинге (MATH500, AIME24, HumanEval); имплементация в vLLM: github.com/huawei-csl/KVarN.

**Приоритет: критично.**

## 06-kernels

**Сейчас в модуле:** иерархия памяти, roofline, BLAS, наивный matvec + векторизация, Metal-кернел simd_matmul (tiny-llm), CUDA-кернел из llm.c, tiled prefill, AVX2-задание.

**Устарело:**
- FlashAttention как «современный» референс без версии: актуальны FA-4 и VC-Attention.
- Отсутствует Triton-трек, хотя Triton остаётся стандартом для кастомных fused-ядер (RMSNorm, SwiGLU, RoPE, FlashAttention, INT8 GEMM, fused MoE) с roofline-анализом.
- Отсутствует Apple-трек: MLX компилируется в Metal; на Mac M5 в 2026 выбор MLX vs llama.cpp/GGUF — живой практический вопрос (Metal-ферма, unified memory).
- Нет кейса о приоритете фьюзинга над микрооптимизациями: ik_llama.cpp дал **26x** ускорение prompt processing для Qwen 3.5 за счёт фьюзинга CUDA-ядер.

**Добавить:**
- FlashAttention-4: до **1605 TFLOPS** на B200, **~2.7x** в vLLM 0.17.0; TMEM, async-ядра, 2-CTA MMA; поддержка Blackwell и Hopper.
- VC-Attention (MachGen, окт. 2026): открытая реализация для Blackwell, почти **2x** быстрее FA-4 в BF16 на B200 (заявление разработчиков).
- Triton-практикум + Liger-Kernel **v0.8.2** (CuTe-ядра для Hopper/Blackwell).

**Приоритет: высокий** (FA-4/VC-Attention — критично для раздела внимания); **средний** (Triton- и Apple-треки).

## 07-fused

**Сейчас в модуле:** fused RMSNorm, fused SwiGLU, FlashAttention (наив, online softmax, FA-2).

**Устарело:**
- FA-2 как вершина прогресса — на Blackwell вершина FA-4, поверх неё VC-Attention с заявленными ~**2x** против FA-4 в BF16 на B200.
- Нет MLA-кернелов — целый класс моделей выпадает.
- Нет готовых библиотечных fused-блоков: Liger-Kernel v0.8.2 даёт fused RMSNorm/RoPE/SwiGLU/MoE на CuTe-ядрах Hopper/Blackwell.

**Добавить:**
- MLA-кернелы: FlashMLA (DeepSeek) и ThunderMLA (ThunderKittens, Together AI) — «мегакернел», убирает накладные расходы запуска ядер, **+20–35%** против FlashMLA; Kimi K2.5 — до **2x** лучший TTFT.
- FlexDecoding для кастомных attention-паттернов.
- FA-4 / VC-Attention как продолжение линии FA-2 (см. модуль 06).

**Приоритет: критично** (FA-2 → FA-4/VC-Attention), **высокий** (MLA-кернелы, Liger-Kernel).

## 08-batching

**Сейчас в модуле:** static batching, continuous batching как цикл событий, prefill vs decode в одном батче, политики шедулера, симулятор.

**Устарело:**
- Политики шедулера подаются как стабильная константа: в SGLang **v0.5.18** (авг. 2026) дефолт — **fcfs** (был **lpm** в v0.4.0). Дефолты меняются, материал обязан быть привязан к версии.
- «Prefill vs decode в одном батче» без chunked prefill: Sarathi-style chunked prefill — стандарт в vLLM, бюджет токенов управляет очередью длинных запросов (проверено по коммиту **ced6857**, сент. 2026).
- Нет сравнения архитектур шедулеров SGLang vs vLLM (2026) — живая тема; под нагрузкой vLLM в **1.5–2.7x** быстрее llama.cpp.

**Добавить:** chunked prefill с бюджетом токенов как отдельный раздел симулятора; версионная таблица дефолтных политик; разбор SGLang vs vLLM по архитектуре шедулера.

**Приоритет: высокий** (chunked prefill, версионность дефолтов), **средний** (сравнение SGLang/vLLM).

## 09-paged-kv

**Сейчас в модуле:** фрагментация, логические/физические блоки, prefix sharing + copy-on-write, внимание по страницам, KV-квант по блокам.

**Устарело:**
- Prefix sharing через PagedAttention как потолок: RadixAttention в SGLang хранит KV-cache в radix-дереве с LRU — более агрессивный prefix caching, чем PagedAttention в vLLM; бренд RadixArk (янв. 2026).
- Блочное KV-квантование как финальная тема: ушло вперёд — per-token scales, Hadamard-ротация, кодбуки; 2-bit KV в reasoning-декодинге — рабочий режим (KVarN, arXiv 2606.03458; TurboQuant).
- Дефолтные paged-кернелы как единственный путь: для MLA-моделей их заменяют ThunderMLA/FlexDecoding.
- Нет цифр по памяти: vLLM ест **~70 ГБ** против **~56 ГБ** у SGLang из-за преаллокации KV.
- Нет карты backend'ов: vLLM attention backends — FLASH_ATTN (дефолт, рекомендация Red Hat 2026) и FLASHINFER.

**Добавить:** RadixAttention (radix-дерево + LRU) против PagedAttention; карту backend'ов vLLM; пер-токен scales/Hadamard/кодбуки как продолжение блочного KV-кванта; ThunderMLA/FlexDecoding как замена дефолтным paged-кернелам.

**Приоритет: высокий.**

## Сводка

| Модуль | Приоритет | Ключевые правки |
|---|---|---|
| 05-quantization | критично | тернарные веса + BITCOS (2609.16338), FP4 (NVFP4/MXFP4, Humming), KVarN (2606.03458) |
| 06-kernels | высокий | FA-4 (1605 TFLOPS B200, 2.7x в vLLM 0.17.0), VC-Attention, Triton/Liger, MLX на M5 |
| 07-fused | критично / высокий | FA-2 → FA-4/VC-Attention; FlashMLA/ThunderMLA (+20–35%, TTFT до 2x) |
| 08-batching | высокий | chunked prefill (ced6857), дефолт fcfs в SGLang v0.5.18, vLLM 1.5–2.7x против llama.cpp |
| 09-paged-kv | высокий | RadixAttention vs PagedAttention, 70 vs 56 ГБ, FLASH_ATTN/FLASHINFER, 2-bit KV в reasoning |
