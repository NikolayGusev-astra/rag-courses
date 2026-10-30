# Unified memory и гибридные ноды — официальные факты (собрано 30.10.2026)

## NVIDIA DGX Spark (источник: nvidia.com/en-us/products/workstations/dgx-spark/)
- GB10 Grace Blackwell Superchip: 20-ядерный Grace CPU, до 1 PFLOP AI (FP4)
- 128 ГБ когерентной unified memory (64 ГБ только через OEM-партнёров)
- ConnectX: до 4 систем DGX Spark работают вместе на больших моделях
- NVIDIA AI software stack, NIM; позиционирование: always-on agent workloads локально
- Замер сообщества (не официальный, помечать «замер авторов»): 35B MoE ~40 tok/s;
  Qwen3.8-27B NVFP4+DFlash2 на SGLang до 72 tok/s

## AMD Ryzen AI Max+ 395 / Strix Halo (источник: amd.com)
- 16 ядер Zen 5 / 32 потока, до 5.1 ГГц, 80 МБ кеша
- До 128 ГБ LPDDR5X-8000, 256-битная шина памяти (~273 ГБ/с теоретический потолок)
- Radeon 8060S: 40 CU RDNA 3.5; NPU XDNA2 50 TOPS; cTDP 45-120 Вт
- ROCm в 2026 работает с llama.cpp; главный рецепт — отдать память под GPU (gtt/uma)

## Apple M-серия (источник: apple.com)
- Когерентная unified memory CPU/GPU/Neural Engine; M3/M5 Ultra до 512 ГБ
- Шина от ~800 ГБ/с (Ultra-конфигурации) — главный advantage для decode

## KTransformers (источник: github.com/kvcache-ai/ktransformers)
- CPU-GPU heterogeneous inference/fine-tune фреймворк (Apache 2.0, ~19.6k stars)
- 15.10.2026: DeepSeek-V4-Flash LoRA fine-tune с нативными MXFP4 routed experts
- 13.09.2026: Kimi K2.5/K2.6 LoRA с RAWINT4 routed experts
- 26.08.2026: GLM-5.3-flash (1M контекст) на consumer GPU
- 16.08.2026: DeepSeek-V4-Flash на одном Ascend NPU + CPU expert offload
- KT-FT preprint (08.10.2026): layout-aware, schedule-optimized CPU-GPU co-computation для MoE fine-tuning
- Паттерн: MoE-эксперты в RAM (холодные) / VRAM (горячие), аккуратная работа с PCIe и кешем экспертов

## Аналитика (расчёты курса)
- Decode memory-bound: tok/s ≈ ширина_шины / байты_на_токен. 1 PFLOP FP4 не двигает decode.
- Сравнение шин: M3 Ultra ~800 ГБ/с > Spark ~273 ГБ/с (LPDDR5X на 256-бит) ≈ Strix Halo;
  дискретный RTX 5090: 32 ГБ GDDR7, ~1.8 ТБ/с — узкая память, но очень широкая шина
- Гибрид: unified-ноды держат большую модель/префилл, GPU — decode/draft; транспорт —
  KV-хэндофф по RDMA/ConnectX/Thunderbolt (см. модуль 13)
- NPU (XDNA2 50 TOPS) в LLM-инференсе ограничен: префилл-класс задач, не decode-горячий путь
