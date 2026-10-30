# Интеграция курса LLM Inference Systems в hermes-site.
# Пишет content/courses/llm-inference/*.md + _course.json, копирует иллюстрации.
# Запуск ПОСЛЕ того, как модули сгенерированы и проверены.
import os, re, shutil

COURSE = r"C:/Work/Assist/llm-inference-course"
SITE = r"C:/Work/Assist/hermes-site/site"
CID = "llm-inference"

MODULES = [
    ("00-intro", "Нужен ли тебе свой движок инференса и ландшафт путей",
     "Три пути к своему LLM-движку: MLX/Apple Silicon, CUDA/PyTorch, CPU/GGUF. Арифметика памяти, Qwen3 как подопытная, ландшафт сервинга vLLM, SGLang, llama.cpp.",
     "Три пути инференса: веса модели расходятся в MLX, CUDA и CPU-GGUF стеки"),
    ("01-tensors", "От matmul к тексту: тензоры и базовые операторы",
     "Matmul, RMSNorm, embedding и Linear руками на numpy с проверкой на оракуле. Почему декодинг ограничен памятью, а не FLOPs.",
     "Токен проходит через все веса модели к логитам и следующему токену"),
    ("02-attention", "Attention, RoPE и GQA",
     "Attention с causal mask, Multi-Head, GQA и RoPE на numpy. Проверка против torch и позиционная зависимость.",
     "Q ищет по ключам, GQA режет KV-кэш в четыре раза"),
    ("03-sampling", "Автогрессивный цикл и сэмплирование",
     "От логитов к токену: temperature, top-k, top-p. Полный forward Qwen3-0.6B на torch CPU и сверка с transformers.",
     "Цикл генерации: токенизатор, forward, сэмплер, снова токенизатор"),
    ("04-kv-cache", "KV-кэш и арифметика памяти",
     "KV-кэш: prefill и decode, формула памяти кэша, GQA-экономия, capacity-cache. Ускорение больше 10x на реальном прогоне.",
     "Промпт префиллится разом, декоды аппендятся в KV-кэш по токену"),
    ("05-quantization", "Квантование весов: W4 и GGUF",
     "Групповое int4-квантование, quantized matvec, формат GGUF и k-quants llama.cpp, GPTQ и AWQ в двух словах.",
     "fp16 веса квантуются группами по 64 и де-квантуются на лету"),
    ("06-kernels", "Кернелы: SIMD-матмул и tiled prefill",
     "Иерархия памяти, roofline, AVX2-кернел matvec на C, Metal и CUDA-листинги, tiled prefill.",
     "Данные идут из DRAM через L2 и SIMD к регистрам"),
    ("07-fused", "Fused-операторы и FlashAttention",
     "Fusion убирает трафик HBM: fused RMSNorm и SwiGLU, FlashAttention с online softmax на numpy.",
     "Q, K, V идут блоками через online softmax без материализации n на n матрицы"),
    ("08-batching", "Continuous batching и чанкерная префилла",
     "Статический против continuous батчинга, Orca, chunked prefill, симулятор шедулера с метриками TTFT и throughput.",
     "Шедулер микширует декоды и куски префилла, запросы входят и выходят на каждом шаге"),
    ("09-paged-kv", "PagedAttention: пейджинг KV-кэша",
     "KV-кэш как виртуальная память: BlockPool, BlockTable, prefix sharing, copy-on-write, paged attention.",
     "Логические блоки через block table указывают на пул физических страниц"),
    ("10-distributed", "Дистрибуция и спекулятивный декодинг",
     "Tensor и pipeline parallel, арифметика из How to Scale Your Model, спекулятивный декодинг и MTP.",
     "Три параллелизма сходятся в один логический forward"),
    ("11-serving", "Прод-сервинг: метрики и движки",
     "TTFT, TPOT, p95 против средних, llama-server и vLLM, prefix caching, стриминг и наблюдаемость.",
     "Клиент по SSE получает токены, движок отдаёт метрики под SLO"),
    ("12-final", "Финальный проект: мини-движок инференса",
     "Собираем qwen-serve-lite: шедулер, paged KV, int4, сэмплер, SSE-сервер. Приёмка шестью тестами.",
     "Все модули курса сходятся в один работающий сервис с метриками"),
    ("13-kv-handoff", "KV-ХЭНДОФФ: ПЕРЕДАЧА КЕША МЕЖДУ МОДЕЛЯМИ И УЗЛАМИ",
     "PD-дизагрегация, RDMA-транспорт (Thunderbolt, ConnectX), перенос кэша внутри семейства (ridge-маппер, CacheBridge) и между семействами (SwaT). Симулятор хэндоффа и ridge-маппер на numpy.",
     "Prefill-кластер передаёт сжатый KV-кэш decode-кластеру по RDMA вместо повторного префилла"),
    ("14-sparse-attention", "РАЗРЕЖЕННОЕ ВНИМАНИЕ И ДЛИННЫЙ КОНТЕКСТ",
     "Четыре оси внимания, MoBA и block-sparse гейтинг, attention sink, KeyDiff-эвикшн, SpecAttn. Block-sparse маска и эвикшн-эксперимент на numpy.",
     "Запрос читает выбранные блоки истории: top-k блоков вместо всех 128"),
    ("15-self-speculative", "САМОСПЕКУЛЯТИВНЫЙ ДЕКОДИНГ: ДРАФТЕР ВНУТРИ МОДЕЛИ",
     "Слой-скип драфт и полная верификация, S2-MoE, SpecAttn, экономика acceptance против стоимости драфта. Эксперимент на toy-модели с адаптивной длиной драфта.",
     "Урезанный проход драфтит, полный проход батч-верифицирует n+1 позиций"),
    ("16-measure", "Как не обмануться числом: методология замера",
     "TTFT/TPOT перцентилями, throughput-latency Парето, препринт-гигиена, приёмочные гейты correct/min. Бенчмарк с прогревом, N=30 и доверительными интервалами.",
     "Один прогон против N прогонов с ДИ: эффект значим только вне интервала"),
    ("17-unified-hybrid", "Unified memory и гибридные ноды: четвёртый путь инференса",
     "Аппаратная ось поверх MLX/CUDA/CPU: Apple M, AMD Strix Halo, NVIDIA DGX Spark, placement, MoE expert offload и гибридная нода с fast-head.",
     "Unified memory, горячие эксперты и KV-хэндофф между большой памятью и быстрой GPU"),
]

HUB_SUMMARY = ("Бесплатный практический курс LLM Inference Systems: тринадцать модулей от matmul до собственного "
               "мини-движка инференса с paged KV-кэшем, continuous batching и W4-квантованием. Ядро tiny-llm, "
               "расширение CUDA, llama.cpp и vLLM.")

def fm_str(s):
    return '"' + s.replace('"', '\\"') + '"'

def main():
    # 1) курс-каталог сайта
    cdir = os.path.join(SITE, "content", "courses", CID)
    os.makedirs(cdir, exist_ok=True)
    # 2) иллюстрации
    for slug, *_ in MODULES:
        src = os.path.join(COURSE, "illustrations", "m" + slug + ".svg")
        dst = os.path.join(SITE, "public", "courses", CID, "images", "m" + slug + ".svg")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    print("иллюстрации скопированы")
    # 3) главы
    for i, (slug, title, summary, alt) in enumerate(MODULES):
        body = open(os.path.join(COURSE, "modules", slug + ".md"), encoding="utf-8").read().strip()
        # Источники модулей могут содержать собственный frontmatter. Он не должен
        # попадать в тело: интегратор формирует единственный frontmatter ниже.
        body = re.sub(r"\A---\s*\n.*?\n---\s*\n*", "", body, count=1, flags=re.S).strip()
        img = f"![{alt}](/courses/{CID}/images/m{slug}.svg)"
        paras = body.split("\n\n")
        if len(paras) > 2:
            body = "\n\n".join(paras[:2] + [img] + paras[2:])
        else:
            body += "\n\n" + img
        nav = ["[Хаб курса](/courses/llm-inference/)"]
        if i > 0:
            nav.append(f"[Предыдущий: {MODULES[i-1][1]}](/courses/llm-inference/{MODULES[i-1][0]}/)")
        if i < len(MODULES) - 1:
            nav.append(f"[Следующий: {MODULES[i+1][1]}](/courses/llm-inference/{MODULES[i+1][0]}/)")
        body += "\n\n---\n\n**Навигация по курсу:** " + " · ".join(nav) + "\n"
        fm = f"""---
chapter: {i}
slug: {slug}
title: {fm_str("Модуль %02d: %s" % (i, title))}
summary: {fm_str(summary)}
---

"""
        open(os.path.join(cdir, slug + ".md"), "w", encoding="utf-8", newline="\n").write(fm + body + "\n")
        print("written", slug, len(body.split()), "слов")
    # _course.json
    import json
    json.dump({"version": "1.0.0", "repo": "https://github.com/skyzh/tiny-llm"},
              open(os.path.join(cdir, "_course.json"), "w", encoding="utf-8"), indent=2)
    print("done")

if __name__ == "__main__":
    main()
