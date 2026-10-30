# Генератор анимированных SVG-диаграмм курса LLM Inference Systems.
# Палитра сайта: фон #101010, акцент #0000F2 (SMIL, работает внутри <img>).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "illustrations")
os.makedirs(OUT, exist_ok=True)
W, H = 960, 360
BG, ACC, TXT, MUT, BOX, GOOD = "#101010", "#0000F2", "#f5f5f5", "#9aa0a6", "#1a1a1a", "#2ecc71"

def arrow(x1, y1, x2, y2, dur=3, begin=0, color=None):
    c = color or MUT
    return f'''<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.5"/>
  <circle r="4" fill="{ACC}">
    <animateMotion dur="{dur}s" repeatCount="indefinite" begin="{begin}s" path="M{x1},{y1} L{x2},{y2}"/>
  </circle>'''

def box(x, y, w, h, label, accent=False, sub="", fill=None):
    f = fill or (ACC if accent else BOX)
    st = "#4444ff" if accent else "#3a3a3a"
    t = f'<text x="{x+w/2}" y="{y+h/2-4 if sub else y+h/2+5}" text-anchor="middle" fill="{TXT}" font-size="15" font-family="Segoe UI, sans-serif">{label}</text>'
    if sub:
        t += f'<text x="{x+w/2}" y="{y+h/2+16}" text-anchor="middle" fill="{MUT}" font-size="12" font-family="Segoe UI, sans-serif">{sub}</text>'
    a = '<animate attributeName="opacity" values="1;0.65;1" dur="2.4s" repeatCount="indefinite"/>' if accent else ""
    return f'<g>{a}<rect x="{x}" y="{y}" rx="10" width="{w}" height="{h}" fill="{f}" stroke="{st}" stroke-width="1.5"/>{t}</g>'

def title(t):
    return f'<text x="24" y="40" fill="{TXT}" font-size="20" font-weight="600" font-family="Segoe UI, sans-serif">{t}</text>'

def svg(name, body, h=H):
    doc = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}">
<rect width="{W}" height="{h}" fill="{BG}" rx="12"/>
{body}
</svg>'''
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(doc)
    print("ok", name)

BW, BH, Y = 150, 64, 140

# 00: три пути инференса
svg("m00-intro.svg", title("LLM-инференс: три пути к своему движку") +
    box(40, Y, 170, BH, "Веса модели", sub="safetensors / GGUF") +
    box(300, Y-80, 180, BH, "MLX / Apple Silicon", sub="tiny-llm, ядро курса") +
    box(300, Y, 180, BH, "CUDA / PyTorch", sub="llm.c, FlashAttention, vLLM") +
    box(300, Y+80, 180, BH, "CPU / GGUF", sub="llama.cpp, k-quants") +
    box(600, Y-80, 160, BH, "Свой движок", sub="paged KV, batching", accent=True) +
    box(600, Y, 160, BH, "Прод-сервинг", sub="TTFT, p95, экономика") +
    box(600, Y+80, 160, BH, "Квантование", sub="W4, память вместо FLOPs") +
    arrow(210, Y+BH/2, 300, Y-80+BH/2) +
    arrow(210, Y+BH/2, 300, Y+BH/2, 3, 0.7) +
    arrow(210, Y+BH/2, 300, Y+80+BH/2, 3, 1.4) +
    arrow(480, Y-80+32, 600, Y-80+32, 3, 2) + arrow(480, Y+32, 600, Y+32, 3, 0.5) +
    arrow(480, Y+80+32, 600, Y+80+32, 3, 1.2))

# 01: matmul
svg("m01-tensors.svg", title("Matmul: каждый токен читает все веса") +
    box(60, Y, 170, BH, "Токен", sub="embedding 1024") +
    box(300, Y, 200, BH, "W: 28 слоёв x 4 матрицы", sub="все ~600М весов", accent=True) +
    box(580, Y, 160, BH, "Логиты 151k") +
    box(770, Y, 150, BH, "Следующий токен") +
    arrow(230, Y+32, 300, Y+32) + arrow(500, Y+32, 580, Y+32) + arrow(740, Y+32, 770, Y+32) +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">Скорость декодинга упирается в чтение памяти: 0.6B fp16 = 1.2 ГБ на токен</text>')

# 02: attention
svg("m02-attention.svg", title("Attention: Q ищет в прошлом, GQA экономит кэш") +
    box(50, Y, 130, BH, "Q", sub="32 головы") +
    box(50, Y+90, 130, BH, "K, V", sub="8 kv-голов") +
    box(290, Y-20, 200, BH, "QKt/sqrt(d), softmax", sub="causal mask") +
    box(290, Y+90, 200, BH, "repeat_kv x4", sub="GQA") +
    box(600, Y+30, 170, BH, "взвеш. сумма V", accent=True) +
    box(790, Y+30, 130, BH, "выход головы") +
    arrow(180, Y+32, 290, Y+12) + arrow(180, Y+122, 290, Y+122) +
    arrow(490, Y+12, 600, Y+52) + arrow(490, Y+122, 600, Y+82) + arrow(770, Y+62, 790, Y+62))

# 03: цикл генерации
svg("m03-sampling.svg", title("Автогрессивный цикл: логиты, токен, снова") +
    box(40, Y, 150, BH, "Токенизатор", sub="BPE 151k") +
    box(240, Y, 170, BH, "Forward 28 слоёв", accent=True) +
    box(460, Y, 150, BH, "Логиты") +
    box(660, Y, 130, BH, "top-k/top-p") +
    box(810, Y, 110, BH, "токен") +
    f'''<path d="M865,204 L865,280 L120,280 L120,204" fill="none" stroke="{MUT}" stroke-width="1.5"/>
  <circle r="4" fill="{ACC}"><animateMotion dur="2.5s" repeatCount="indefinite" path="M865,204 L865,280 L120,280 L120,204"/></circle>
  <text x="480" y="300" text-anchor="middle" fill="{MUT}" font-size="13">каждый шаг: ещё один проход по весам, memory-bound</text>''' +
    arrow(190, Y+32, 240, Y+32) + arrow(410, Y+32, 460, Y+32) +
    arrow(610, Y+32, 660, Y+32) + arrow(790, Y+32, 810, Y+32))

# 04: kv cache
svg("m04-kv-cache.svg", title("KV-кэш: прошлое не пересчитываем") +
    box(50, Y, 170, BH, "Промпт 2048", sub="prefill, разом") +
    box(290, Y, 180, BH, "K,V кэш по слоям", sub="fp16: сотни МБ", accent=True) +
    box(540, Y, 160, BH, "Новый токен", sub="decode, по одному") +
    box(760, Y, 150, BH, "append в кэш") +
    arrow(220, Y+32, 290, Y+32) + arrow(470, Y+32, 540, Y+32) + arrow(700, Y+32, 760, Y+32) +
    f'<circle r="4" fill="{ACC}"><animateMotion dur="2s" repeatCount="indefinite" path="M760,172 L835,172 L835,262 L375,262 L375,204"/></circle>' +
    f'<text x="{W/2}" y="300" text-anchor="middle" fill="{MUT}" font-size="13">без кэша шаг растёт O(n2), с кэшем O(n); GQA режет кэш в 4 раза</text>')

# 05: quantization
svg("m05-quantization.svg", title("Квантование: память вместо FLOPs") +
    box(50, Y, 190, BH, "fp16 / 16 ГБ", sub="8B модель") +
    box(300, Y, 180, BH, "группы по 64", sub="scale + int4", accent=True) +
    box(540, Y, 170, BH, "~4.5 ГБ", sub="в 3.5 раза меньше") +
    box(760, Y, 150, BH, "де-квант", sub="на лету в matmul") +
    arrow(240, Y+32, 300, Y+32) + arrow(480, Y+32, 540, Y+32) + arrow(710, Y+32, 760, Y+32) +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">веса меньше, трафик памяти меньше, декодинг быстрее даже на CPU</text>')

# 06: kernels
svg("m06-kernels.svg", title("Кернел: тайлинг под кэш и SIMD") +
    box(50, Y, 170, BH, "DRAM ~200 циклов", sub="боттлнек") +
    box(290, Y, 170, BH, "Блоки в L2", sub="тайлинг", accent=True) +
    box(530, Y, 170, BH, "SIMD FMA", sub="8 float за такт") +
    box(760, Y, 150, BH, "Регистры", sub="L1 ~4 цикла") +
    arrow(220, Y+32, 290, Y+32) + arrow(460, Y+32, 530, Y+32) + arrow(700, Y+32, 760, Y+32) +
    f'<circle r="4" fill="{GOOD}"><animateMotion dur="2.4s" repeatCount="indefinite" path="M220,172 L290,172 L460,172 L530,172 L700,172 L760,172"/></circle>' +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">roofline: считаем не FLOPs, а сколько данных нужно на операцию</text>')

# 07: flash attention
svg("m07-fused.svg", title("FlashAttention: не материализуем S=QKt") +
    box(50, Y, 170, BH, "Q K V", sub="блоками 4x4") +
    box(290, Y, 200, BH, "tile: QKt, online softmax", sub="бегущий max, сумма", accent=True) +
    box(560, Y, 170, BH, "rescale + PV") +
    box(780, Y, 140, BH, "O", sub="без n2 в HBM") +
    arrow(220, Y+32, 290, Y+32) + arrow(490, Y+32, 560, Y+32) + arrow(730, Y+32, 780, Y+32) +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">память O(n) вместо O(n2); тот же ответ, другие проходы</text>')

# 08: batching
svg("m08-batching.svg", title("Continuous batching: запросы входят и выходят") +
    box(40, 70, 170, BH, "Очередь запросов", sub="приходят вразнобой") +
    box(40, Y+90, 170, BH, "Префилл чанками", sub="бюджет на шаг") +
    box(320, 90, 220, 180, "Шедулер: микс декодов и кусок префилла", accent=True) +
    box(640, 70, 150, BH, "выполнившиеся", sub="выходят") +
    box(640, Y+90, 150, BH, "новые", sub="входят сразу") +
    arrow(210, 102, 320, 130) + arrow(210, 232, 320, 210) +
    arrow(540, 130, 640, 102) + arrow(540, 210, 640, 232) +
    f'<text x="{W/2}" y="300" text-anchor="middle" fill="{MUT}" font-size="13">GPU простаивает, только когда кончились запросы, а не самый длинный</text>')

# 09: paged kv
svg("m09-paged-kv.svg", title("PagedAttention: KV как страницы памяти") +
    box(50, Y, 170, BH, "Логич. блоки", sub="по 16 токенов") +
    box(290, Y, 160, BH, "Block table", sub="логич. к физич.", accent=True) +
    box(520, Y, 200, BH, "Пул физич. блоков", sub="без резерва max_len") +
    box(780, Y, 140, BH, "Общий префикс", sub="shared blocks") +
    arrow(220, Y+32, 290, Y+32) + arrow(450, Y+32, 520, Y+32) + arrow(720, Y+32, 780, Y+32) +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">фрагментация 60-80% превращается в меньше 4%; copy-on-write при расхождении</text>')

# 10: distributed
svg("m10-distributed.svg", title("Дистрибуция: TP, PP и спекуляция") +
    box(50, Y-70, 180, BH, "Tensor parallel", sub="головы по GPU") +
    box(50, Y, 180, BH, "Pipeline", sub="слои по узлам") +
    box(50, Y+70, 180, BH, "Спекулятивный", sub="draft, проверка") +
    box(420, Y-20, 220, 120, "Один логический forward", accent=True) +
    box(740, Y-20, 170, BH, "токены", sub="x2-3 быстрее") +
    arrow(230, Y-38, 420, 120) + arrow(230, 172, 420, 172) + arrow(230, Y+102, 420, 220) +
    arrow(640, 172, 740, 172))

# 11: serving
svg("m11-serving.svg", title("Сервинг: TTFT и TPOT под SLO") +
    box(50, Y, 170, BH, "Клиент", sub="SSE-стрим") +
    box(290, Y, 170, BH, "llama-server / vLLM", accent=True) +
    box(530, Y, 170, BH, "TTFT p95 < 500мс", sub="TPOT < 50мс") +
    box(760, Y, 150, BH, "metrics json", sub="queue, tok/s") +
    arrow(220, Y+32, 290, Y+32) + arrow(460, Y+32, 530, Y+32) + arrow(700, Y+32, 760, Y+32) +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">p95, не средние: хвост важнее среднего, среднее врёт</text>')

# 12: final
svg("m12-final.svg", title("Финал: твой мини-движок инференса") +
    box(40, Y, 140, BH, "tokenizer") +
    box(200, Y, 150, BH, "шедулер", sub="continuous+чанки") +
    box(370, Y, 190, BH, "модель", sub="int4, paged KV, fused") +
    box(580, Y, 130, BH, "сэмплер") +
    box(730, Y, 180, BH, "SSE + /metrics", accent=True) +
    arrow(180, Y+32, 200, Y+32) + arrow(350, Y+32, 370, Y+32) +
    arrow(560, Y+32, 580, Y+32) + arrow(710, Y+32, 730, Y+32) +
    f'<circle r="4" fill="{ACC}"><animateMotion dur="3s" repeatCount="indefinite" path="M180,204 L910,204 L910,262 L100,262 L100,204"/></circle>' +
    f'<text x="{W/2}" y="300" text-anchor="middle" fill="{MUT}" font-size="13">все 12 модулей сходятся в один работающий сервис</text>')

print("done")

# 13: KV-хэндофф
svg("m13-kv-handoff.svg", title("KV-хэндофф: кэш вместо префилла") +
    box(40, Y-50, 180, BH, "Умная модель", sub="prefill 25k, ~10 ГБ кэша", accent=True) +
    box(300, Y-50, 200, BH, "Сжатие + RDMA", sub="int4: 2.5 ГБ, ~0.1 с") +
    box(570, Y-50, 180, BH, "Дешёвая модель", sub="декод без префилла", accent=True) +
    box(40, Y+60, 180, BH, "Без хэндоффа", sub="префилл заново: 10-60 с") +
    box(570, Y+60, 180, BH, "Префилл:", sub="десятки секунд") +
    arrow(220, Y-50+32, 300, Y-50+32) + arrow(500, Y-50+32, 570, Y-50+32) +
    arrow(220, Y+60+32, 570, Y+60+32, 5, 0.8, color="#8b0000") +
    f'<text x="{W/2}" y="330" text-anchor="middle" fill="{MUT}" font-size="13">Порог: remote prefill + transfer &lt; local prefill</text>')

# 14: разреженное внимание
svg("m14-sparse-attention.svg", title("Разреженное внимание: выбор осей") +
    box(40, 90, 200, 56, "Ось 1: головы", sub="MHA / GQA / MLA") +
    box(280, 90, 200, 56, "Ось 2: доступ", sub="full / local / sparse", accent=True) +
    box(520, 90, 200, 56, "Ось 3: позиция", sub="RoPE / YaRN") +
    box(760, 90, 160, 56, "Ось 4: кернел", sub="FA-4 / FlashMLA") +
    f'<g>' + "".join(
        f'<rect x="{40+i*16}" y="190" width="10" height="120" fill="{"#0000F2" if i in (1,2,17,18,34) else "#1a1a1a"}"/>'
        for i in range(56)) + '</g>' +
    f'<text x="{W/2}" y="336" text-anchor="middle" fill="{MUT}" font-size="13">Запрос видит выбранные блоки истории: top-k вместо всех 128</text>')

# 15: самоспекуляция
svg("m15-self-speculative.svg", title("Самоспекуляция: драфтер внутри модели") +
    box(60, Y, 200, BH, "Полная модель (L слоёв)", sub="верификация n+1 позиций", accent=True) +
    box(340, Y, 180, BH, "Урезанная (L-K слоёв)", sub="драфт подряд") +
    box(600, Y, 160, BH, "Принято", sub="acceptance x блок") +
    arrow(260, Y+32, 340, Y+32) + arrow(520, Y+32, 600, Y+32) +
    f'<path d="M600,{Y+32} C 700,{Y+32} 700,{Y-40} 440,{Y-40}" fill="none" stroke="{ACC}" stroke-width="1.2"/>' +
    f'<text x="{W/2}" y="300" text-anchor="middle" fill="{MUT}" font-size="13">Экономика: acceptance против стоимости draft-проходов; язык и токенизатор влияют</text>')

# 16: методология
svg("m16-measure.svg", title("Как не обмануться числом") +
    box(40, Y, 200, BH, "Один прогон", sub="«в 5 раз быстрее»", accent=True) +
    box(300, Y, 220, BH, "N=30, прогрев, p50/p95/p99", sub="Доверительный интервал") +
    box(580, Y, 160, BH, "Эффект вне ДИ?", sub="да -> значим") +
    f'<text x="{W/2}" y="290" text-anchor="middle" fill="{MUT}" font-size="13">Вендорские числа не переносятся; acceptance меряется всегда; язык + токенизатор фиксируются</text>')

# 17: unified memory и гибридная нода
svg("m17-unified-hybrid.svg", title("Unified memory: четвёртая ось инференса") +
    box(30, 90, 200, 64, "Большая память", sub="Apple / Strix Halo / Spark", accent=True) +
    box(285, 90, 190, 64, "Placement", sub="hot / cold experts") +
    box(530, 90, 180, 64, "Быстрая GPU", sub="decode / draft") +
    box(765, 90, 160, 64, "SLO", sub="TTFT / TPOT") +
    arrow(230, 122, 285, 122) + arrow(475, 122, 530, 122) + arrow(710, 122, 765, 122) +
    box(170, 220, 220, 64, "Prefill + target", sub="KV в общей памяти") +
    box(570, 220, 220, 64, "KV-handoff / RDMA", sub="remote + transfer") +
    arrow(390, 252, 570, 252, 4, 0.5) +
    f'<text x="{W/2}" y="335" text-anchor="middle" fill="{MUT}" font-size="13">Не TOPS решает decode: сравни байты/с, размещение и стоимость передачи</text>')
