"""Курс RAG Engineering, глава 07-patterny. Версия 1.0.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 07-patterny.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================"""Сравнение baseline RRF и RRF+HyDE на золотом наборе. Только стандартная библиотека."""

import hashlib
import math
import re

# --- Корпус: 6 документов регламентов --------------------------------------
DOCS = {
    "REG-2024-101": (
        "Пароли сотрудников хранятся только в корпоративном менеджере секретов. "
        "Передавать пароли в мессенджерах запрещено. "
        "Плановая смена пароля выполняется каждые девяносто дней."
    ),
    "REG-2024-104": (
        "Резервные копии баз данных создаются ежедневно. "
        "Срок хранения резервной копии составляет тридцать дней. "
        "Восстановление из копии проверяется ежеквартально."
    ),
    "REG-2024-111": (
        "Об инциденте безопасности сотрудник сообщает в дежурную службу в течение часа. "
        "Дежурная служба регистрирует инцидент в журнале. "
        "Владелец системы уведомляется о каждом инциденте."
    ),
    "REG-2024-118": (
        "Журналы доступа к информационным системам хранятся двенадцать месяцев. "
        "Порядок хранения журналов утверждает руководитель службы безопасности. "
        "Выгрузка журналов выполняется по запросу аудитора."
    ),
    "REG-2024-123": (
        "Внешний подрядчик получает доступ только после подписания соглашения о конфиденциальности. "
        "Список систем для подрядчика согласует владелец системы. "
        "Доступ подрядчика пересматривается раз в полгода."
    ),
    "REG-2024-127": (
        "Каждый сотрудник проходит обучение по информационной безопасности при приёме на работу. "
        "Повторное обучение проводится ежегодно. "
        "Результаты обучения хранятся в кадровой системе."
    ),
}

# Золотой набор: вопрос и документ, который считаем релевантным
GOLD = [
    ("Сколько времени хранят журналы доступа к системам?", "REG-2024-118"),
    ("Что делать сотруднику при обнаружении инцидента?", "REG-2024-111"),
    ("Как часто проверяют восстановление резервных копий?", "REG-2024-104"),
    ("Какие условия доступа есть у внешних подрядчиков?", "REG-2024-123"),
    ("Где разрешено хранить пароли сотрудников?", "REG-2024-101"),
]

# --- Чанки: одно предложение = один чанк, помним родительский документ -----
CHUNKS = []  # (chunk_id, doc_id, text)
for doc_id, text in DOCS.items():
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    for i, sentence in enumerate(sentences):
        CHUNKS.append((f"{doc_id}#{i}", doc_id, sentence))

CHUNK_DOC = {cid: doc_id for cid, doc_id, _ in CHUNKS}
CHUNK_TEXT = {cid: text for cid, _, text in CHUNKS}

# --- Токенизация и очень грубый стеммер ------------------------------------
_SUFFIXES = (
    "тся", "ся", "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими",
    "ов", "ев", "ах", "ях", "ам", "ям", "ой", "ый", "ая", "ое", "ые", "ий",
    "ия", "ии", "ом", "ем", "ую", "юю", "а", "я", "ы", "и", "о", "е", "у", "ю", "ь",
)

def stem(token):
    for suffix in _SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token

def tokenize(text):
    return [stem(t) for t in re.findall(r"[а-яёa-z0-9]+", text.lower())]

# --- BM25 -------------------------------------------------------------------
K1, B = 1.5, 0.75

CHUNK_TOKENS = {cid: tokenize(text) for cid, text in CHUNK_TEXT.items()}
AVG_LEN = sum(len(toks) for toks in CHUNK_TOKENS.values()) / len(CHUNK_TOKENS)

DF = {}
for toks in CHUNK_TOKENS.values():
    for token in set(toks):
        DF[token] = DF.get(token, 0) + 1
N = len(CHUNK_TOKENS)

def bm25_search(query, k=None):
    query_tokens = tokenize(query)
    scores = {}
    for cid, toks in CHUNK_TOKENS.items():
        tf = {}
        for token in toks:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(1 + (N - DF[token] + 0.5) / (DF[token] + 0.5))
            denom = tf[token] + K1 * (1 - B + B * len(toks) / AVG_LEN)
            score += idf * tf[token] * (K1 + 1) / denom
        scores[cid] = score
    ranked = [cid for cid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return ranked if k is None else ranked[:k]

# --- Dense: hash-эмбеддер ---------------------------------------------------
DIM = 64

def normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec

def embed(text, dim=DIM):
    # для продакшена замените на живой эмбеддер
    vec = [0.0] * dim
    for token in tokenize(text):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    return normalize(vec)

CHUNK_VECS = {cid: embed(text) for cid, text in CHUNK_TEXT.items()}

def dense_search(query, k=None, query_vec=None):
    qv = query_vec if query_vec is not None else embed(query)
    scores = {
        cid: sum(a * b for a, b in zip(qv, vec)) for cid, vec in CHUNK_VECS.items()
    }
    ranked = [cid for cid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return ranked if k is None else ranked[:k]

# --- Слияние RRF ------------------------------------------------------------
def rrf_fuse(rankings, k=60, top=5):
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ranked = [cid for cid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return ranked[:top]

# --- HyDE с оффлайн-заглушкой ----------------------------------------------
# Шаблоны намеренно общие: заглушка имитирует только форму трюка (вопрос
# разворачивается в утверждения на языке предметной области), а не смысл.
# Реальный прирост даёт только живая LLM, и лишь если он виден на GOLD.
HYDE_TEMPLATES = (
    "Регламент устанавливает требования и порядок по вопросу: {q}.",
    "Ответственное подразделение утверждает правила, сроки хранения и порядок доступа.",
    "Документ описывает, кто, когда и в каком порядке выполняет требуемое действие.",
)

def hyde_expand(query):
    """Детерминированная заглушка вместо вызова LLM."""
    return " ".join(tpl.format(q=query) for tpl in HYDE_TEMPLATES)

def hyde_search(query, k=5):
    hypothetical = hyde_expand(query)
    # Гипотетический документ плюс исходный запрос: складываем и нормализуем
    combined = normalize(
        [a + b for a, b in zip(embed(hypothetical), embed(query))]
    )
    dense_ranking = dense_search(query, query_vec=combined)
    return rrf_fuse([bm25_search(query), dense_ranking], top=k)

# --- Метрика и main ---------------------------------------------------------
def recall_at_5(ranked_chunk_ids, gold_doc):
    return 1.0 if any(CHUNK_DOC[cid] == gold_doc for cid in ranked_chunk_ids) else 0.0

def main():
    base_scores, hyde_scores = [], []
    print("Сравнение на золотом наборе, метрика recall@5:")
    for question, gold_doc in GOLD:
        base_rank = rrf_fuse([bm25_search(question), dense_search(question)], top=5)
        hyde_rank = hyde_search(question, k=5)
        r_base = recall_at_5(base_rank, gold_doc)
        r_hyde = recall_at_5(hyde_rank, gold_doc)
        base_scores.append(r_base)
        hyde_scores.append(r_hyde)
        print(f"  rrf={r_base:.2f}  rrf+hyde={r_hyde:.2f}  | {question}")

    mean_base = sum(base_scores) / len(base_scores)
    mean_hyde = sum(hyde_scores) / len(hyde_scores)
    print(f"\nСредний recall@5: baseline={mean_base:.2f}, rrf+hyde={mean_hyde:.2f}")

    if mean_hyde > mean_base:
        print("Вердикт: метрика выросла, на этом наборе HyDE дал прирост.")
    elif mean_hyde < mean_base:
        print("Вердикт: метрика упала, HyDE навредил на этом наборе.")
    else:
        print("Вердикт: метрика не изменилась, прироста нет.")

# --- Опционально: настоящий HyDE через LLM (в этом скрипте не вызывается) ---
# Требует сетевого клиента и ключа, поэтому в оффлайн-скрипт не включён.
# Проверяйте прирост на GOLD так же, как для заглушки.
#
# def generate_hyde_via_api(query, model="<ваша модель>", api_key=None):
#     """Просит LLM написать гипотетический документ по запросу."""
#     prompt = (
#         "Напиши короткий фрагмент регламента, который отвечает на вопрос. "
#         f"Вопрос: {query}"
#     )
#     response = call_your_llm_provider(prompt, model=model, api_key=api_key)
#     return response.text

if __name__ == "__main__":
    main()

