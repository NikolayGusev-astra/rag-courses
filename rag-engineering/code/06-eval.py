"""Курс RAG Engineering, глава 06-eval. Версия 1.1.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 06-eval.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================

import math

def recall_at_k(relevant, retrieved, k):
    """Доля релевантных документов, попавших в первые k позиций."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & set(relevant)
    return len(hits) / len(relevant)

def reciprocal_rank(relevant, retrieved):
    """Обратный ранг первого релевантного документа."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def dcg(grades):
    """DCG с логарифмическим дисконтом по позициям."""
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))

def ndcg_at_k(graded_relevant, retrieved, k):
    """graded_relevant: словарь doc_id -> степень релевантности 0..2."""
    gains = [graded_relevant.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(graded_relevant.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def evaluate(gold, search_fn, k=5):
    """gold: список пар (вопрос, список id релевантных документов)."""
    recalls, rrs, ndcgs = [], [], []
    for question, relevant in gold:
        retrieved = [doc["id"] for doc in search_fn(question, k)]
        # В GOLD разметка бинарная, поэтому 1 для релевантных, 0 для остальных
        graded = {doc_id: 1 for doc_id in relevant}
        recalls.append(recall_at_k(relevant, retrieved, k))
        rrs.append(reciprocal_rank(relevant, retrieved))
        ndcgs.append(ndcg_at_k(graded, retrieved, k))
    n = len(gold)
    return {
        "recall@k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
        "ndcg@k": sum(ndcgs) / n,
    }


"""Eval-раннер поиска: recall@k, MRR и nDCG@k. Только стандартная библиотека."""
import hashlib
import math
import re
import sys

K = 5
THRESHOLD = 0.6

# Корпус служебных регламентов: 6 документов
DOCS = [
    {"id": "d1", "text": "Регламент отпусков: заявление на ежегодный оплачиваемый отпуск подается не позднее чем за две недели, отпускные выплачиваются за три рабочих дня до начала отпуска."},
    {"id": "d2", "text": "Инструкция по VPN: корпоративный VPN подключается через клиент WireGuard, конфигурация выдается в отделе ИТ, пароль к VPN меняется раз в квартал."},
    {"id": "d3", "text": "Резервное копирование: полные бэкапы баз данных выполняются ежедневно в 02:00, проверка восстановления из копии проводится раз в месяц."},
    {"id": "d4", "text": "Управление доступами: заявка на доступ к новой системе оформляется в Service Desk и согласуется с владельцем системы."},
    {"id": "d5", "text": "Командировки и компенсации: суточные выплачиваются за каждый день командировки, расходы на проезд компенсируются по чекам после возвращения."},
    {"id": "d6", "text": "Регламент REG-2024-118: хранение резервных копий во внешнем контуре, согласно артикулу KB-7734 срок хранения архивных копий составляет один год."},
]

# Золотой набор: пары «вопрос, список id релевантных документов»
GOLD = [
    {"question": "За сколько дней до начала отпуска нужно подать заявление?", "relevant": ["d1"]},
    {"question": "Как подключить корпоративный VPN и где взять конфигурацию?", "relevant": ["d2"]},
    {"question": "Как часто выполняются полные резервные копии баз данных?", "relevant": ["d3"]},
    {"question": "Как вернуть стоимость билетов после служебной поездки?", "relevant": ["d5"]},
    {"question": "Какой срок хранения задает артикул KB-7734 в регламенте REG-2024-118?", "relevant": ["d6"]},
]

def tokenize(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())

def embed(text, dim=64):
    # Для продакшена замените на живой эмбеддер
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def bm25_search(docs, query, k=K, k1=1.5, b=0.75):
    corpus = [tokenize(doc["text"]) for doc in docs]
    query_tokens = tokenize(query)
    n = len(corpus)
    # пустой корпус и пустой запрос не должны валить раннер: возвращаем пустую выдачу
    if n == 0 or not query_tokens:
        return []
    avgdl = sum(len(tokens) for tokens in corpus) / n
    df = {}
    for tokens in corpus:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    scored = []
    for doc, tokens in zip(docs, corpus):
        tf = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1
        score = 0.0
        for word in query_tokens:
            if word not in tf:
                continue
            idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * tf[word] * (k1 + 1) / (tf[word] + k1 * (1 - b + b * len(tokens) / avgdl))
        scored.append((-score, doc["id"]))
    # без пересечений (score=0) документ не выдаём: иначе мусорный запрос получает произвольную выдачу
    scored = [pair for pair in scored if -pair[0] > 0.0]
    scored.sort()
    return [doc_id for _, doc_id in scored[:k]]

def dense_search(docs, query, k=K):
    query_vec = embed(query)
    scored = []
    for doc in docs:
        doc_vec = embed(doc["text"])
        scored.append((-sum(a * b for a, b in zip(query_vec, doc_vec)), doc["id"]))
    scored.sort()
    return [doc_id for _, doc_id in scored[:k]]

def rrf_fuse(list_a, list_b, k_rrf=60):
    scores = {}
    for ranked in (list_a, list_b):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))

def recall_at_k(ranked, relevant, k=K):
    # кривая строка GOLD не должна ронять весь прогон: нет релевантных -> recall 0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

def mrr(ranked, relevant):
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked, relevant, k=K):
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0

# Три retrieval-пайплайна из модуля 04
PIPELINES = {
    "bm25": lambda docs, query, k: bm25_search(docs, query, k),
    "dense": lambda docs, query, k: dense_search(docs, query, k),
    "rrf": lambda docs, query, k: rrf_fuse(bm25_search(docs, query, k), dense_search(docs, query, k)),
}

def main():
    header = f"{'pipeline':8}{'recall@5':>10}{'mrr':>7}{'ndcg@5':>8}  вопрос"
    print(header)
    means = {}
    for name, run in PIPELINES.items():
        recalls, mrrs, ndcgs = [], [], []
        for item in GOLD:
            ranked = run(DOCS, item["question"], K)
            recall = recall_at_k(ranked, item["relevant"], K)
            rr = mrr(ranked, item["relevant"])
            ndcg = ndcg_at_k(ranked, item["relevant"], K)
            recalls.append(recall)
            mrrs.append(rr)
            ndcgs.append(ndcg)
            print(f"{name:8}{recall:10.2f}{rr:7.2f}{ndcg:8.2f}  {item['question']}")
        means[name] = (sum(recalls) / len(recalls), sum(mrrs) / len(mrrs), sum(ndcgs) / len(ndcgs))
    print("\nСредние по золотому набору:")
    for name, (recall, rr, ndcg) in means.items():
        print(f"{name:8}{recall:10.2f}{rr:7.2f}{ndcg:8.2f}")
    best = max(value[0] for value in means.values())
    passed = best >= THRESHOLD
    print(f"\nПорог recall@5 = {THRESHOLD:.2f}, лучший пайплайн = {best:.2f}: {'OK' if passed else 'FAIL'}")
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())

