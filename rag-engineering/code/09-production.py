"""Курс RAG Engineering, глава 09-production. Версия 1.0.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 09-production.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict

VERSION = "reg-index-2024-11-01"
EMBED_DIM = 64
_QUERY_EMBED_CACHE = OrderedDict()
_QUERY_EMBED_MAX = 128
_ANSWER_CACHE = {}
_CACHE_TTL = 60.0

USERS = {
    "alice": {"hr", "finance"},
    "bob": {"it"},
    "carol": {"finance"},
}

DOCS = [
    {
        "id": "vacation-hr",
        "department": "hr",
        "text": "Отпуск оформляется за две недели. Сотрудник подает заявление в HR-системе, руководитель согласует даты. Перенос отпуска возможен по согласованию.",
    },
    {
        "id": "vpn-it",
        "department": "it",
        "text": "VPN подключается через корпоративный клиент. Доступ выдается отделом IT после заявки. Пароль нельзя передавать коллегам.",
    },
    {
        "id": "backup-it",
        "department": "it",
        "text": "Резервные копии служебных систем создаются ежедневно. Восстановление данных выполняет IT-дежурный по заявке. Проверка бэкапов проводится раз в месяц.",
    },
    {
        "id": "access-it",
        "department": "it",
        "text": "Доступы к внутренним системам выдаются по принципу минимальных прав. Заявка на доступ проходит согласование руководителя и IT.",
    },
    {
        "id": "travel-finance",
        "department": "finance",
        "text": "Командировки согласует руководитель. Финансовый отдел компенсирует проезд и проживание по документам.",
    },
    {
        "id": "compensation-finance",
        "department": "finance",
        "text": "Компенсации выплачиваются после проверки документов. Финансовый отдел принимает заявления в установленной форме.",
    },
    {
        "id": "REG-2024-118",
        "department": "finance",
        "text": "Регламент REG-2024-118 описывает редкий артикул 77-К для служебных закупок. Заявки с этим артикулом проверяет финансовый контроль.",
    },
]


def tokenize(text):
    return re.findall(r"[a-zа-яё0-9-]+", text.lower())


def normalize_query(query):
    return " ".join(query.lower().strip().split())


# Детерминированный hash-эмбеддер для офлайн-запуска.
# для продакшена замените на живой эмбеддер
def embed(text, dim=EMBED_DIM):
    key = (text, dim)
    if key in _QUERY_EMBED_CACHE:
        _QUERY_EMBED_CACHE.move_to_end(key)
        return list(_QUERY_EMBED_CACHE[key])

    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(dim):
            byte = digest[index % len(digest)]
            position = (byte + index * 13) % dim
            vec[position] += (byte / 255.0) - 0.5

    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0.0:
        vec = [value / norm for value in vec]

    _QUERY_EMBED_CACHE[key] = tuple(vec)
    if len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_MAX:
        _QUERY_EMBED_CACHE.popitem(last=False)
    return vec


def cosine(left, right):
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def allowed_departments(user):
    return set(USERS.get(user, set()))


def bm25_search(query, k, allowed):
    # Фильтрация до скоринга: в индекс попадают только разрешенные отделы.
    chunks = [doc for doc in DOCS if doc["department"] in allowed]
    if not chunks:
        return []

    tokenized = [tokenize(doc["text"]) for doc in chunks]
    avgdl = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    df = {}
    for tokens in tokenized:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    query_terms = tokenize(query)
    n = len(chunks)
    k1 = 1.5
    b = 0.75
    scored = []

    for doc, tokens in zip(chunks, tokenized):
        tf = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1

        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1.0 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * (tf[term] * (k1 + 1.0)) / denom

        if score > 0.0:
            scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def dense_search(query, k, allowed):
    # Выключатель для лабы: имитация отказа векторной ветки.
    if os.environ.get("RAG_DENSE") == "off":
        raise RuntimeError("плотный поиск отключен переменной RAG_DENSE=off")
    # Плотный поиск видит только документы из разрешенных отделов.
    chunks = [doc for doc in DOCS if doc["department"] in allowed]
    if not chunks:
        return []

    query_vec = embed(query)
    scored = []
    for doc in chunks:
        doc_vec = embed(doc["text"])
        score = cosine(query_vec, doc_vec)
        if score > 0.0:
            scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def rrf_fuse(bm25_hits, dense_hits, limit=4, rrf_k=60):
    scores = {}
    docs_by_id = {}

    for rank, (doc, _score) in enumerate(bm25_hits, start=1):
        doc_id = doc["id"]
        docs_by_id[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (doc, _score) in enumerate(dense_hits, start=1):
        doc_id = doc["id"]
        docs_by_id[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_id[doc_id] for doc_id, _score in ordered[:limit]]


def _response_cache_key(query, allowed):
    return (normalize_query(query), tuple(sorted(allowed)), VERSION)


def answer(query, user):
    allowed = allowed_departments(user)
    key = _response_cache_key(query, allowed)
    now = time.monotonic()
    cached = _ANSWER_CACHE.get(key)

    if cached is not None and cached[0] > now:
        value = cached[1]
        return {
            "answer": value["answer"],
            "citations": [dict(citation) for citation in value["citations"]],
            "degraded": value["degraded"],
        }

    if cached is not None:
        del _ANSWER_CACHE[key]

    k = 4
    bm25_hits = bm25_search(query, k, allowed)
    degraded = False

    try:
        dense_hits = dense_search(query, k, allowed)
    except Exception:
        # Деградация: плотный поиск недоступен, остаемся на BM25.
        dense_hits = []
        degraded = True

    fused = rrf_fuse(bm25_hits, dense_hits, limit=k)
    citations = [{"id": doc["id"], "text": doc["text"]} for doc in fused]

    if not fused:
        answer_text = "По вашему запросу ничего не найдено в доступных отделах."
    else:
        # Промпт-заглушка собирается без вызова LLM.
        prompt_lines = ["Ответьте на вопрос по фрагментам ниже."]
        for index, doc in enumerate(fused, start=1):
            prompt_lines.append(f"[{index}] {doc['text']}")
        prompt = "\n".join(prompt_lines)
        answer_text = "\n\n".join(
            f"[{index}] {doc['text']}"
            for index, doc in enumerate(fused, start=1)
        )

    result = {
        "answer": answer_text,
        "citations": citations,
        "degraded": degraded,
    }

    if not degraded:
        _ANSWER_CACHE[key] = (
            time.monotonic() + _CACHE_TTL,
            {
                "answer": result["answer"],
                "citations": [dict(citation) for citation in result["citations"]],
                "degraded": result["degraded"],
            },
        )

    return result


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="RAG service")

    @app.get("/search")
    def search_endpoint(query: str, user: str):
        result = answer(query, user)
        return {"citations": result["citations"]}

    @app.get("/answer")
    def answer_endpoint(query: str, user: str):
        result = answer(query, user)

        async def event_stream():
            yield "event: citations\n"
            payload = {"citations": result["citations"]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            text = result["answer"]
            step = 48
            for start in range(0, len(text), step):
                piece = text[start:start + step]
                yield "event: token\n"
                yield f"data: {json.dumps({'token': piece}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main():
    print("Офлайн-проверка RAG-сервиса")
    print("Версия индекса:", VERSION)

    normal = answer("Как оформить отпуск?", "alice")
    print("Обычный вопрос, цитаты:", [citation["id"] for citation in normal["citations"]])
    print("Фрагмент ответа:", normal["answer"][:140].replace("\n", " "))

    alice_backup = answer("резервные копии", "alice")
    bob_backup = answer("резервные копии", "bob")
    alice_ids = [citation["id"] for citation in alice_backup["citations"]]
    bob_ids = [citation["id"] for citation in bob_backup["citations"]]
    print("alice, бэкапы:", alice_ids)
    print("bob, бэкапы:", bob_ids)
    print("Права alice, it-документ скрыт:", "backup-it" not in alice_ids)
    print("Права bob, it-документ виден:", "backup-it" in bob_ids)

    cache_query = "Как получить доступ к VPN?"
    cache_user = "bob"
    start = time.perf_counter()
    answer(cache_query, cache_user)
    first_time = time.perf_counter() - start
    start = time.perf_counter()
    answer(cache_query, cache_user)
    second_time = time.perf_counter() - start
    cache_key = _response_cache_key(cache_query, allowed_departments(cache_user))
    print(f"Кэш ответа: первый вызов {first_time:.6f} с, второй {second_time:.6f} с")
    print("Ключ ответа в кэше:", cache_key in _ANSWER_CACHE)

    global dense_search
    original_dense_search = dense_search

    def broken_dense_search(query, k, allowed):
        raise RuntimeError("dense_search временно недоступен")

    dense_search = broken_dense_search
    try:
        degraded_result = answer("уникальный запрос про сбой плотного поиска", "bob")
    finally:
        dense_search = original_dense_search
    print("Деградация, degraded:", degraded_result["degraded"])

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("Офлайн-часть прошла, серверная проверка пропущена: fastapi или TestClient недоступен.")
        return

    try:
        client = TestClient(build_app())
        search_response = client.get("/search", params={"query": "отпуск", "user": "alice"})
        print("GET /search статус:", search_response.status_code)
        print("GET /search JSON:", search_response.json())

        answer_response = client.get("/answer", params={"query": "отпуск", "user": "alice"})
        print("GET /answer статус:", answer_response.status_code)
        stream_lines = answer_response.text.splitlines()
        print("Первые строки стрима:", stream_lines[:8])
    except Exception as exc:
        print("Серверная проверка не выполнена:", exc)


if __name__ == "__main__":
    main()


# load_test.py: 100 запросов к /search, замер латентности каждого
# Перед запуском подними сервис в другом терминале:
# uvicorn rag_service:build_app --factory
import json
import time
import urllib.request
from urllib.parse import quote

BASE = "http://127.0.0.1:8000/search"
QUERIES = ["как оформить отпуск", "где хранятся бэкапы", "артикул из REG-2024-118"]


def one(query):
    # GET с параметрами: тот же маршрут, что тестировался через TestClient
    url = f"{BASE}?query={quote(query)}&user=alice"
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=30) as resp:
        resp.read()  # дочитываем тело, иначе замер получится оптимистичным
    return (time.perf_counter() - start) * 1000  # миллисекунды


one(QUERIES[0])  # прогрев, в статистику не идёт

latencies = []
for i in range(100):
    latencies.append(one(QUERIES[i % len(QUERIES)]))

latencies.sort()
median = latencies[len(latencies) // 2]
p95 = latencies[int(0.95 * len(latencies))]  # индекс 95 из ста элементов

print(f"медиана: {median:.1f} мс")
print(f"p95: {p95:.1f} мс")

