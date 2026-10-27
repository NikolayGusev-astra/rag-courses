"""Курс RAG Engineering, глава 05-reranking. Версия 1.0.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 05-reranking.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================import math
import re
import time
import hashlib

def tokenize(text):
    return re.findall(r"[а-яё0-9a-z-]+", text.lower())

def bm25_search(docs, query, k):
    q_tokens = tokenize(query)
    if not docs or not q_tokens:
        return []
    doc_ids = [d["id"] for d in docs]
    n = len(doc_ids)
    # Индексируем заголовок вместе с текстом: заголовок несёт тему документа.
    doc_tokens = {d["id"]: tokenize(d["title"] + " " + d["text"]) for d in docs}
    avgdl = sum(len(t) for t in doc_tokens.values()) / n
    df = {}
    for qt in set(q_tokens):
        df[qt] = sum(1 for d in doc_ids if qt in set(doc_tokens[d]))
    k1, b = 1.5, 0.75
    scores = {}
    for d in doc_ids:
        toks = doc_tokens[d]
        dl = len(toks)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for qt in q_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[d] = s
    return sorted(doc_ids, key=lambda d: -scores[d])[:k]

def embed(text, dim=64):
    # для продакшена замените на живой эмбеддер
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec

def dense_search(docs, query, k):
    qv = embed(query)
    scored = []
    for d in docs:
        did = d["id"]
        dv = embed(d["title"] + " " + d["text"])
        dot = sum(a * b for a, b in zip(qv, dv))
        scored.append((dot, did))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]

def rrf_fuse(list_a, list_b, k_rrf=60):
    scores = {}
    for lst in (list_a, list_b):
        for rank, did in enumerate(lst, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores, key=lambda d: -scores[d])

def cross_score(query, doc_text):
    # для продакшена замените на bge-reranker или API-реранкер, контракт тот же: список кандидатов, список скоров
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    doc_tokens = set(tokenize(doc_text))
    hit = sum(1 for t in q_tokens if t in doc_tokens) / len(q_tokens)
    phrase = 0.5 if query.lower() in doc_text.lower() else 0.0
    return hit + phrase - 0.1 * len(doc_text) / 500

def rerank(query, doc_ids, docs, top_n=5):
    texts = {d["id"]: d["title"] + " " + d["text"] for d in docs}
    scored = [(cross_score(query, texts.get(d, "")), d) for d in doc_ids]
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top_n]]

def recall_at5(found_doc_ids, relevant_ids):
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return len(set(found_doc_ids[:5]) & rel) / len(rel)


DOCS = [
    {
        "id": 1,
        "title": "Отпуска",
        "text": "Ежегодный оплачиваемый отпуск предоставляется сотруднику по графику, утвержденному до начала календарного года. Перенос отпуска возможен по согласованию с руководителем, если это не нарушает производственный процесс."
    },
    {
        "id": 2,
        "title": "Отпускные",
        "text": "Отпускные выплачиваются не позднее чем за три календарных дня до начала отпуска. Расчет производится исходя из среднего заработка за последние двенадцать месяцев, а при увольнении компенсация выплачивается в день расчета."
    },
    {
        "id": 3,
        "title": "VPN",
        "text": "Удаленный доступ к корпоративной сети предоставляется только через VPN. Запрещается подключаться к рабочим ресурсам через публичные сети без шифрования, а также передавать логины и пароли третьим лицам."
    },
    {
        "id": 4,
        "title": "Инциденты VPN",
        "text": "При потере соединения VPN сотрудник обязан зафиксировать время сбоя, проверить настройки клиента и сообщить в службу поддержки. Если доступ нужен срочно, руководитель может запросить временный резервный канал."
    },
    {
        "id": 5,
        "title": "Бэкапы",
        "text": "Резервное копирование критичных данных выполняется ежедневно, а полные копии создаются не реже одного раза в неделю. Копии хранятся в защищенном хранилище, доступ к ним имеют только администраторы и ответственные за сервис."
    },
    {
        "id": 6,
        "title": "Восстановление",
        "text": "Восстановление данных из резервной копии проводится после согласования с владельцем сервиса. Сначала проверяют целостность копии на тестовом стенде, затем выполняют восстановление и фиксируют результат в журнале инцидентов."
    },
    {
        "id": 7,
        "title": "Доступы",
        "text": "Доступ к информационным системам выдается по заявке руководителя и только в объеме, необходимом для выполнения должностных обязанностей. При изменении роли сотрудника права пересматриваются в течение двух рабочих дней."
    },
    {
        "id": 8,
        "title": "Пароли",
        "text": "Пароли должны содержать не менее двенадцати символов, включая заглавные и строчные буквы, цифры и специальные знаки. Запрещено передавать пароли коллегам и использовать один пароль для разных систем."
    },
    {
        "id": 9,
        "title": "Командировки",
        "text": "Командировка оформляется на основании приказа руководителя. Сотрудник обязан согласовать маршрут, забронировать проживание и сохранить документы, подтверждающие расходы. Аванс выдается не позднее чем за два рабочих дня до выезда."
    },
    {
        "id": 10,
        "title": "Расходы",
        "text": "Расходы сотрудников возмещаются при наличии чеков и подтверждающих документов. Заявка на компенсацию подается в бухгалтерию в течение десяти рабочих дней после возвращения. Суммы сверх установленных лимитов требуют отдельного согласования."
    }
]


GOLD = [
    ("Когда сотрудник может перенести ежегодный отпуск?", [1]),
    ("Как часто делается резервное копирование критичных данных?", [5]),
    ("Как получить доступ к информационной системе компании?", [7]),
    ("Как оформляется командировка и когда выдают аванс?", [9]),
    ("Как возмещаются расходы сотрудника после поездки?", [10]),
]


def main():
    rows = []
    for question, relevant in GOLD:
        fused = rrf_fuse(bm25_search(DOCS, question, 10), dense_search(DOCS, question, 10))[:10]
        before = recall_at5(fused[:5], relevant)
        t0 = time.perf_counter()
        reranked = rerank(question, fused, DOCS, top_n=5)
        ms = (time.perf_counter() - t0) * 1000.0
        after = recall_at5(reranked, relevant)
        rows.append((question, before, after, ms))

    w = 62
    print(f"{'вопрос':<{w}} {'recall@5 до':>12} {'после':>7} {'мс':>9}")
    print("-" * (w + 32))
    for question, before, after, ms in rows:
        print(f"{question[:w]:<{w}} {before:>12.2f} {after:>7.2f} {ms:>9.2f}")
    print("-" * (w + 32))
    n = len(rows)
    print(
        f"{'СРЕДНЕЕ':<{w}} "
        f"{sum(r[1] for r in rows) / n:>12.2f} "
        f"{sum(r[2] for r in rows) / n:>7.2f} "
        f"{sum(r[3] for r in rows) / n:>9.2f}"
    )


def rerank_with_transformers(query, doc_ids):
    """опционально, требует внешних пакетов"""
    raise RuntimeError('см. текст модуля: нужен transformers или API-реранкер')


def rerank_with_api(query, doc_ids):
    """опционально, требует внешних пакетов"""
    raise RuntimeError('см. текст модуля: нужен transformers или API-реранкер')


if __name__ == '__main__':
    main()

