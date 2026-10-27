"""Курс RAG Engineering, глава 02-vektornye-bd. Версия 1.0.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 02-vektornye-bd.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================"""Мини векторная БД на чистом stdlib: оффлайн, без зависимостей.
Запуск: python mini_vector_db.py
"""

import hashlib
import math
import time

# Размерность вектора, в продакшене совпадает с размерностью модели.
DIM = 64
# Размер корпуса и число повторов на один замер.
CORPUS_SIZE = 2000
REPEATS = 7


def embed(text, dim=DIM):
    """Детерминированный hash-эмбеддер для тестов и замеров.
    Для продакшена замените на живой эмбеддер: контракт тот же,
    текст на входе, список float на выходе.
    """
    vector = [0.0] * dim
    for token in text.lower().split():
        # blake2b стабилен между запусками и платформами
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % dim
        sign = 1.0 if (number >> 16) % 2 == 0 else -1.0
        vector[index] += sign
    # После нормировки косинус равен скалярному произведению
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def match_filter(payload, payload_filter):
    """Все пары ключ-значение фильтра должны совпасть с payload."""
    for key, expected in payload_filter.items():
        if payload.get(key) != expected:
            return False
    return True


class VectorDB:
    """In-memory векторная БД с полным перебором.
    Хранилище это dict: id -> (вектор, payload).
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        self._rows = {}
        self.last_scanned = 0

    def count(self):
        """Сколько записей сейчас в базе."""
        return len(self._rows)

    def upsert(self, id, vector, payload=None):
        """Вставляет или обновляет запись по id."""
        if len(vector) != self.dim:
            raise ValueError("размерность вектора не совпадает с базой")
        self._rows[id] = (list(vector), dict(payload or {}))

    def search(self, query_vector, k=3, filter=None):
        """Ищет top-k по косинусу, возвращает (результаты, латентность в мс).
        Фильтр по payload применяется внутри цикла, до подсчета score.
        """
        if len(query_vector) != self.dim:
            raise ValueError("размерность запроса не совпадает с базой")
        start = time.perf_counter()
        scored = []
        scanned = 0
        for doc_id, (vector, payload) in self._rows.items():
            scanned += 1
            if filter is not None and not match_filter(payload, filter):
                continue
            # Косинус для нормированных векторов это скалярное произведение
            score = math.fsum(a * b for a, b in zip(query_vector, vector))
            scored.append((score, doc_id, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [
            {"id": doc_id, "score": score, "payload": payload}
            for score, doc_id, payload in scored[:k]
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.last_scanned = scanned
        return top, elapsed_ms


TENANTS = ("acme", "globex", "initech", "umbrella")
TOPICS = ("billing", "auth", "search", "deploy")


def build_corpus(size=CORPUS_SIZE):
    """Детерминированный корпус: id, текст и payload."""
    rows = []
    for i in range(size):
        tenant = TENANTS[i % len(TENANTS)]
        topic = TOPICS[(i // len(TENANTS)) % len(TOPICS)]
        text = "{} документ {} про {} для {}".format(topic, i, topic, tenant)
        payload = {"tenant_id": tenant, "topic": topic, "doc_id": "doc-{}".format(i)}
        rows.append(("chunk-{}".format(i), text, payload))
    return rows


def load_db(rows, dim=DIM):
    """Считает эмбеддинги и складывает все в VectorDB."""
    db = VectorDB(dim=dim)
    for doc_id, text, payload in rows:
        db.upsert(doc_id, embed(text, dim=dim), payload)
    return db


def measure(db, query_vector, k, payload_filter=None):
    """Повторяет поиск и возвращает лучшую латентность в мс и top."""
    best_ms = None
    top = []
    for _ in range(REPEATS):
        top, latency_ms = db.search(query_vector, k=k, filter=payload_filter)
        if best_ms is None or latency_ms < best_ms:
            best_ms = latency_ms
    return best_ms, top


def main():
    db = load_db(build_corpus())
    print("векторов в базе:", db.count(), "размерность:", db.dim)

    query_vector = embed("billing документ 42 про billing для acme")
    tenant_filter = {"tenant_id": "acme"}
    print("фильтр:", tenant_filter)
    print()

    header = "{:<22}{:>4}{:>12}{:>18}".format(
        "сценарий", "k", "кандидатов", "латентность, мс"
    )
    print(header)
    print("-" * len(header))

    scenarios = (
        ("без фильтра", 3, None),
        ("без фильтра", 20, None),
        ("фильтр tenant_id", 3, tenant_filter),
        ("фильтр tenant_id", 20, tenant_filter),
    )
    for name, k, payload_filter in scenarios:
        latency_ms, top = measure(db, query_vector, k, payload_filter)
        print(
            "{:<22}{:>4}{:>12}{:>18.4f}".format(
                name, k, db.last_scanned, latency_ms
            )
        )
        best = top[0]
        print(
            "    top-1: {} score={:.4f} tenant={}".format(
                best["id"], best["score"], best["payload"]["tenant_id"]
            )
        )

    print()
    print("k меняет только стоимость сортировки, перебор от k не зависит")
    print("фильтр снижает число кандидатов и обычно снижает латентность")
    print("для продакшена замените embed на живой эмбеддер, dict на внешнюю БД")


if __name__ == "__main__":
    main()

