"""Курс RAG Engineering, глава 10-federated. Версия 1.2.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 10-federated.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================

# file: federated_search.py
# Федеративный поиск по многим источникам: коннекторы, health, fusion, provenance.
# Оффлайн, чистый stdlib. Имена источников и корпуса учебные.
import json
import os
import time

# Корпус источников. В бою под этим слоем стоят REST API, файлы, чужие индексы.
CORPORA = {
    "tracker": [
        {"id": "INT-6515", "title": "Стенд ИИ-платформы", "text": "Стенд разворачивается через ansible, доступ по VPN, токен в vault. Инструкции по поднятию стенда в приложенном документе."},
        {"id": "INT-6601", "title": "Миграция синкера", "text": "Синкер переносит пользователей и группы. При конфликте DN запись пишется в отчёт расхождений, миграция продолжается."},
    ],
    "wiki": [
        {"id": "wiki/rag-faq", "title": "FAQ по поиску", "text": "Гибридный поиск сочетает лексический и векторный сигналы. RRF сливает списки без калибровки скоров."},
        {"id": "wiki/onboarding", "title": "Онбординг инженера", "text": "Первая неделя: доступы, окружение, чтение регламентов. Наставник проверяет критерии готовности."},
    ],
    "local_docs": [
        {"id": "file/runbook-vpn", "title": "Ранбук VPN", "text": "Если VPN не подключается, проверить сертификат и время на машине. Время, сбитое на час, ломает рукопожатие TLS."},
    ],
}


class Connector:
    """Коннектор источника: поиск, живость, происхождение."""

    def __init__(self, name, docs):
        self.name = name
        self._docs = docs

    def health(self):
        # В бою здесь HEAD-запрос или чтение индикатора. Дешёвая проверка,
        # которая говорит координатору: источник жив, ходи.
        return {"source": self.name, "available": True}

    def search(self, query, k=3):
        # Учебный лексический поиск. Контракт важнее содержимого:
        # на входе строка запроса, на выходе фрагменты с origin.
        tokens = query.lower().split()
        scored = []
        for doc in self._docs:
            text = (doc["title"] + " " + doc["text"]).lower()
            score = sum(1 for t in tokens if t in text)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        hits = []
        for score, doc in scored[:k]:
            hits.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "origin": self.name,
            })
        return hits


def build_connectors():
    return [Connector(name, docs) for name, docs in CORPORA.items()]


# file: federated_search.py (продолжение)
import itertools


def rrf_fuse(rankings, k=60):
    """RRF из модуля 04: сливаем выдачи источников в один список."""
    scores = {}
    for ranking in rankings:
        for pos, hit in enumerate(ranking, start=1):
            key = hit["id"]
            scores.setdefault(key, {"hit": hit, "score": 0.0})
            scores[key]["score"] += 1.0 / (k + pos)
    ranked = sorted(scores.values(), key=lambda item: -item["score"])
    return [item["hit"] for item in ranked]


class Coordinator:
    def __init__(self, connectors):
        self._connectors = {c.name: c for c in connectors}
        # Спросить health один раз на старте и кэшировать на короткое время:
        # в бою health-опрос дешёвый, но не бесплатный.
        self._health = {}
        self.refresh_health()

    def refresh_health(self):
        self._health = {}
        for name, conn in self._connectors.items():
            try:
                self._health[name] = conn.health()
            except Exception as exc:
                # источник недоступен - это штатная ситуация, а не ошибка поиска
                self._health[name] = {"source": name, "available": False, "reason": str(exc)}

    def available_sources(self):
        return [name for name, state in self._health.items() if state.get("available")]

    def search(self, query, k=3):
        t0 = time.perf_counter()
        sources_used, rankings = [], []
        for name in self.available_sources():
            try:
                hits = self._connectors[name].search(query, k=k)
            except Exception:
                # поиск в живом источнике может упасть на самом запросе:
                # деградируем на остальных, а не на всём ответе
                self._health[name] = {"source": name, "available": False, "reason": "search failed"}
                continue
            if hits:
                sources_used.append(name)
                rankings.append(hits)
        fused = rrf_fuse(rankings)[:k] if rankings else []
        return {
            "results": fused,
            "sources_used": sources_used,
            "sources_absent": [name for name in self._health if name not in sources_used],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


if __name__ == "__main__":
    coordinator = Coordinator(build_connectors())
    for query in ["как поднять стенд", "vpn не подключается", "синкер миграция"]:
        answer = coordinator.search(query)
        print(f"запрос: {query}")
        for hit in answer["results"]:
            print(f"  [{hit['origin']}] {hit['id']}: {hit['title']}")
        print(f"  источники: {answer['sources_used']}, "
              f"нет: {answer['sources_absent']}, {answer['latency_ms']} мс")


# file: federated_search.py (продолжение)
class WeightedCoordinator(Coordinator):
    """Координатор с весами источников и телеметрией попаданий."""

    def __init__(self, connectors):
        super().__init__(connectors)
        self._weights = {name: 1.0 for name in self._connectors}
        self._hits = {name: 0 for name in self._connectors}
        self._queries = 0

    def search(self, query, k=3):
        rankings = []
        for name in self.available_sources():
            try:
                hits = self._connectors[name].search(query, k=k)
            except Exception:
                continue
            if hits:
                rankings.append(hits)
        weighted = []
        for ranking in rankings:
            weighted.append([
                {**hit, "origin": hit["origin"]} for hit in ranking
            ])
        scores = {}
        for ranking in weighted:
            origin = ranking[0]["origin"] if ranking else None
            weight = self._weights.get(origin, 1.0)
            for pos, hit in enumerate(ranking, start=1):
                entry = scores.setdefault(hit["id"], {"hit": hit, "score": 0.0})
                entry["score"] += weight / (60 + pos)
        fused = [item["hit"] for item in sorted(scores.values(), key=lambda item: -item["score"])]
        fused = fused[:k]
        # Телеметрия: источник считается полезным, если его фрагмент попал в топ.
        top_origins = {hit["origin"] for hit in fused}
        self._queries += 1
        for name in top_origins:
            self._hits[name] = self._hits.get(name, 0) + 1
        return {"results": fused, "sources_used": sorted(top_origins)}

    def report(self):
        return {
            "queries": self._queries,
            "hit_share": {
                name: (self._hits[name] / self._queries if self._queries else 0.0)
                for name in self._connectors
            },
        }

