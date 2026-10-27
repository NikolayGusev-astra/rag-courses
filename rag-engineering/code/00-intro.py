"""Курс RAG Engineering, глава 00-intro. Версия 1.2.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 00-intro.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================

import hashlib
import math
import re


def tokenize(text: str) -> list[str]:
    # Простая токенизация для русского и латинского текста.
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def embed(text: str, dim: int = 64) -> list[float]:
    # Детерминированный hash-эмбеддер для офлайн-демонстрации.
    # для продакшена замените на живой эмбеддер.
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


CORPUS = [
    "Для удаленной работы используйте только корпоративный VPN. Личный VPN запрещен, а подключение к внутренним системам без него блокируется.",
    "Заявление на отпуск подается через портал не позднее чем за две недели. Согласование руководителя обязательно, перенос дат обсуждается отдельно.",
    "Резервные копии критичных данных создаются ежедневно. Раз в месяц дежурная команда проверяет восстановление на тестовом контуре.",
    "Доступ к общим папкам выдается по заявке в службу поддержки. Права пересматриваются при переводе сотрудника или смене проекта.",
    "Пароли к рабочим сервисам нельзя передавать коллегам. При подозрении на утечку смените пароль и сообщите в информационную безопасность.",
    "Инциденты с оборудованием фиксируются в общем журнале. После устранения сбоя исполнитель добавляет краткое описание причины."
]

DOC_TOKENS = [tokenize(document) for document in CORPUS]
DOC_VECTORS = [embed(document) for document in CORPUS]
DF = {}
for tokens in DOC_TOKENS:
    for token in set(tokens):
        DF[token] = DF.get(token, 0) + 1
IDF = {token: math.log(len(CORPUS) / frequency) for token, frequency in DF.items()}


def lex_score(query_tokens, doc_tokens):
    # Считаем tf*idf по точным совпадениям токенов запроса и документа.
    if not query_tokens or not doc_tokens:
        return 0.0
    score = 0.0
    for token in set(query_tokens):
        if token not in doc_tokens:
            continue
        tf = doc_tokens.count(token) / len(doc_tokens)
        qtf = query_tokens.count(token) / len(query_tokens)
        score += tf * IDF.get(token, 0.0) * qtf
    return score


def cosine(left, right):
    # Векторы уже нормализованы, поэтому косинус равен dot product.
    return sum(a * b for a, b in zip(left, right))


def lexical_search(query: str, limit: int = 2):
    query_tokens = tokenize(query)
    ranked = []
    for index, tokens in enumerate(DOC_TOKENS):
        ranked.append((lex_score(query_tokens, tokens), index))
    ranked.sort(reverse=True)
    return ranked[:limit]


def vector_search(query: str, limit: int = 2):
    query_vector = embed(query)
    ranked = []
    for index, document_vector in enumerate(DOC_VECTORS):
        ranked.append((cosine(query_vector, document_vector), index))
    ranked.sort(reverse=True)
    return ranked[:limit]


def main() -> None:
    queries = [
        "Как получить доступ к VPN из дома?",
        "Сколько дней отпуска можно взять без сохранения зарплаты?",
        "Где хранятся резервные копии и как часто их проверяют?",
    ]
    for query in queries:
        print(f"\nЗапрос: {query}")
        print("Лексический top-2:")
        for score, index in lexical_search(query):
            print(f"  {score:.3f} | {CORPUS[index]}")
        print("Векторный top-2:")
        for score, index in vector_search(query):
            print(f"  {score:.3f} | {CORPUS[index]}")


if __name__ == "__main__":
    main()

