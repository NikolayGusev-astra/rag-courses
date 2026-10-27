"""Курс RAG Engineering, глава 01-mini-rag. Версия 1.1.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 01-mini-rag.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================

import hashlib
import math
import re


def embed(text: str, dim: int = 96) -> list[float]:
    # Детерминированный hash-эмбеддер для учебного примера.
    # Для продакшена замените на живой эмбеддер.
    vector = [0.0] * dim
    tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        # Берем несколько позиций из SHA-256 и раскладываем токен по вектору.
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 8, 2):
            position = int.from_bytes(digest[offset:offset + 2], "big") % dim
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            vector[position] += sign

    # L2-нормализация превращает скалярное произведение в косинус.
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def chunk(text, max_chars=220):
    # Делим текст на предложения, затем собираем их в чанки.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # Слишком длинное предложение режем по словам.
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = (piece + " " + word).strip()
                if len(candidate) > max_chars and piece:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
            continue
        candidate = (current + " " + sentence).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def build_index(docs):
    # Создаем плоский список чанков с метаданными и векторами.
    index = []
    for doc in docs:
        for number, piece in enumerate(chunk(doc["text"]), start=1):
            index.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "chunk_id": number,
                "text": piece,
                "vector": embed(piece),
            })
    return index


def search(query, index, k=3):
    # Векторы нормализованы, поэтому косинус равен скалярному произведению.
    query_vector = embed(query)
    hits = []
    for item in index:
        score = sum(a * b for a, b in zip(query_vector, item["vector"]))
        hits.append({**item, "score": score})
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:k]


def build_prompt(query, hits):
    # Собираем промпт с нумерованным контекстом и требованием цитировать источники.
    lines = [
        "Ты помощник по внутренним регламентам компании.",
        "Отвечай только по блоку КОНТЕКСТ и не добавляй внешние знания.",
        "",
        "КОНТЕКСТ:",
    ]
    for number, hit in enumerate(hits, start=1):
        lines.append(f"[{number}] {hit['title']}")
        lines.append(hit["text"])
        lines.append("")
    lines.extend([
        "ВОПРОС:",
        query,
        "",
        "ПРАВИЛА ОТВЕТА:",
        "1. Опирайся только на блок КОНТЕКСТ.",
        "2. После каждого утверждения ставь ссылку вида [1].",
        "3. Если данных не хватает, прямо скажи, что ответа в контексте нет.",
    ])
    return "\n".join(lines)


DOCS = [
    {"id": "d1", "title": "Отпуска: ежегодный отпуск", "text": "Ежегодный оплачиваемый отпуск составляет двадцать восемь календарных дней. Даты отпуска согласует руководитель отдела с учетом графика. Заявление подают через кадровую систему не позднее чем за четырнадцать дней."},
    {"id": "d2", "title": "Отпуска: перенос и компенсация", "text": "Перенести отпуск можно по согласованию с руководителем и отделом кадров. Заменять отпуск денежной компенсацией запрещено, кроме случаев увольнения. Накопленные дни отображаются в личном кабинете."},
    {"id": "d3", "title": "VPN: подключение", "text": "Корпоративный VPN используют только для рабочих ресурсов. Подключение выдают сотрудникам после подтверждения учетной записи службой поддержки. Личные устройства подключать к VPN запрещено."},
    {"id": "d4", "title": "VPN: потеря устройства", "text": "При потере ноутбука с VPN сотрудник немедленно звонит в службу поддержки. Доступ к VPN блокируют, а инцидент передают в информационную безопасность. Самостоятельно восстанавливать подключение нельзя."},
    {"id": "d5", "title": "Бэкапы: расписание", "text": "Резервные копии рабочих систем создаются ежедневно. Полные копии формируются раз в неделю и хранятся отдельно от основного контура. Восстановление данных выполняет дежурная смена."},
    {"id": "d6", "title": "Бэкапы: проверка", "text": "Служба эксплуатации ежемесячно проверяет выборочное восстановление файлов. Результаты проверки фиксируют в журнале. При ошибке создают заявку и повторяют резервное копирование."},
    {"id": "d7", "title": "Доступы: выдача и отзыв", "text": "Доступ к внутренним системам выдают по заявке руководителя. Права должны соответствовать рабочим обязанностям. При увольнении или смене роли доступ отзывают в тот же рабочий день."},
    {"id": "d8", "title": "Командировки: оформление", "text": "Командировку согласует руководитель до покупки билетов. Сотрудник оформляет поездку в системе и прикладывает маршрут. Расходы возмещают по подтверждающим документам после возвращения."},
]


def main():
    # Индексируем документы один раз и выполняем два демонстрационных запроса.
    index = build_index(DOCS)
    queries = [
        "Сколько дней отпуска положено и как подать заявление?",
        "Что делать при потере ноутбука с VPN?",
    ]
    for query_number, query in enumerate(queries, start=1):
        print(f"\nЗАПРОС {query_number}: {query}")
        hits = search(query, index, k=3)
        for rank, hit in enumerate(hits, start=1):
            print(f"{rank}. [{hit['doc_id']}] {hit['title']} | score={hit['score']:.4f}")
        if query_number == 1:
            print("\nГОТОВЫЙ ПРОМПТ ДЛЯ ПЕРВОГО ЗАПРОСА:")
            print(build_prompt(query, hits))


if __name__ == "__main__":
    main()

