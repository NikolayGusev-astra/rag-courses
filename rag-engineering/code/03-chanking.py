"""Курс RAG Engineering, глава 03-chanking. Версия 1.0.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 03-chanking.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================import hashlib
import math
import re


DOCS = [
    {
        "id": "vacation_policy",
        "title": "Регламент отпусков",
        "text": (
            "Каждый сотрудник имеет право на ежегодный оплачиваемый отпуск "
            "продолжительностью двадцать восемь календарных дней. График отпусков "
            "утверждается руководителем подразделения до начала календарного года. "
            "Заявление на отпуск подается не позднее чем за четырнадцать календарных "
            "дней. При необходимости даты можно перенести по согласованию с "
            "руководителем. Неиспользованные дни сохраняются в соответствии с "
            "трудовым законодательством."
        ),
    },
    {
        "id": "vacation_pay",
        "title": "Расчет отпускных",
        "text": (
            "Отпускные рассчитываются исходя из среднего заработка за двенадцать "
            "календарных месяцев, предшествующих месяцу начала отпуска. В расчет "
            "включаются оклад, регулярные премии и иные выплаты, предусмотренные "
            "системой оплаты труда. Выплата производится не позднее чем за три "
            "календарных дня до начала отпуска. При изменении дат отпуска бухгалтерия "
            "пересчитывает сумму. Вопросы по расчету направляйте в бухгалтерию."
        ),
    },
    {
        "id": "vpn_rules",
        "title": "Использование VPN",
        "text": (
            "Корпоративный VPN используется только для служебных задач и доступа к "
            "внутренним ресурсам компании. Подключаться разрешено с рабочих устройств, "
            "прошедших проверку безопасности. Передавать учетные данные и "
            "конфигурационные файлы третьим лицам запрещено. При потере устройства или "
            "подозрении на компрометацию немедленно сообщите в службу безопасности. "
            "Отключайте VPN после завершения работы, если постоянное соединение не "
            "требуется."
        ),
    },
    {
        "id": "vpn_incidents",
        "title": "Инциденты VPN",
        "text": (
            "При невозможности подключения к VPN сначала проверьте интернет-соединение "
            "и актуальность клиента. Если ошибка сохраняется, создайте заявку в "
            "сервис-деске, указав время, устройство и текст сообщения. Не пытайтесь "
            "самостоятельно изменять сертификаты или обходить ограничения. Служба "
            "безопасности анализирует инциденты и уведомляет пользователя о дальнейших "
            "действиях. Критические случаи обрабатываются в приоритетном порядке."
        ),
    },
    {
        "id": "backup_policy",
        "title": "Резервное копирование",
        "text": (
            "Рабочие документы должны храниться в корпоративных сетевых хранилищах, где "
            "резервное копирование выполняется автоматически. Полные копии создаются "
            "ежедневно, а проверка восстановления проводится не реже одного раза в "
            "квартал. Локальные копии на ноутбуках не считаются надежным способом "
            "сохранения данных. При обнаружении потери файлов необходимо обратиться в "
            "сервис-деск и указать путь, дату и время последнего изменения."
        ),
    },
    {
        "id": "access_policy",
        "title": "Управление доступами",
        "text": (
            "Доступ к информационным системам предоставляется по принципу минимально "
            "необходимых прав. Заявка на новый доступ согласуется с руководителем и "
            "владельцем системы. Права пересматриваются при смене должности, переводе в "
            "другое подразделение и увольнении. Использовать чужие учетные записи "
            "запрещено. Пароли должны быть уникальными и надежными, а многофакторная "
            "аутентификация обязательна для критичных сервисов."
        ),
    },
    {
        "id": "travel_policy",
        "title": "Командировки",
        "text": (
            "Командировка оформляется на основании служебного задания и приказа "
            "руководителя. До поездки сотрудник согласует даты, маршрут и бюджет с "
            "непосредственным руководителем. Авансовый отчет и подтверждающие документы "
            "необходимо предоставить в бухгалтерию в течение пяти рабочих дней после "
            "возвращения. Суточные выплачиваются за каждый день командировки. Изменение "
            "маршрута допускается только по согласованию."
        ),
    },
    {
        "id": "travel_expenses",
        "title": "Расходы в командировке",
        "text": (
            "В командировке оплачиваются проезд, проживание и суточные в пределах "
            "утвержденных норм. Дополнительные расходы требуют предварительного "
            "согласования с руководителем. Сохраняйте чеки, билеты и счета: без "
            "подтверждающих документов компенсация не производится. Корпоративная "
            "банковская карта используется только для служебных целей. При утере карты "
            "или документов сразу уведомите бухгалтерию и руководителя."
        ),
    },
]


GOLD = [
    (
        "За сколько дней нужно подать заявление на ежегодный отпуск?",
        ["vacation_policy"],
    ),
    (
        "Как рассчитываются отпускные и когда их выплачивают?",
        ["vacation_pay"],
    ),
    (
        "Можно ли передавать коллегам конфигурацию корпоративного VPN?",
        ["vpn_rules"],
    ),
    (
        "Что делать при потере рабочих файлов и как часто создаются резервные копии?",
        ["backup_policy"],
    ),
    (
        "Как получить доступ к информационной системе и кто его согласует?",
        ["access_policy"],
    ),
    (
        "Какие документы нужны для отчета по командировочным расходам?",
        ["travel_expenses", "travel_policy"],
    ),
]


def chunk_fixed(text, size=160):
    if size <= 0:
        raise ValueError("size должен быть больше нуля")
    return [text[start:start + size] for start in range(0, len(text), size)]


def chunk_recursive(text, size=200):
    if size <= 0:
        raise ValueError("size должен быть больше нуля")

    chunks = []
    current = ""
    paragraphs = re.split(r"\n\s*\n", text)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > size:
                if current:
                    chunks.append(current)
                    current = ""
                for start in range(0, len(sentence), size):
                    chunks.append(sentence[start:start + size])
                continue

            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= size:
                current = candidate
            else:
                chunks.append(current)
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_overlap(text, size=160, overlap=40):
    if size <= 0:
        raise ValueError("size должен быть больше нуля")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap должен быть от 0 до size - 1")

    if not text:
        return []

    step = size - overlap
    chunks = []

    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break

    return chunks


def embed(text, dim=64):
    # для продакшена замените на живой эмбеддер, hash-версия чтобы код бежал оффлайн
    vector = [0.0] * dim
    tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dim
        vector[bucket] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]

    return vector


def search(chunks, query, k=10):
    query_vector = embed(query)
    scored = []

    for index, chunk in enumerate(chunks):
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        chunk_vector = embed(text)
        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_vector, chunk_vector)
        )
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:k]]


def recall_at_k(found_doc_ids, relevant_ids):
    return 1.0 if set(found_doc_ids).intersection(relevant_ids) else 0.0


def main():
    strategies = [
        ("fixed", lambda text: chunk_fixed(text, size=160)),
        ("recursive", lambda text: chunk_recursive(text, size=200)),
        ("overlap", lambda text: chunk_overlap(text, size=160, overlap=40)),
    ]

    rows = []

    for strategy_name, chunker in strategies:
        chunks = []

        for doc in DOCS:
            for piece in chunker(doc["text"]):
                chunks.append({
                    "doc_id": doc["id"],
                    "text": piece,
                })

        recall_values = []

        for question, relevant_ids in GOLD:
            found = search(chunks, question, k=10)
            found_doc_ids = [
                item["doc_id"] if isinstance(item, dict) else item
                for item in found
            ]
            recall_values.append(recall_at_k(found_doc_ids, relevant_ids))

        average_recall = sum(recall_values) / len(recall_values)
        rows.append((strategy_name, len(chunks), average_recall))

    print(f"{'стратегия':<12} | {'чанков':>7} | {'recall@10':>9}")
    print("-" * 36)

    for strategy_name, chunk_count, average_recall in rows:
        print(
            f"{strategy_name:<12} | {chunk_count:>7} | "
            f"{average_recall:>9.2f}"
        )


if __name__ == "__main__":
    main()

