"""Курс RAG Engineering, глава 04-gibridnyj-poisk. Версия 1.2.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 04-gibridnyj-poisk.py (оффлайн, чистый stdlib, если в главе не сказано иное).
"""


# ======================================================================

import hashlib
import math
import re


def tokenize(text):
    return re.findall(r"[а-яё0-9a-z-]+", (text or "").lower())


def bm25_search(docs, query, k):
    k1 = 1.5
    b = 0.75
    n = len(docs)
    if n == 0:
        return []

    tokenized = []
    for d in docs:
        tokens = tokenize(str(d.get("title", "")) + " " + str(d.get("text", "")))
        tokenized.append(tokens)

    dlen = [len(t) for t in tokenized]
    avglen = sum(dlen) / n if n else 0.0

    df = {}
    for tokens in tokenized:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    q_terms = tokenize(query)
    scores = {}
    for i, d in enumerate(docs):
        tf_counts = {}
        for term in tokenized[i]:
            tf_counts[term] = tf_counts.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            if term not in df:
                continue
            tf = tf_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf + k1 * (1.0 - b + b * (dlen[i] / avglen if avglen else 0.0))
            score += idf * (tf * (k1 + 1.0) / denom)
        scores[d["id"]] = score

    # нулевой скор это отсутствие пересечений: такие документы не выдаём,
    # иначе мусорный запрос получает произвольную выдачу с рейтингом
    ranked = sorted(
        ((doc_id, s) for doc_id, s in scores.items() if s > 0.0),
        key=lambda x: (-x[1], str(x[0])),
    )
    return [doc_id for doc_id, _ in ranked[:k]]


def embed(text, dim=64):
    # для продакшена замените на живой эмбеддер
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dim
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


def dense_search(docs, query, k):
    q_vec = embed(query)
    scored = []
    for d in docs:
        text = str(d.get("title", "")) + " " + str(d.get("text", ""))
        d_vec = embed(text)
        dot = sum(a * b for a, b in zip(q_vec, d_vec))
        scored.append((d["id"], dot))
    scored.sort(key=lambda x: (-x[1], str(x[0])))
    return [doc_id for doc_id, _ in scored[:k]]


def rrf_fuse(list_a, list_b, k_rrf=60):
    scores = {}
    for rank, doc_id in enumerate(list_a, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, doc_id in enumerate(list_b, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))
    return [doc_id for doc_id, _ in ranked]


DOCS = [
    {
        "id": "vacation",
        "title": "Регламент об отпусках",
        "text": "Каждый сотрудник имеет право на ежегодный оплачиваемый отпуск продолжительностью 28 календарных дней. График отпусков утверждается до 15 декабря на следующий год. Заявление подается не позднее чем за 14 дней до начала отдыха через HR-портал. Перенос допускается по согласованию с руководителем."
    },
    {
        "id": "vacation_pay",
        "title": "Расчет отпускных",
        "text": "Отпускные рассчитываются исходя из среднего дневного заработка за последние 12 месяцев. Выплата производится не позднее чем за три календарных дня до начала отпуска. Если срок нарушен, работодатель обязан выплатить компенсацию за задержку. Больничный в период отпуска продлевает отдых."
    },
    {
        "id": "vpn",
        "title": "Регламент доступа к VPN",
        "text": "Доступ к корпоративной сети извне предоставляется только через VPN. Заявку оформляют в сервис-деске, указывая обоснование и срок. Пароль меняется каждые 90 дней, многофакторная аутентификация обязательна. Запрещено передавать ключи и подключаться с личных устройств без согласования с ИБ."
    },
    {
        "id": "vpn_incidents",
        "title": "Порядок обработки инцидентов VPN",
        "text": "При недоступности VPN или массовых ошибках подключения необходимо немедленно создать инцидент в сервис-деске с пометкой «VPN-инцидент». Указывают время, площадку и коды ошибок. Дежурный инженер эскалирует проблему в ИБ и сетевую команду в течение 15 минут."
    },
    {
        "id": "backups",
        "title": "Регламент резервного копирования",
        "text": "Резервное копирование критичных систем выполняется ежедневно в 02:00, полные копии каждое воскресенье. Срок хранения инкрементальных копий 30 дней, полных 12 месяцев. Восстановление проверяется ежеквартально. Запрос на восстановление подается через сервис-деск с указанием даты."
    },
    {
        "id": "access",
        "title": "Регламент управления доступами",
        "text": "Доступ к информационным системам предоставляется по принципу минимально необходимых прав. Заявка согласуется с руководителем и владельцем системы. При переводе или увольнении права пересматриваются в течение одного рабочего дня. Пароли запрещено передавать коллегам."
    },
    {
        "id": "trips",
        "title": "Регламент командировок",
        "text": "Служебная командировка оформляется не позднее чем за пять рабочих дней до выезда. Суточные выплачиваются за каждый день поездки, включая выходные. Билеты и проживание бронируются через корпоративный портал. Отчет о командировке сдается в течение трех дней после возвращения."
    },
    {
        "id": "expenses",
        "title": "Регламент компенсации расходов",
        "text": "Компенсация расходов производится при наличии подтверждающих документов. Авансовый отчет подается в течение пяти рабочих дней. Лимит на такси в городе 1500 рублей за поездку. Представительские расходы согласуются заранее с финансовым директором."
    },
    {
        "id": "reg2024",
        "title": "Регламент REG-2024-118",
        "text": "Регламент REG-2024-118 устанавливает единый порядок согласования внутренних документов. Он вступает в силу с 1 января 2025 года. Ответственный за исполнение : отдел внутреннего контроля. Изменения вносятся приказом руководителя, а нарушения фиксируются в журнале."
    }
]


GOLD = [
    ("Сколько дней ежегодного отпуска и за сколько подать заявление?", ["vacation"]),
    ("Как подключиться к корпоративной сети извне и что для этого нужно?", ["vpn"]),
    ("Как часто выполняется резервное копирование и сколько хранятся копии?", ["backups"]),
    ("Как получить доступ к информационной системе и кто согласует?", ["access"]),
    ("Как оформляется командировка и в какой срок сдать отчёт?", ["trips"]),
]

ART_Q = "Что устанавливает регламент REG-2024-118?"
ART_REL = ["reg2024"]

def main():
    sums = [0.0, 0.0, 0.0]

    for q, rel in GOLD:
        b = bm25_search(DOCS, q, 10)
        d = dense_search(DOCS, q, 10)
        r = rrf_fuse(b, d)

        rb = len(set(rel) & set(b[:5])) / len(rel)
        rd = len(set(rel) & set(d[:5])) / len(rel)
        rr = len(set(rel) & set(r[:5])) / len(rel)

        sums[0] += rb
        sums[1] += rd
        sums[2] += rr

        print(f"{q}")
        print(f"  recall@5 bm25={rb:.2f} dense={rd:.2f} rrf={rr:.2f}")

    n = len(GOLD)
    print(f"\nСредние recall@5: bm25={sums[0]/n:.2f} dense={sums[1]/n:.2f} rrf={sums[2]/n:.2f}")

    # top-1 для артикульного вопроса: где точный редкий токен решает
    tb = bm25_search(DOCS, ART_Q, 1)[0]
    td = dense_search(DOCS, ART_Q, 1)[0]
    tr = rrf_fuse(bm25_search(DOCS, ART_Q, 10), dense_search(DOCS, ART_Q, 10))[0]
    print(f"Артикульный вопрос: top-1 bm25={tb} dense={td} rrf={tr}")

if __name__ == '__main__':
    main()

