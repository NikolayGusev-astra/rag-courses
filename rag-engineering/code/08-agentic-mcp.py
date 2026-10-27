"""Курс RAG Engineering, глава 08-agentic-mcp. Версия 1.1.0.
Репо: https://github.com/NikolayGusev-astra/rag-courses. Запуск: python 08-agentic-mcp.py (оффлайн, чистый stdlib, если в главе не сказано иное).
Особенности запуска: требует `pip install "mcp>=2"`; файл содержит сервер, клиент и фрагменты агента — запускай kb_server/kb_client по инструкции главы
"""


# ======================================================================

TOOLS = {
    "search_regulations": {
        "description": "Поиск по служебным регламентам",
        "run": lambda query: search_regulations(query),
    },
    "calculate": {
        "description": "Вычисление арифметического выражения",
        # eval даже с пустыми __builtins__ не песочница: выражение вида
        # ().__class__.__bases__[0].__subclasses__() вылезает наружу.
        # В боевом коде здесь строгий парсер арифметики, см. safe_calculate ниже.
        "run": lambda expression: safe_calculate(expression),
    },
}

def run_agent(question: str, max_steps: int = 4) -> str:
    """Цикл вызова инструментов с жёстким ограничением шагов."""
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        decision = llm_step(messages, tools=list(TOOLS.keys()))

        if decision["type"] == "final":
            return decision["text"]

        name = decision.get("tool")
        if name not in TOOLS:
            # Модель выбрала несуществующий инструмент
            messages.append({"role": "tool", "content": "Неизвестный инструмент"})
            continue

        result = TOOLS[name]["run"](**decision.get("args", {}))
        messages.append({"role": "tool", "name": name, "content": str(result)})

    return "Не удалось получить ответ за отведённое число шагов."


import ast
import operator

# Строгий калькулятор: только числа и четыре операции, никакой оценки строк.
# eval здесь не подходит: даже с пустыми __builtins__ выражение через
# ().__class__.__bases__[0].__subclasses__() добирается до os и выполняет код.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def safe_calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError(f"не поддерживается: {type(node).__name__}")

    return str(visit(tree))


# файл: kb_server.py
# MCP-сервер базы знаний: гибридный поиск на stdlib, транспорт stdio
import math
import re
from collections import Counter

from mcp.server import MCPServer

# Корпус служебных регламентов
DOCS = [
    {
        "id": "REG-2024-101",
        "title": "Регламент приёма заявок",
        "text": "Заявка регистрируется в течение одного рабочего дня. "
                "Ответственный сотрудник проверяет полноту данных и присваивает "
                "заявке статус «в работе». Неполные заявки возвращаются автору "
                "с перечнем недостающих полей.",
    },
    {
        "id": "REG-2024-104",
        "title": "Порядок обработки обращений",
        "text": "Обращения делятся на срочные и обычные. Срочные обрабатываются "
                "в день поступления, обычные в течение трёх рабочих дней. "
                "Каждое обращение получает номер и попадает в журнал учёта.",
    },
    {
        "id": "REG-2024-112",
        "title": "Согласование договоров",
        "text": "Проект договора направляется на согласование юридической службе "
                "и финансовому отделу. Срок согласования не превышает пяти рабочих "
                "дней. Замечания фиксируются в листе согласования.",
    },
    {
        "id": "REG-2024-118",
        "title": "Резервное копирование и восстановление данных",
        "text": "Резервные копии служебных систем создаются ежедневно. Полная "
                "копия формируется раз в неделю. Восстановление данных выполняется "
                "по заявке администратора, проверка целостности копии обязательна "
                "перед восстановлением.",
    },
    {
        "id": "REG-2024-121",
        "title": "Доступ к служебной информации",
        "text": "Доступ выдаётся по принципу минимальных прав. Заявка на доступ "
                "согласуется с руководителем подразделения. Права пересматриваются "
                "не реже одного раза в год и отзываются при увольнении сотрудника.",
    },
    {
        "id": "REG-2024-130",
        "title": "Передача документов в архив",
        "text": "Документы передаются в архив по описи. Опись подписывает "
                "ответственный за делопроизводство. Документы с истёкшим сроком "
                "хранения уничтожаются по акту в установленном порядке.",
    },
]

def tokenize(text):
    """Простая токенизация: нижний регистр, буквы и цифры."""
    return re.findall(r"[a-zа-яё0-9]+", text.lower())

def embed(text, dim=64):
    # для продакшена замените на живой эмбеддер
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = 0
        for ch in tok:
            h = (h * 31 + ord(ch)) % 1000003
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def bm25_search(query, docs, top_k=5, k1=1.5, b=0.75):
    """Разреженный поиск BM25 на stdlib."""
    corpus = {d["id"]: tokenize(d["title"] + " " + d["text"]) for d in docs}
    n = len(docs)
    avgdl = sum(len(t) for t in corpus.values()) / max(n, 1)

    df = Counter()
    for toks in corpus.values():
        for term in set(toks):
            df[term] += 1

    q_terms = tokenize(query)
    scores = {}
    for d in docs:
        toks = corpus[d["id"]]
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term in q_terms:
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * tf[term] * (k1 + 1) / denom
        scores[d["id"]] = score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, s in ranked[:top_k] if s > 0]

def dense_search(query, docs, top_k=5):
    """Плотный поиск по косинусу на hash-эмбеддере."""
    q = embed(query)
    scored = []
    for d in docs:
        v = embed(d["title"] + " " + d["text"])
        sim = sum(a * c for a, c in zip(q, v))
        scored.append((d["id"], sim))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:top_k]]

def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion: сливаем несколько ранжирований в одно."""
    scores = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def hybrid_search(query, top_k=3):
    """Гибрид: BM25 плюс плотный поиск, слияние через RRF."""
    sparse = bm25_search(query, DOCS, top_k=5)
    dense = dense_search(query, DOCS, top_k=5)
    return rrf_fuse([sparse, dense])[:top_k]

# Создаём сервер; имя видно клиенту в метаданных
mcp = MCPServer("kb")

@mcp.tool()
async def search(query: str) -> str:
    """Гибридный поиск по базе регламентов.

    Возвращает топ-3 документа в формате «id | title | фрагмент текста».
    """
    doc_ids = hybrid_search(query)
    if not doc_ids:
        return "Ничего не найдено."

    by_id = {d["id"]: d for d in DOCS}
    lines = []
    for doc_id in doc_ids:
        d = by_id[doc_id]
        snippet = d["text"][:80].replace("\n", " ")
        lines.append(f"{d['id']} | {d['title']} | {snippet}")
    return "\n".join(lines)

@mcp.tool()
async def get_document(doc_id: str) -> str:
    """Вернуть полный текст регламента по идентификатору, например REG-2024-118."""
    for d in DOCS:
        if d["id"] == doc_id.strip():
            return f"{d['id']} | {d['title']}\n\n{d['text']}"
    return f"Документ {doc_id} не найден."

if __name__ == "__main__":
    # stdio-транспорт по умолчанию; ничего лишнего в stdout не пишем
    mcp.run()


# файл: kb_client.py
# MCP-клиент: запускает kb_server.py по stdio и вызывает инструменты
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main() -> None:
    # Описываем, как поднять сервер: тот же интерпретатор плюс путь к файлу
    params = StdioServerParameters(
        command=sys.executable,
        args=["kb_server.py"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # Рукопожатие: обмен возможностями
            await session.initialize()

            # Смотрим, какие инструменты сервер объявил
            tools = await session.list_tools()
            print("Доступные инструменты:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # Вызываем поиск и печатаем текстовые блоки ответа
            result = await session.call_tool(
                "search", {"query": "резервное копирование данных"}
            )
            print("\nРезультат search:")
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    print(text)

if __name__ == "__main__":
    asyncio.run(main())

