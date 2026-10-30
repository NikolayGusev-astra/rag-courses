# Генерация одного модуля курса через tkbk (deepseek-v4.1-flash).
# Использование: python gen_module.py <slug> <title> <outline-file>
# outline - текстовый файл с подробным планом модуля (темы, формулы, задания).
import sys, os
sys.path.insert(0, r"C:/Work/Assist/rag-course/harness")
from tkbk_gen import generate

COURSE = r"C:/Work/Assist/llm-inference-course"

def main():
    slug, title, outline_path = sys.argv[1], sys.argv[2], sys.argv[3]
    outline = open(outline_path, encoding="utf-8").read()
    spec = open(os.path.join(COURSE, "specs", "module-spec.md"), encoding="utf-8").read()
    system = ("Ты автор технического курса на русском языке. Пиши сразу, без долгих "
              "размышлений. Выдавай только markdown-содержимое модуля, без пояснений.")
    prompt = (spec + "\n\n# Модуль для написания\nslug: " + slug +
              "\nНазвание: " + title + "\n\n# Подробный план модуля\n" + outline +
              "\n\nНапиши полный модуль по формату и плану. Код полный и рабочий на CPU.")
    # Модули 3-6k слов: 524-k токен на вызов мало; бьём на 2 вызова (上半/下半) при
    # большом объёме. Начинаем с одного вызова max 16000, при обрыве — продолжение.
    out = generate(prompt, system=system, max_tokens=16000)
    dst = os.path.join(COURSE, "modules", slug + ".md")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write(out.strip() + "\n")
    print("written", dst, len(out.split()), "слов")

if __name__ == "__main__":
    main()
