# Продолжение обрезанного модуля через tkbk. Использование:
# python continue_module.py <slug> <outline-file>
# Проверяет хвост файла; если нет завершения (навигационной строки) — просит
# tkbk продолжить ровно с места обрыва и доклеивает. Максимум N итераций.
import sys, os
sys.path.insert(0, r"C:/Work/Assist/rag-course/harness")
from tkbk_gen import generate

COURSE = r"C:/Work/Assist/llm-inference-course"
END_MARK = "Навигация по курсу"

def main():
    slug, outline_path = sys.argv[1], sys.argv[2]
    outline = open(outline_path, encoding="utf-8").read()
    dst = os.path.join(COURSE, "modules", slug + ".md")
    text = open(dst, encoding="utf-8").read()
    for i in range(4):
        if END_MARK in text and len(text.split()) >= 2500:
            break
        tail = text[-3000:]
        prompt = (
            "Markdown-модуль технического курса оборван посередине. Ниже: (1) план "
            "модуля, (2) последние строки оборванного файла. Продолжи текст РОВНО с "
            "места обрыва: не повторяй уже написанное, не начинай заново. Если обрыв "
            "внутри кода — сначала заверши текущую функцию, затем остаток кода и все "
            "оставшиеся разделы плана до конца формата (## Как проверить, ## Сводка, "
            "финальная строка навигации '**Навигация по курсу:** ...'). Выдавай только "
            "продолжение markdown, без пояснений и без ограждений кодового блока "
            "сверх необходимых.\n\n# План модуля\n" + outline +
            "\n\n# Последние строки файла (место обрыва)\n" + tail
        )
        cont = generate(prompt, system="Ты дописываешь технический курс на русском. Продолжай бесшовно.", max_tokens=16000)
        text = text.rstrip() + "\n" + cont.strip() + "\n"
        open(dst, "w", encoding="utf-8", newline="\n").write(text)
        print(f"iteration {i+1}: now {len(text.split())} words, end-mark={'yes' if END_MARK in text else 'no'}")
    print("final words:", len(text.split()))

if __name__ == "__main__":
    main()
