import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "data" / "qa"

INPUT_FILE = QA_DIR / "qa_pairs_2026.json"
OUTPUT_FILES = {
    "en": QA_DIR / "qa_pairs_en_2026.json",
    "ru": QA_DIR / "qa_pairs_ru_2026.json",
}


def build_language_file(data, language):
    source_key = "question_en" if language == "en" else "question"
    language_name = "English" if language == "en" else "Russian"
    qa_pairs = []

    for qa in data["qa_pairs"]:
        language_qa = {
            "id": qa["id"],
            "difficulty": qa["difficulty"],
            "category": qa.get("category", ""),
            "question": qa[source_key],
            "answer": qa["answer"],
        }
        if "answer_detail" in qa:
            language_qa["answer_detail"] = qa["answer_detail"]
        qa_pairs.append(language_qa)

    return {
        "description": f"{language_name} QA pairs for Приложение 14 (2026) — Kyrgyz Republic Program-Based Budget",
        "language": language,
        "source_file": data.get("source_file", ""),
        "notes": data.get("notes", ""),
        "qa_pairs": qa_pairs,
    }


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    for language, output_file in OUTPUT_FILES.items():
        language_data = build_language_file(data, language)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(language_data, f, ensure_ascii=False, indent=2)
        print(f"{language.upper()} questions saved to {output_file.name} ({len(language_data['qa_pairs'])} questions)")


if __name__ == "__main__":
    main()
