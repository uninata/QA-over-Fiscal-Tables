"""
Run all QA experiments: every model × every language × every context format.
No retrieval — just send context + question to each model, collect answers.

    python run_experiments.py
"""

import json
import os
import csv
import math
import time
import urllib.request
from pathlib import Path

from utils import ROOT, load_env, resolve_project_path, get_results_dir, get_run_id

QA_DIR = ROOT / "data" / "qa"
STATEMENTS_DIR = ROOT / "data" / "statements"
PROCESSED_DIR = ROOT / "data" / "processed"

QA_FILE = QA_DIR / "qa_pairs_2026.json"
QA_FILES = {
    "en": QA_DIR / "qa_pairs_en_2026.json",
    "ru": QA_DIR / "qa_pairs_ru_2026.json",
}
STATEMENTS_FILE = STATEMENTS_DIR / "statements_2026.json"
STATEMENT_FILES = {
    "en": STATEMENTS_DIR / "statements_en_2026.json",
    "ru": STATEMENTS_DIR / "statements_ru_2026.json",
}
MARKDOWN_FILE = PROCESSED_DIR / "budget_2026_full.md"

load_env()

STATEMENTS_FILE = resolve_project_path(os.environ.get("STATEMENTS_FILE"), STATEMENTS_DIR / "statements_2026.json")
STATEMENT_FILES = {
    "en": resolve_project_path(os.environ.get("STATEMENTS_EN_FILE"), STATEMENTS_DIR / "statements_en_2026.json"),
    "ru": resolve_project_path(os.environ.get("STATEMENTS_RU_FILE"), STATEMENTS_DIR / "statements_ru_2026.json"),
}
RESULTS_DIR = get_results_dir()
RUN_ID = get_run_id()


# ═══════════════════════════════════════════════════════════════
#   CONFIGURE EXPERIMENTS


_MC_URL = os.environ.get("METACENTRUM_BASE_URL", "")
_MC_KEY = os.environ.get("METACENTRUM_API_KEY", "")

OPENAI_MODELS = [
    # {"model": "gemma4",                          "base_url": _MC_URL, "api_key": _MC_KEY},
    # {"model": "deepseek-v3.2",                   "base_url": _MC_URL, "api_key": _MC_KEY},
    # {"model": "deepseek-v3.2-thinking",          "base_url": _MC_URL, "api_key": _MC_KEY},
    # {"model": "mistral-small-4",                 "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "mistral-medium-3.5",            "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "qwen3.5",                       "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "qwen3.5-122b",                  "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "deepseek-v4-pro-thinking",      "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "glm-5",                         "base_url": _MC_URL, "api_key": _MC_KEY},
    {"model": "kimi-k2.6",                     "base_url": _MC_URL, "api_key": _MC_KEY},
    # {"model": "gpt-oss-120b",                  "base_url": _MC_URL, "api_key": _MC_KEY},
]

# Context formats: which representation of the budget table to send
# Each entry is (context_type, token_budget) where token_budget is None for unlimited.
# We use ~4 chars/token as a rough proxy.
CHARS_PER_TOKEN = 4

TOKEN_BUDGETS = [8000, 16000, None]  # None = unlimited (full context)

CONTEXTS = []
for _budget in TOKEN_BUDGETS:
    CONTEXTS.append(("statements", _budget))
    CONTEXTS.append(("markdown", _budget))
    CONTEXTS.append(("structured", _budget))

LANGUAGES = [
#     {
#         "code": "en",
#         "statement_key": "statements_en",
#         "prompt": """You are a budget analyst answering questions about the Kyrgyz Republic's 2026 program-based budget (Приложение 14).
# Answer based ONLY on the provided context. Be precise with numbers. If the context doesn't contain enough information, say so.
# Give a concise answer — just the facts, no preamble.""",
#     },
    {
        "code": "ru",
        "statement_key": "statements_ru",
        "prompt": """Вы бюджетный аналитик и отвечаете на вопросы о программном бюджете Кыргызской Республики на 2026 год (Приложение 14).
Отвечайте ТОЛЬКО на основе предоставленного контекста. Точно указывайте числа. Если в контексте недостаточно информации, так и скажите.
Дайте краткий ответ — только факты, без вступления.""",
    },
]


# ── Load Data ──────────────────────────────────────────────────

def load_qa(language):
    """Load questions for one language.
    Prefer split files where each row has a language-local `question` field;
    fall back to the original combined QA file for compatibility.
    """
    qa_file = QA_FILES.get(language, QA_FILE)
    if qa_file.exists():
        with open(qa_file, encoding="utf-8") as f:
            return json.load(f)["qa_pairs"]

    with open(QA_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)["qa_pairs"]

    source_key = "question_en" if language == "en" else "question"
    return [
        {
            **qa,
            "question": qa.get(source_key) or qa.get("question_en") or qa["question"],
        }
        for qa in qa_pairs
    ]

def load_all_statements(language):
    """Load statements for one language as a list of strings."""
    statements_file = STATEMENT_FILES.get(language, STATEMENTS_FILE)
    if not statements_file.exists():
        statements_file = STATEMENTS_FILE

    with open(statements_file, encoding="utf-8") as f:
        data = json.load(f)
    statement_key = f"statements_{language}"
    statements = []
    for row in data:
        row_statements = row.get(statement_key, []) or row.get("statements", [])
        for s in row_statements:
            statements.append(s)
    return statements

class StatementRetriever:
    """TF-IDF retriever built once, reused for all questions."""
    def __init__(self, statements):
        self.statements = statements
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.backend = "sklearn"
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5),
                max_features=50000, sublinear_tf=True
            )
            self.matrix = self.vectorizer.fit_transform(statements)
            print(f"  TF-IDF index: {self.matrix.shape[0]} docs, {self.matrix.shape[1]} features")
        except ModuleNotFoundError:
            self.backend = "simple"
            self._build_simple_index()
            print(f"  Simple TF-IDF index: {len(self.doc_vectors)} docs, {len(self.idf)} features")

    @staticmethod
    def _char_ngrams(text, min_n=3, max_n=5):
        padded = f" {text.lower()} "
        grams = []
        for n in range(min_n, max_n + 1):
            grams.extend(padded[i:i+n] for i in range(max(0, len(padded) - n + 1)))
        return grams

    def _build_simple_index(self):
        from collections import Counter, defaultdict
        doc_counts = []
        df = defaultdict(int)
        for statement in self.statements:
            counts = Counter(self._char_ngrams(statement))
            doc_counts.append(counts)
            for gram in counts:
                df[gram] += 1

        n_docs = len(self.statements)
        self.idf = {
            gram: math.log((1 + n_docs) / (1 + freq)) + 1
            for gram, freq in df.items()
        }
        self.doc_vectors = []
        for counts in doc_counts:
            vec = {
                gram: (1 + math.log(count)) * self.idf[gram]
                for gram, count in counts.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
            self.doc_vectors.append((vec, norm))

    def _retrieve_simple(self, question, top_k):
        from collections import Counter
        q_counts = Counter(self._char_ngrams(question))
        q_vec = {
            gram: (1 + math.log(count)) * self.idf.get(gram, 0.0)
            for gram, count in q_counts.items()
            if gram in self.idf
        }
        q_norm = math.sqrt(sum(weight * weight for weight in q_vec.values())) or 1.0
        scores = []
        for idx, (doc_vec, doc_norm) in enumerate(self.doc_vectors):
            score = sum(weight * doc_vec.get(gram, 0.0) for gram, weight in q_vec.items())
            if score > 0:
                scores.append((score / (doc_norm * q_norm), idx))
        scores.sort(reverse=True)
        return [self.statements[idx] for _, idx in scores[:top_k]]

    def ranked(self, question, top_k=200):
        """Return (score, statement) pairs for inspection/debugging."""
        if self.backend == "sklearn":
            q_vec = self.vectorizer.transform([question])
            scores = (self.matrix @ q_vec.T).toarray().squeeze()
            top_idx = scores.argsort()[-top_k:][::-1]
            return [
                (float(scores[i]), self.statements[i])
                for i in top_idx
                if scores[i] > 0
            ]

        from collections import Counter
        q_counts = Counter(self._char_ngrams(question))
        q_vec = {
            gram: (1 + math.log(count)) * self.idf.get(gram, 0.0)
            for gram, count in q_counts.items()
            if gram in self.idf
        }
        q_norm = math.sqrt(sum(weight * weight for weight in q_vec.values())) or 1.0
        scores = []
        for idx, (doc_vec, doc_norm) in enumerate(self.doc_vectors):
            score = sum(weight * doc_vec.get(gram, 0.0) for gram, weight in q_vec.items())
            if score > 0:
                scores.append((score / (doc_norm * q_norm), self.statements[idx]))
        scores.sort(reverse=True, key=lambda item: item[0])
        return scores[:top_k]

    def retrieve(self, question, top_k=200, char_budget=None):
        """Retrieve top statements by TF-IDF relevance.
        If char_budget is set, pack as many as fit within that character limit.
        If char_budget is None, return up to top_k statements."""
        candidates = [statement for _, statement in self.ranked(question, top_k=top_k)]

        if char_budget is None:
            selected = candidates[:top_k]
        else:
            selected = []
            total_chars = 0
            for s in candidates:
                line = f"- {s}\n"
                if total_chars + len(line) > char_budget:
                    break
                selected.append(s)
                total_chars += len(line)

        return "\n".join(f"- {s}" for s in selected)

def load_markdown_context():
    return MARKDOWN_FILE.read_text(encoding="utf-8")


def truncate_markdown(full_md, char_budget):
    """Truncate markdown table to char_budget, cutting at line boundaries."""
    if char_budget is None or len(full_md) <= char_budget:
        return full_md
    # Cut at the last newline before the budget
    truncated = full_md[:char_budget]
    last_nl = truncated.rfind("\n")
    if last_nl > 0:
        truncated = truncated[:last_nl]
    return truncated


# ── LLM Caller ─────────────────────────────────────────────────

def call_openai(prompt, model, base_url, api_key):
    base_url = base_url.rstrip("/")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip() or "[ERROR: empty response]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Prompt ─────────────────────────────────────────────────────

STRUCTURED_INSTRUCTIONS = {
    "en": {
        "lookup": "Find the relevant row in the table and report the exact value.",
        "count": "Count the distinct items in the table that match the question. Show the count.",
        "aggregation": "Sum the relevant numeric values from the table. Show the individual values and the total.",
        "comparison": "Find both entities in the table, compare their values, and state which is larger and by how much.",
        "filter": "Scan all rows, apply the filter condition, and count or list the matches.",
        "indicator": "Look at the indicator and unit columns to answer the question.",
        "multi-step": "This requires multiple steps: filter, compute, and/or rank. Show your work.",
    },
    "ru": {
        "lookup": "Найдите нужную строку в таблице и укажите точное значение.",
        "count": "Подсчитайте количество уникальных элементов в таблице, соответствующих вопросу. Укажите число.",
        "aggregation": "Сложите соответствующие числовые значения из таблицы. Покажите отдельные значения и итого.",
        "comparison": "Найдите оба объекта в таблице, сравните их значения и укажите, какой больше и на сколько.",
        "filter": "Просмотрите все строки, примените условие фильтра и подсчитайте или перечислите совпадения.",
        "indicator": "Используйте столбцы показателей и единиц измерения для ответа на вопрос.",
        "multi-step": "Это требует нескольких шагов: фильтрация, вычисление и/или ранжирование. Покажите ход решения.",
    },
}


SCRATCHPAD_INSTRUCTIONS = {
    "en": (
        "Before answering, write a <scratchpad> block. "
        "Inside it, list every relevant row you extract from the context, "
        "state the numbers, and show the arithmetic step-by-step. "
        "Then close </scratchpad> and provide your final answer on a new line."
    ),
    "ru": (
        "Перед ответом напишите блок <scratchpad>. "
        "В нём перечислите каждую релевантную строку из контекста, "
        "укажите числа и покажите вычисления пошагово. "
        "Затем закройте </scratchpad> и дайте окончательный ответ на новой строке."
    ),
}

THINKING_MODELS = {"deepseek-v4-pro-thinking"}


def build_prompt(question, context, system_prompt, category=None, language=None,
                 scratchpad=False):
    instruction = ""
    if category and language:
        lang_instructions = STRUCTURED_INSTRUCTIONS.get(language, {})
        instruction = lang_instructions.get(category, "")
    scratchpad_block = ""
    if scratchpad and language:
        scratchpad_block = SCRATCHPAD_INSTRUCTIONS.get(language, "")
    parts = [system_prompt]
    if instruction:
        parts.append(f"INSTRUCTION: {instruction}")
    if scratchpad_block:
        parts.append(scratchpad_block)
    parts.append(f"CONTEXT:\n{context}")
    parts.append(f"QUESTION: {question}")
    parts.append("ANSWER:")
    return "\n\n".join(parts)


def strip_scratchpad(answer):
    """Remove <scratchpad>...</scratchpad> block, return only the final answer."""
    import re
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", answer, flags=re.DOTALL).strip()
    return cleaned if cleaned else answer


# ── Main ───────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    qa_pairs_by_language = {lang["code"]: load_qa(lang["code"]) for lang in LANGUAGES}
    markdown_context = load_markdown_context()
    statements_by_language = {lang["code"]: load_all_statements(lang["code"]) for lang in LANGUAGES}
    for lang in LANGUAGES:
        print(f"  {len(qa_pairs_by_language[lang['code']])} {lang['code'].upper()} questions")
        print(f"  {len(statements_by_language[lang['code']])} {lang['code'].upper()} statements available for retrieval")
    print(f"  markdown context: {len(markdown_context)} chars (~{len(markdown_context)//4} tokens)")
    print("Building retrievers...")
    retrievers = {
        lang["code"]: StatementRetriever(statements_by_language[lang["code"]])
        for lang in LANGUAGES
    }
    from structured_retriever import StructuredRetriever, load_structured_data
    structured_rows = load_structured_data()
    structured_retriever = StructuredRetriever(structured_rows)
    print(f"  structured data: {len(structured_rows)} rows")

    # CSV: one row per (model, context, question) — written incrementally
    csv_path = RESULTS_DIR / f"all_results_{RUN_ID}.csv"
    fieldnames = ["model", "language", "context", "question_id", "difficulty", "category",
                   "question", "question_en", "question_ru", "reference_answer", "model_answer",
                   "raw_model_answer", "token_budget", "context_chars"]
    csv_rows = []
    # Write header immediately
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_runs = len(OPENAI_MODELS) * len(LANGUAGES) * len(CONTEXTS)
    count = 0

    for cfg in OPENAI_MODELS:
        for lang in LANGUAGES:
            language = lang["code"]
            qa_pairs = qa_pairs_by_language[language]
            retriever = retrievers[language]
            for (context_type, token_budget) in CONTEXTS:
                count += 1
                model = cfg["model"]
                char_budget = token_budget * CHARS_PER_TOKEN if token_budget else None
                budget_label = f"{token_budget}tok" if token_budget else "full"
                context_label = f"{language}_{context_type}_{budget_label}"
                label = f"{context_label}_{model}"
                print(f"\n{'='*60}")
                print(f"  [{count}/{total_runs}] {label}")
                print(f"{'='*60}")

                t0 = time.time()

                for qa in qa_pairs:
                    qid = qa["id"]
                    question = qa["question"]
                    question_en = question if language == "en" else ""
                    question_ru = question if language == "ru" else ""
                    reference = qa["answer"]

                    if context_type == "statements":
                        context_text = retriever.retrieve(question, char_budget=char_budget)
                    elif context_type == "structured":
                        category = qa.get("category", "")
                        context_text = structured_retriever.retrieve(
                            question, category=category, char_budget=char_budget
                        )
                    else:
                        context_text = truncate_markdown(markdown_context, char_budget)

                    use_scratchpad = (
                        context_type == "structured"
                        and model not in THINKING_MODELS
                    )
                    if context_type == "structured":
                        prompt = build_prompt(question, context_text, lang["prompt"],
                                             category=qa.get("category", ""), language=language,
                                             scratchpad=use_scratchpad)
                    else:
                        prompt = build_prompt(question, context_text, lang["prompt"])
                    raw_answer = call_openai(prompt, model, cfg["base_url"], cfg["api_key"])
                    answer = strip_scratchpad(raw_answer) if use_scratchpad else raw_answer

                    csv_rows.append({
                        "model": model,
                        "language": language,
                        "context": context_label,
                        "question_id": qid,
                        "difficulty": qa["difficulty"],
                        "category": qa.get("category", ""),
                        "question": question,
                        "question_en": question_en,
                        "question_ru": question_ru,
                        "reference_answer": reference,
                        "model_answer": answer,
                        "raw_model_answer": raw_answer if use_scratchpad else "",
                        "token_budget": token_budget or "full",
                        "context_chars": len(context_text),
                    })

                    # Truncate answer for display
                    short = answer[:80].replace('\n', ' ')
                    print(f"  Q{qid:2d} [{qa['difficulty']:6s}] {short}...")

                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.0f}s")

                # Flush this run's rows to CSV immediately
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerows(csv_rows[-len(qa_pairs):])

    print(f"\nResults saved to {csv_path}")
    print(f"Total rows: {len(csv_rows)}")

    # Also save as JSON for convenience
    json_path = RESULTS_DIR / f"all_results_{RUN_ID}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(csv_rows, f, ensure_ascii=False, indent=2)
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
