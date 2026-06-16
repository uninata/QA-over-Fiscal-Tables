"""
Round 4 — Statements-only experiment with increased top-k.

Tests whether giving models more (or all) statements improves performance,
especially on categories where statements currently struggle (filter, multi-step, aggregation).

Same setup as Round 3:
  - Category-specific instructions per question
  - Scratchpad prompting for non-thinking models (Qwen 3.5)
  - DeepSeek V4 Pro (thinking) skips scratchpad (reasons internally)
  - Russian only

Two conditions:
  - top500: Top 500 TF-IDF-ranked statements (~40K tokens)
  - all:    All 1,424 statements (~100K tokens)

Comparison baseline: Round 3 ru_statements_full (top 200, ~14K tokens)

Usage:
    python src/run_statements_experiment.py
    python src/run_statements_experiment.py --dry-run
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, load_env, get_results_dir

load_env()

RESULTS_DIR = get_results_dir()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────

THINKING_MODELS = {"deepseek-v4-pro-thinking"}

MODELS = [
    {
        "model": "qwen3.5",
        "base_url": os.environ.get("METACENTRUM_BASE_URL", ""),
        "api_key": os.environ.get("METACENTRUM_API_KEY", ""),
    },
    {
        "model": "deepseek-v4-pro-thinking",
        "base_url": os.environ.get("METACENTRUM_BASE_URL", ""),
        "api_key": os.environ.get("METACENTRUM_API_KEY", ""),
    },
]

# ── Statement conditions ──────────────────────────────────────
# (label, top_k) — None means all statements
CONDITIONS = [
    ("top500", 500),
    ("all", None),
]

# ── Prompts (same as Round 3) ─────────────────────────────────

SYSTEM_PROMPT = (
    "Вы бюджетный аналитик и отвечаете на вопросы о программном бюджете "
    "Кыргызской Республики на 2026 год (Приложение 14).\n"
    "Отвечайте ТОЛЬКО на основе предоставленного контекста. "
    "Точно указывайте числа. Если в контексте недостаточно информации, так и скажите.\n"
    "Дайте краткий ответ — только факты, без вступления."
)

CATEGORY_INSTRUCTIONS = {
    "lookup": "Найдите нужную строку в таблице и укажите точное значение.",
    "count": "Подсчитайте количество уникальных элементов в таблице, соответствующих вопросу. Укажите число.",
    "aggregation": "Сложите соответствующие числовые значения из таблицы. Покажите отдельные значения и итого.",
    "comparison": "Найдите оба объекта в таблице, сравните их значения и укажите, какой больше и на сколько.",
    "filter": "Просмотрите все строки, примените условие фильтра и подсчитайте или перечислите совпадения.",
    "indicator": "Используйте столбцы показателей и единиц измерения для ответа на вопрос.",
    "multi-step": "Это требует нескольких шагов: фильтрация, вычисление и/или ранжирование. Покажите ход решения.",
}

SCRATCHPAD_INSTRUCTION = (
    "Перед ответом напишите блок <scratchpad>. "
    "В нём перечислите каждую релевантную строку из контекста, "
    "укажите числа и покажите вычисления пошагово. "
    "Затем закройте </scratchpad> и дайте окончательный ответ на новой строке."
)


def load_qa():
    qa_path = ROOT / "data" / "qa" / "qa_pairs_ru_2026.json"
    with open(qa_path, encoding="utf-8") as f:
        return json.load(f)["qa_pairs"]


def load_statements():
    from run_experiments import load_all_statements, StatementRetriever
    stmts = load_all_statements("ru")
    retriever = StatementRetriever(stmts)
    return stmts, retriever


def retrieve_statements(retriever, question, top_k, all_statements):
    if top_k is None:
        ranked = retriever.ranked(question, top_k=len(all_statements))
    else:
        ranked = retriever.ranked(question, top_k=top_k)
    return "\n".join(f"- {s}" for _, s in ranked)


def build_prompt(question, context, category, use_scratchpad):
    parts = [SYSTEM_PROMPT]

    instruction = CATEGORY_INSTRUCTIONS.get(category, "")
    if instruction:
        parts.append(f"ИНСТРУКЦИЯ: {instruction}")

    if use_scratchpad:
        parts.append(SCRATCHPAD_INSTRUCTION)

    parts.append(f"КОНТЕКСТ:\n{context}")
    parts.append(f"ВОПРОС: {question}")
    parts.append("ОТВЕТ:")
    return "\n\n".join(parts)


def strip_scratchpad(answer):
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", answer, flags=re.DOTALL).strip()
    return cleaned if cleaned else answer


def call_api(prompt, model, base_url, api_key):
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
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip() or "[ERROR: empty response]"
    except Exception as e:
        return f"[ERROR: {e}]"


def setup_logging():
    import logging
    log_path = RESULTS_DIR / "v5_statements.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def main():
    dry_run = "--dry-run" in sys.argv
    log = setup_logging()

    qa_pairs = load_qa()
    all_statements, retriever = load_statements()

    csv_path = RESULTS_DIR / "all_results_v5_statements.csv"
    fieldnames = [
        "model", "context", "condition", "question_id", "difficulty", "category",
        "question", "reference_answer", "model_answer", "raw_model_answer",
        "context_chars", "n_statements",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total = len(MODELS) * len(CONDITIONS) * len(qa_pairs)
    count = 0

    for cfg in MODELS:
        model = cfg["model"]
        is_thinking = model in THINKING_MODELS

        for cond_label, top_k in CONDITIONS:
            context_label = f"ru_statements_{cond_label}"
            round_rows = []

            log.info(f"{'='*60}")
            log.info(f"  {model} / {context_label}")
            log.info(f"  top_k={'all' if top_k is None else top_k}")
            log.info(f"  scratchpad={'no (thinking model)' if is_thinking else 'yes'}")
            log.info(f"{'='*60}")

            t0 = time.time()
            for qa in qa_pairs:
                count += 1
                question = qa["question"]
                reference = qa["answer"]
                category = qa.get("category", "")
                use_scratchpad = not is_thinking

                context = retrieve_statements(retriever, question, top_k, all_statements)
                n_statements = context.count("\n- ") + 1

                if dry_run:
                    log.info(f"  Q{qa['id']:>2} [{category:>11s}] {len(context):>6} chars, "
                             f"{n_statements} stmts, scratchpad={'Y' if use_scratchpad else 'N'} — DRY RUN")
                    raw_answer = "[DRY RUN]"
                    answer = raw_answer
                else:
                    prompt = build_prompt(question, context, category, use_scratchpad)
                    raw_answer = call_api(prompt, model, cfg["base_url"], cfg["api_key"])
                    answer = strip_scratchpad(raw_answer) if use_scratchpad else raw_answer
                    status = "OK" if not answer.startswith("[ERROR") else "ERR"
                    log.info(f"  [{count}/{total}] Q{qa['id']:>2} [{category:>11s}] "
                             f"{len(context):>6} chars → {status}  {answer[:60]}")

                row = {
                    "model": model,
                    "context": context_label,
                    "condition": cond_label,
                    "question_id": qa["id"],
                    "difficulty": qa["difficulty"],
                    "category": category,
                    "question": question,
                    "reference_answer": reference,
                    "model_answer": answer,
                    "raw_model_answer": raw_answer if use_scratchpad else "",
                    "context_chars": len(context),
                    "n_statements": n_statements,
                }
                round_rows.append(row)

            elapsed = time.time() - t0

            # Write all rows for this round to CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(round_rows)

            log.info(f"  Completed {model}/{context_label}: {len(round_rows)} rows in {elapsed:.0f}s — saved to CSV")

    log.info(f"Results saved to {csv_path}")
    log.info(f"Total: {count} question-answer pairs")
    log.info(f"To evaluate:")
    log.info(f"  EVALUATION_INPUT_CSV={csv_path} EXPERIMENT_RUN_ID=v5_statements python src/evaluate_results.py")
    log.info(f"To compare with Round 3:")
    log.info(f"  python src/compare_runs.py v4_round3 v5_statements --by-category")


if __name__ == "__main__":
    main()
