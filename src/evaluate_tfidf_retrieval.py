"""
Evaluate TF-IDF retrieval before running answer-generation or judge models.

For each language and token budget, this script retrieves statement context for
each question and checks whether the retrieved context contains the reference
answer.

Output:
  results/tfidf_retrieval_results.csv
  results/tfidf_retrieval_summary.csv
"""

import csv
import os
import re

import run_experiments as experiments
from utils import get_results_dir, get_run_id

RESULTS_DIR = get_results_dir()
RUN_ID = get_run_id()
OUTPUT_CSV = RESULTS_DIR / f"tfidf_retrieval_results_{RUN_ID}.csv"
SUMMARY_CSV = RESULTS_DIR / f"tfidf_retrieval_summary_{RUN_ID}.csv"


def normalize_text(text):
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[«»\"'(),.;:—–-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_number(number):
    value = number.replace(",", "").replace(" ", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def extract_numbers(text):
    numbers = re.findall(r"\d[\d,\s]*(?:\.\d+)?%?", text)
    return [normalize_number(n.rstrip("%")) for n in numbers]


def is_substantive_number(number):
    return "." in number or len(number) >= 3


def contains_exact_reference(context, reference):
    normalized_reference = normalize_text(reference)
    normalized_context = normalize_text(context)
    if re.fullmatch(r"\d+", normalized_reference):
        return re.search(rf"(^|\s){re.escape(normalized_reference)}($|\s)", normalized_context) is not None
    return normalized_reference in normalized_context


def number_coverage(context, reference):
    reference_numbers = extract_numbers(reference)
    context_numbers = set(extract_numbers(context))
    found = [n for n in reference_numbers if n and n in context_numbers]
    return reference_numbers, found


def evaluate_one_context(qa, context, language, context_label, token_budget):
    """Evaluate a single (question, context) pair and return a result row."""
    reference = qa["answer"]
    exact_match = contains_exact_reference(context, reference)
    ref_numbers, found_numbers = number_coverage(context, reference)
    substantive_ref_numbers = [n for n in ref_numbers if is_substantive_number(n)]
    substantive_found_numbers = [n for n in found_numbers if n in substantive_ref_numbers]
    all_numbers_found = (
        bool(substantive_ref_numbers)
        and len(substantive_found_numbers) == len(substantive_ref_numbers)
    )
    contains_reference_answer = exact_match or all_numbers_found

    return {
        "language": language,
        "context": context_label,
        "question_id": qa["id"],
        "difficulty": qa["difficulty"],
        "category": qa.get("category", ""),
        "question": qa["question"],
        "reference_answer": reference,
        "contains_reference_answer": int(contains_reference_answer),
        "exact_reference_match": int(exact_match),
        "all_reference_numbers_found": int(all_numbers_found),
        "reference_numbers": "|".join(ref_numbers),
        "found_reference_numbers": "|".join(found_numbers),
        "substantive_reference_numbers": "|".join(substantive_ref_numbers),
        "token_budget": token_budget or "full",
        "context_chars": len(context),
    }


def evaluate_retrieval():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    from structured_retriever import StructuredRetriever, load_structured_data
    structured_rows = load_structured_data()
    structured_retriever = StructuredRetriever(structured_rows)

    for lang in experiments.LANGUAGES:
        language = lang["code"]
        qa_pairs = experiments.load_qa(language)
        statements = experiments.load_all_statements(language)
        retriever = experiments.StatementRetriever(statements)

        for token_budget in experiments.TOKEN_BUDGETS:
            char_budget = token_budget * experiments.CHARS_PER_TOKEN if token_budget else None
            budget_label = f"{token_budget}tok" if token_budget else "full"

            for qa in qa_pairs:
                # Evaluate statement retrieval
                context = retriever.retrieve(qa["question"], char_budget=char_budget)
                rows.append(evaluate_one_context(
                    qa, context, language,
                    f"{language}_statements_{budget_label}", token_budget,
                ))

                # Evaluate structured retrieval
                category = qa.get("category", "")
                struct_context = structured_retriever.retrieve(
                    qa["question"], category=category, char_budget=char_budget,
                )
                rows.append(evaluate_one_context(
                    qa, struct_context, language,
                    f"{language}_structured_{budget_label}", token_budget,
                ))

    write_results(rows)
    write_summary(rows)
    return rows


def write_results(rows):
    fieldnames = [
        "language", "context", "question_id", "difficulty", "category",
        "question", "reference_answer", "contains_reference_answer",
        "exact_reference_match", "all_reference_numbers_found",
        "reference_numbers", "found_reference_numbers",
        "substantive_reference_numbers",
        "token_budget", "context_chars",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Retrieval results saved to {OUTPUT_CSV}")


def write_summary(rows):
    groups = {}
    for row in rows:
        key = (row["language"], row["context"], row["token_budget"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (language, context, token_budget), group in sorted(groups.items()):
        total = len(group)
        contains = sum(int(r["contains_reference_answer"]) for r in group)
        exact = sum(int(r["exact_reference_match"]) for r in group)
        numbers = sum(int(r["all_reference_numbers_found"]) for r in group)
        summary_rows.append({
            "language": language,
            "context": context,
            "token_budget": token_budget,
            "n_questions": total,
            "contains_reference_answer_rate": round(contains / total, 3),
            "exact_reference_match_rate": round(exact / total, 3),
            "all_reference_numbers_found_rate": round(numbers / total, 3),
        })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "language", "context", "token_budget", "n_questions",
            "contains_reference_answer_rate",
            "exact_reference_match_rate",
            "all_reference_numbers_found_rate",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Retrieval summary saved to {SUMMARY_CSV}")
    print("\nTF-IDF retrieval summary")
    for row in summary_rows:
        print(
            f"  {row['context']:<25} "
            f"contains={row['contains_reference_answer_rate']:.2f} "
            f"exact={row['exact_reference_match_rate']:.2f} "
            f"numbers={row['all_reference_numbers_found_rate']:.2f}"
        )


if __name__ == "__main__":
    evaluate_retrieval()
