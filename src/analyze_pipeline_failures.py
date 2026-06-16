"""
Join TF-IDF retrieval diagnostics with judged model answers.

This separates failures into:
  - retrieval_miss_model_wrong: context did not contain the answer, model wrong
  - retrieval_hit_model_wrong: context contained the answer, model still wrong
  - retrieval_miss_model_right: model got it right without retrieved evidence
  - retrieval_hit_model_right: retrieval and model both worked

Inputs:
  results/tfidf_retrieval_results.csv
  results/evaluated_results_v1.csv

Outputs:
  results/pipeline_failure_analysis.csv
  results/pipeline_failure_summary.csv
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

from utils import ROOT, resolve_project_path, latest_result_file, get_results_dir

RESULTS_DIR = get_results_dir()

CORRECT_SCORE_THRESHOLD = 2


def resolve_input_csv(env_name, versioned_pattern, fallback_name):
    configured = os.environ.get(env_name)
    if configured:
        return resolve_project_path(configured)
    latest = latest_result_file(RESULTS_DIR, versioned_pattern)
    if latest:
        return latest
    return RESULTS_DIR / fallback_name


RETRIEVAL_CSV = resolve_input_csv(
    "RETRIEVAL_INPUT_CSV",
    "tfidf_retrieval_results_*.csv",
    "tfidf_retrieval_results.csv",
)
EVALUATED_CSV = resolve_input_csv(
    "EVALUATED_INPUT_CSV",
    "evaluated_results_*.csv",
    "evaluated_results_v1.csv",
)
RUN_ID = (
    os.environ.get("PIPELINE_ANALYSIS_RUN_ID")
    or os.environ.get("EXPERIMENT_RUN_ID")
    or datetime.now().strftime("%Y%m%d_%H%M%S")
)
OUTPUT_CSV = RESULTS_DIR / f"pipeline_failure_analysis_{RUN_ID}.csv"
SUMMARY_CSV = RESULTS_DIR / f"pipeline_failure_summary_{RUN_ID}.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_language(row):
    language = row.get("language", "").strip()
    if language:
        return language
    return "en"


def normalize_context(row):
    language = normalize_language(row)
    context = row["context"]
    if context.startswith(("en_", "ru_")):
        return context
    if context.startswith("statements_"):
        return f"{language}_{context}"
    return context


def join_key(row):
    return (
        normalize_language(row),
        normalize_context(row),
        str(row["question_id"]),
    )


def classify(retrieval_hit, model_correct):
    if retrieval_hit and model_correct:
        return "retrieval_hit_model_right"
    if retrieval_hit and not model_correct:
        return "retrieval_hit_model_wrong"
    if not retrieval_hit and model_correct:
        return "retrieval_miss_model_right"
    return "retrieval_miss_model_wrong"


def analyze():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_rows = read_csv(RETRIEVAL_CSV)
    evaluated_rows = read_csv(EVALUATED_CSV)

    retrieval_by_key = {join_key(row): row for row in retrieval_rows}
    joined = []
    skipped = 0

    for eval_row in evaluated_rows:
        normalized_context = normalize_context(eval_row)
        if "_statements_" not in normalized_context:
            skipped += 1
            continue

        key = join_key(eval_row)
        retrieval_row = retrieval_by_key.get(key)
        if not retrieval_row:
            skipped += 1
            continue

        judge_score = int(eval_row["judge_score"])
        retrieval_hit = int(retrieval_row["contains_reference_answer"]) == 1
        model_correct = judge_score >= CORRECT_SCORE_THRESHOLD
        failure_type = classify(retrieval_hit, model_correct)

        joined.append({
            "model": eval_row["model"],
            "language": normalize_language(eval_row),
            "context": normalized_context,
            "question_id": eval_row["question_id"],
            "difficulty": eval_row.get("difficulty", ""),
            "category": eval_row.get("category", ""),
            "question": eval_row.get("question") or eval_row.get("question_en") or eval_row.get("question_ru", ""),
            "reference_answer": eval_row["reference_answer"],
            "model_answer": eval_row["model_answer"],
            "judge_score": judge_score,
            "judge_reasoning": eval_row.get("judge_reasoning", ""),
            "retrieval_hit": int(retrieval_hit),
            "exact_reference_match": retrieval_row["exact_reference_match"],
            "all_reference_numbers_found": retrieval_row["all_reference_numbers_found"],
            "failure_type": failure_type,
            "token_budget": eval_row.get("token_budget") or retrieval_row.get("token_budget", ""),
            "context_chars": eval_row.get("context_chars", ""),
        })

    write_joined(joined)
    write_summary(joined)

    print(f"Joined rows: {len(joined)}")
    if skipped:
        print(f"Skipped rows without matching retrieval diagnostics: {skipped}")


def write_joined(rows):
    fieldnames = [
        "model", "language", "context", "question_id", "difficulty", "category",
        "question", "reference_answer", "model_answer", "judge_score",
        "judge_reasoning", "retrieval_hit", "exact_reference_match",
        "all_reference_numbers_found", "failure_type", "token_budget",
        "context_chars",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detailed analysis saved to {OUTPUT_CSV}")


def summarize_group(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        total = len(group)
        counts = defaultdict(int)
        retrieval_hits = 0
        model_correct = 0
        for row in group:
            counts[row["failure_type"]] += 1
            retrieval_hits += int(row["retrieval_hit"])
            model_correct += int(int(row["judge_score"]) >= CORRECT_SCORE_THRESHOLD)

        summary = {k: v for k, v in zip(group_keys, key)}
        summary.update({
            "n": total,
            "retrieval_hit_rate": round(retrieval_hits / total, 3),
            "model_correct_rate": round(model_correct / total, 3),
            "retrieval_miss_model_wrong": counts["retrieval_miss_model_wrong"],
            "retrieval_hit_model_wrong": counts["retrieval_hit_model_wrong"],
            "retrieval_miss_model_right": counts["retrieval_miss_model_right"],
            "retrieval_hit_model_right": counts["retrieval_hit_model_right"],
        })
        summary_rows.append(summary)
    return summary_rows


def write_summary(rows):
    summary_rows = []
    summary_rows.extend(summarize_group(rows, ["model", "language", "context"]))
    summary_rows.extend(summarize_group(rows, ["language", "context"]))

    fieldnames = [
        "model", "language", "context", "n", "retrieval_hit_rate",
        "model_correct_rate", "retrieval_miss_model_wrong",
        "retrieval_hit_model_wrong", "retrieval_miss_model_right",
        "retrieval_hit_model_right",
    ]

    for row in summary_rows:
        row.setdefault("model", "ALL")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary saved to {SUMMARY_CSV}")
    print("\nPipeline failure summary")
    for row in summary_rows:
        if row["model"] != "ALL":
            continue
        print(
            f"  {row['language']} {row['context']:<24} "
            f"retrieval_hit={row['retrieval_hit_rate']:.2f} "
            f"model_correct={row['model_correct_rate']:.2f} "
            f"retrieval_miss_wrong={row['retrieval_miss_model_wrong']} "
            f"hit_but_wrong={row['retrieval_hit_model_wrong']}"
        )


if __name__ == "__main__":
    analyze()
