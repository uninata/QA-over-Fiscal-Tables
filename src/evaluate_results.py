"""
LLM-as-Judge Evaluation
========================
Reads results/all_results.csv, sends each (question, reference, model_answer) triple
to a judge LLM, gets a score 0-3, saves scored results.

    python evaluate_results.py

Scoring rubric (0-3):
  3 = Correct: answer contains the right value/entity and is factually accurate
  2 = Partially correct: right entity but wrong number, or right number but missing context
  1 = Related but wrong: mentions relevant topic but gives incorrect answer
  0 = Wrong or refusal: completely wrong, hallucinated, or refused to answer

Output: results/evaluated_results_v1.csv (adds judge_score and judge_reasoning columns)
"""

import json
import os
import csv
import time
import urllib.request
from datetime import datetime

from utils import ROOT, load_env, resolve_project_path, latest_result_file, get_results_dir

load_env()

RESULTS_DIR = get_results_dir()


def resolve_input_csv():
    configured = os.environ.get("EVALUATION_INPUT_CSV")
    if configured:
        return resolve_project_path(configured)
    latest = latest_result_file(RESULTS_DIR, "all_results_*.csv")
    return latest or RESULTS_DIR / "all_results.csv"


def run_id_from_input(input_csv):
    configured = os.environ.get("EXPERIMENT_RUN_ID")
    if configured:
        return configured
    stem = input_csv.stem
    if stem.startswith("all_results_"):
        return stem.removeprefix("all_results_")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


INPUT_CSV = resolve_input_csv()
RUN_ID = run_id_from_input(INPUT_CSV)
OUTPUT_CSV = RESULTS_DIR / f"evaluated_results_{RUN_ID}.csv"
SUMMARY_CSV = RESULTS_DIR / f"evaluation_summary_{RUN_ID}.csv"

# ── Judge Configuration ────────────────────────────────────────
# Use a strong model as judge. Pick one that's NOT being evaluated,
# or use the strongest available.
JUDGE_MODEL = "gpt-oss-120b"
JUDGE_BASE_URL = os.environ.get("METACENTRUM_BASE_URL", "").rstrip("/")
JUDGE_API_KEY = os.environ.get("METACENTRUM_API_KEY", "")


JUDGE_PROMPT = """You are evaluating answers to questions about the Kyrgyz Republic's 2026 program-based budget.

Score the MODEL ANSWER by comparing it to the REFERENCE ANSWER using this rubric:
  3 = Correct: contains the right value/entity and is factually accurate
  2 = Partially correct: right entity but wrong number, or right number but missing key context
  1 = Related but wrong: mentions relevant topic but gives incorrect answer
  0 = Wrong, hallucinated, or refused to answer

IMPORTANT:
- Minor formatting differences (тыс. сом vs thousand som, quotes vs no quotes) do NOT matter
- Rounding is acceptable (42,380,106.4 and 42.4 billion som are both correct for the same value)
- The answer must contain the KEY FACT from the reference to score 2 or 3
- If the model says it cannot answer or lacks information, score 0

QUESTION: {question}
REFERENCE ANSWER: {reference}
MODEL ANSWER: {model_answer}

Respond with ONLY a JSON object, no other text:
{{"score": <0-3>, "reasoning": "<one sentence explanation>"}}"""


def call_judge(question, reference, model_answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        model_answer=model_answer,
    )
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        f"{JUDGE_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JUDGE_API_KEY}",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            return parsed.get("score", -1), parsed.get("reasoning", "")
    except json.JSONDecodeError:
        return -1, f"[PARSE ERROR: {text[:100]}]"
    except Exception as e:
        return -1, f"[ERROR: {e}]"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"No results found at {INPUT_CSV}")
        print("Run run_experiments.py first.")
        return

    # Read input
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} results to evaluate")
    print(f"Judge model: {JUDGE_MODEL}")

    # Evaluate each row
    evaluated = []
    for i, row in enumerate(rows):
        model_answer = row["model_answer"]

        # Skip errors
        if model_answer.startswith("[ERROR"):
            row["judge_score"] = -1
            row["judge_reasoning"] = "API error, not evaluated"
            evaluated.append(row)
            print(f"  [{i+1}/{len(rows)}] {row['model']} / {row['context']} / Q{row['question_id']} — SKIPPED (error)")
            continue

        score, reasoning = call_judge(
            row.get("question") or row.get("question_en") or row.get("question_ru", ""),
            row["reference_answer"],
            model_answer,
        )
        row["judge_score"] = score
        row["judge_reasoning"] = reasoning
        evaluated.append(row)

        label = f"{row['model']}/{row['context']}/Q{row['question_id']}"
        print(f"  [{i+1}/{len(rows)}] {label:50s} → score={score}  {reasoning[:60]}")

        # Be nice to the API
        time.sleep(0.5)

    # Write evaluated CSV
    fieldnames = list(rows[0].keys()) + (["judge_score", "judge_reasoning"] if "judge_score" not in rows[0] else [])
    # Deduplicate fieldnames while preserving order
    seen = set()
    unique_fields = []
    for fn in fieldnames:
        if fn not in seen:
            unique_fields.append(fn)
            seen.add(fn)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=unique_fields)
        writer.writeheader()
        writer.writerows(evaluated)
    print(f"\nEvaluated results saved to {OUTPUT_CSV}")

    # Generate summary
    generate_summary(evaluated)


def generate_summary(evaluated):
    """Print and save a summary table: model × context → average score."""
    from collections import defaultdict

    # Group scores
    scores = defaultdict(list)
    scores_by_diff = defaultdict(lambda: defaultdict(list))

    for row in evaluated:
        s = int(row["judge_score"])
        if s < 0:
            continue
        key = (row["model"], row["context"])
        scores[key].append(s)
        scores_by_diff[key][row["difficulty"]].append(s)

    # Print summary
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY (LLM-as-Judge, scale 0-3)")
    print(f"{'='*70}")
    print(f"\n{'Model':<30} {'Context':<12} {'Avg':>5} {'Easy':>6} {'Med':>6} {'Hard':>6}")
    print("-" * 70)

    summary_rows = []
    for (model, context), sc in sorted(scores.items()):
        avg = sum(sc) / len(sc)
        by_d = scores_by_diff[(model, context)]
        easy = sum(by_d["easy"]) / len(by_d["easy"]) if by_d["easy"] else 0
        med = sum(by_d["medium"]) / len(by_d["medium"]) if by_d["medium"] else 0
        hard = sum(by_d["hard"]) / len(by_d["hard"]) if by_d["hard"] else 0

        strict_acc = sum(1 for s in sc if s == 3) / len(sc)
        lenient_acc = sum(1 for s in sc if s >= 2) / len(sc)

        print(f"{model:<30} {context:<12} {avg:>5.2f} {easy:>6.2f} {med:>6.2f} {hard:>6.2f}  strict={strict_acc:.0%} lenient={lenient_acc:.0%}")

        summary_rows.append({
            "model": model,
            "context": context,
            "avg_score": round(avg, 2),
            "easy_avg": round(easy, 2),
            "medium_avg": round(med, 2),
            "hard_avg": round(hard, 2),
            "strict_accuracy": round(strict_acc, 3),
            "lenient_accuracy": round(lenient_acc, 3),
            "n_evaluated": len(sc),
        })

    # Save summary CSV
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "context", "avg_score",
                                                "easy_avg", "medium_avg", "hard_avg",
                                                "strict_accuracy", "lenient_accuracy", "n_evaluated"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
