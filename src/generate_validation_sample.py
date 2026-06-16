"""
Generate a stratified sample for human validation of LLM-as-Judge scores.

Produces an Excel file with model answers, reference answers, and judge scores
hidden in a separate sheet — the annotator scores blind, then agreement is computed.

Usage:
    python src/generate_validation_sample.py                        # default: 60 samples from V4
    python src/generate_validation_sample.py --n 100                # larger sample
    python src/generate_validation_sample.py --run v3_reeval --n 80
    python src/generate_validation_sample.py --seed 42              # reproducible
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_results_dir

RESULTS_DIR = get_results_dir()


def load_evaluated(run_id):
    path = RESULTS_DIR / f"evaluated_results_{run_id}.csv"
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r["judge_score"]) >= 0]


def stratified_sample(rows, n, seed):
    rng = random.Random(seed)

    by_score = defaultdict(list)
    for r in rows:
        by_score[r["judge_score"]].append(r)

    # Allocate proportionally, but ensure at least 3 per score bucket
    # and oversample score=1 and score=2 (the ambiguous middle)
    score_weights = {"0": 1.0, "1": 2.0, "2": 2.0, "3": 1.0}
    total_weight = sum(score_weights[s] * len(by_score[s]) for s in by_score)

    allocation = {}
    for s in sorted(by_score):
        raw = score_weights[s] * len(by_score[s]) / total_weight * n
        allocation[s] = max(3, round(raw))

    # Adjust to hit target n
    while sum(allocation.values()) > n:
        biggest = max(allocation, key=lambda s: allocation[s])
        allocation[biggest] -= 1
    while sum(allocation.values()) < n:
        smallest = min(allocation, key=lambda s: allocation[s])
        allocation[smallest] += 1

    sampled = []
    for s in sorted(by_score):
        pool = by_score[s]
        k = min(allocation[s], len(pool))
        sampled.extend(rng.sample(pool, k))

    rng.shuffle(sampled)
    return sampled


def write_annotation_csv(sampled, output_path):
    fieldnames = [
        "sample_id", "question_id", "difficulty", "category",
        "question", "reference_answer", "model_answer",
        "human_score", "human_notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(sampled, 1):
            question = r.get("question") or r.get("question_ru") or r.get("question_en", "")
            writer.writerow({
                "sample_id": i,
                "question_id": r["question_id"],
                "difficulty": r["difficulty"],
                "category": r.get("category", ""),
                "question": question,
                "reference_answer": r["reference_answer"],
                "model_answer": r["model_answer"],
                "human_score": "",
                "human_notes": "",
            })

    print(f"Annotation sheet: {output_path}")


def write_answer_key(sampled, output_path):
    fieldnames = [
        "sample_id", "question_id", "model", "context",
        "judge_score", "judge_reasoning",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(sampled, 1):
            writer.writerow({
                "sample_id": i,
                "question_id": r["question_id"],
                "model": r["model"],
                "context": r["context"],
                "judge_score": r["judge_score"],
                "judge_reasoning": r["judge_reasoning"],
            })

    print(f"Answer key (hidden): {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate human validation sample")
    parser.add_argument("--run", default="v4_round3", help="Run ID to sample from")
    parser.add_argument("--n", type=int, default=60, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = load_evaluated(args.run)
    sampled = stratified_sample(rows, args.n, args.seed)

    print(f"Sampled {len(sampled)} from {len(rows)} scored responses ({args.run})")
    print(f"Score distribution in sample: ", end="")
    from collections import Counter
    dist = Counter(r["judge_score"] for r in sampled)
    print(", ".join(f"score {s}: {dist[s]}" for s in sorted(dist)))

    annotation_path = RESULTS_DIR / f"human_validation_{args.run}.csv"
    key_path = RESULTS_DIR / f"human_validation_{args.run}_key.csv"

    write_annotation_csv(sampled, annotation_path)
    write_answer_key(sampled, key_path)

    print(f"\nProtocol:")
    print(f"  1. Open {annotation_path.name}")
    print(f"  2. Score each row (human_score: 0-3) using the rubric below")
    print(f"  3. Run: python src/compute_judge_agreement.py --run {args.run}")


if __name__ == "__main__":
    main()
