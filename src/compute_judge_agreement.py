"""
Compute agreement between human scores and LLM-as-Judge scores.

Reads the filled-in human validation CSV and the answer key, computes:
  - Exact agreement rate
  - Agreement within ±1
  - Cohen's kappa (weighted, linear)
  - Confusion matrix
  - Per-score-bucket accuracy
  - Systematic bias (judge higher/lower than human)

Usage:
    python src/compute_judge_agreement.py --run v4_round3
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_results_dir

RESULTS_DIR = get_results_dir()


def load_pairs(run_id):
    human_path = RESULTS_DIR / f"human_validation_{run_id}.csv"
    key_path = RESULTS_DIR / f"human_validation_{run_id}_key.csv"

    if not human_path.exists():
        raise SystemExit(f"Not found: {human_path}")
    if not key_path.exists():
        raise SystemExit(f"Not found: {key_path}")

    with open(human_path, encoding="utf-8") as f:
        human_rows = {r["sample_id"]: r for r in csv.DictReader(f)}
    with open(key_path, encoding="utf-8") as f:
        key_rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    pairs = []
    skipped = 0
    for sid in sorted(human_rows, key=int):
        h = human_rows[sid]
        k = key_rows[sid]
        if not h["human_score"].strip():
            skipped += 1
            continue
        pairs.append({
            "sample_id": sid,
            "question_id": k["question_id"],
            "model": k["model"],
            "context": k["context"],
            "human": int(h["human_score"]),
            "judge": int(k["judge_score"]),
            "human_notes": h.get("human_notes", ""),
            "judge_reasoning": k.get("judge_reasoning", ""),
        })

    if skipped:
        print(f"Skipped {skipped} unannotated rows")
    return pairs


def cohens_kappa_linear(pairs, n_classes=4):
    n = len(pairs)
    if n == 0:
        return 0.0

    # Observed weighted agreement (linear weights)
    max_diff = n_classes - 1
    observed = sum(1 - abs(p["human"] - p["judge"]) / max_diff for p in pairs) / n

    # Expected weighted agreement
    human_dist = Counter(p["human"] for p in pairs)
    judge_dist = Counter(p["judge"] for p in pairs)
    expected = 0
    for h in range(n_classes):
        for j in range(n_classes):
            weight = 1 - abs(h - j) / max_diff
            expected += weight * (human_dist[h] / n) * (judge_dist[j] / n)

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v4_round3")
    args = parser.parse_args()

    pairs = load_pairs(args.run)
    n = len(pairs)
    if n == 0:
        print("No scored pairs found. Fill in human_score column first.")
        return

    print(f"Loaded {n} human-judge score pairs\n")

    # Exact agreement
    exact = sum(1 for p in pairs if p["human"] == p["judge"])
    print(f"Exact agreement:  {exact}/{n} ({exact/n:.0%})")

    # Within ±1
    within1 = sum(1 for p in pairs if abs(p["human"] - p["judge"]) <= 1)
    print(f"Within ±1:        {within1}/{n} ({within1/n:.0%})")

    # Cohen's kappa
    kappa = cohens_kappa_linear(pairs)
    print(f"Cohen's κ (linear weighted): {kappa:.3f}")

    # Interpretation
    if kappa >= 0.8:
        interp = "almost perfect"
    elif kappa >= 0.6:
        interp = "substantial"
    elif kappa >= 0.4:
        interp = "moderate"
    elif kappa >= 0.2:
        interp = "fair"
    else:
        interp = "slight"
    print(f"  Interpretation: {interp} agreement")

    # Bias
    diffs = [p["judge"] - p["human"] for p in pairs]
    mean_diff = sum(diffs) / n
    print(f"\nMean bias (judge - human): {mean_diff:+.2f}")
    if mean_diff > 0.2:
        print("  → Judge is systematically generous")
    elif mean_diff < -0.2:
        print("  → Judge is systematically harsh")
    else:
        print("  → No systematic bias")

    higher = sum(1 for d in diffs if d > 0)
    lower = sum(1 for d in diffs if d < 0)
    same = sum(1 for d in diffs if d == 0)
    print(f"  Judge higher: {higher}  Same: {same}  Judge lower: {lower}")

    # Confusion matrix
    print(f"\nConfusion matrix (rows=human, cols=judge):")
    print(f"{'':>8}", end="")
    for j in range(4):
        print(f"  J={j}", end="")
    print()
    for h in range(4):
        print(f"  H={h}  ", end="")
        for j in range(4):
            count = sum(1 for p in pairs if p["human"] == h and p["judge"] == j)
            print(f"  {count:>3}", end="")
        print()

    # Per-score accuracy
    print(f"\nPer judge-score accuracy (how often human agrees):")
    for s in range(4):
        bucket = [p for p in pairs if p["judge"] == s]
        if not bucket:
            continue
        agree = sum(1 for p in bucket if p["human"] == s)
        within = sum(1 for p in bucket if abs(p["human"] - s) <= 1)
        print(f"  Judge={s}: {len(bucket)} samples, exact={agree}/{len(bucket)} ({agree/len(bucket):.0%}), ±1={within}/{len(bucket)} ({within/len(bucket):.0%})")

    # Disagreements
    disagreements = [p for p in pairs if abs(p["human"] - p["judge"]) >= 2]
    if disagreements:
        print(f"\nLarge disagreements (|diff| ≥ 2): {len(disagreements)}")
        for p in disagreements:
            print(f"  Sample {p['sample_id']} Q{p['question_id']} ({p['model']}/{p['context']}): "
                  f"human={p['human']} judge={p['judge']}  "
                  f"notes: {p['human_notes'][:60]}")

    # Binary agreement (correct = ≥2)
    print(f"\nBinary agreement (correct = score ≥ 2):")
    h_correct = [p["human"] >= 2 for p in pairs]
    j_correct = [p["judge"] >= 2 for p in pairs]
    binary_agree = sum(1 for h, j in zip(h_correct, j_correct) if h == j)
    print(f"  Agreement: {binary_agree}/{n} ({binary_agree/n:.0%})")
    tp = sum(1 for h, j in zip(h_correct, j_correct) if h and j)
    fp = sum(1 for h, j in zip(h_correct, j_correct) if not h and j)
    fn = sum(1 for h, j in zip(h_correct, j_correct) if h and not j)
    tn = sum(1 for h, j in zip(h_correct, j_correct) if not h and not j)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main()
