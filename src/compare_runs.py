"""
Compare evaluation results across experiment runs.

Usage:
    python src/compare_runs.py                          # compare all evaluated_results_*.csv
    python src/compare_runs.py v1_reeval v4_round3      # compare specific runs
    python src/compare_runs.py v1_reeval v3_reeval v4_round3 --model qwen3.5
    python src/compare_runs.py v1_reeval v4_round3 --context ru_markdown_full
    python src/compare_runs.py v1_reeval v4_round3 --by-question
    python src/compare_runs.py v1_reeval v4_round3 --by-category
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_results_dir

RESULTS_DIR = get_results_dir()


def discover_runs():
    runs = {}
    for p in sorted(RESULTS_DIR.glob("evaluated_results_*.csv")):
        run_id = p.stem.removeprefix("evaluated_results_")
        runs[run_id] = p
    return runs


def load_run(path):
    with open(path, encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            score = int(r["judge_score"])
            if score < 0:
                continue
            rows.append({
                "model": r["model"],
                "context": r["context"],
                "question_id": r["question_id"],
                "difficulty": r["difficulty"],
                "category": r.get("category", ""),
                "score": score,
            })
    return rows


def accuracy(scores):
    if not scores:
        return {"n": 0, "avg": 0, "strict": 0, "lenient": 0}
    n = len(scores)
    return {
        "n": n,
        "avg": round(sum(scores) / n, 2),
        "strict": round(sum(1 for s in scores if s == 3) / n, 3),
        "lenient": round(sum(1 for s in scores if s >= 2) / n, 3),
    }


def fmt_pct(v):
    return f"{v:.0%}"


def fmt_delta(v):
    return f"{v:+.0%}" if v else "—"


def print_table(headers, rows, col_widths=None):
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def compare_overall(runs_data, run_ids):
    print("\n=== OVERALL COMPARISON ===\n")
    headers = ["Run", "N", "Avg", "Strict", "Lenient", "Hard Str", "Hard Len"]
    rows = []
    for rid in run_ids:
        all_scores = [r["score"] for r in runs_data[rid]]
        hard_scores = [r["score"] for r in runs_data[rid] if r["difficulty"] == "hard"]
        a = accuracy(all_scores)
        h = accuracy(hard_scores)
        rows.append([rid, a["n"], a["avg"], fmt_pct(a["strict"]), fmt_pct(a["lenient"]),
                      fmt_pct(h["strict"]), fmt_pct(h["lenient"])])
    print_table(headers, rows)


def compare_by_model(runs_data, run_ids, context_filter=None):
    print("\n=== BY MODEL (best config per model) ===\n")
    if context_filter:
        print(f"  [filtered to context: {context_filter}]\n")

    for rid in run_ids:
        print(f"--- {rid} ---")
        data = runs_data[rid]
        if context_filter:
            data = [r for r in data if r["context"] == context_filter]

        by_mc = defaultdict(list)
        for r in data:
            by_mc[(r["model"], r["context"])].append(r["score"])

        best = {}
        for (m, c), scores in by_mc.items():
            a = accuracy(scores)
            if m not in best or a["lenient"] > best[m]["lenient"] or \
               (a["lenient"] == best[m]["lenient"] and a["avg"] > best[m]["avg"]):
                hard = [s for (mm, cc), sc in by_mc.items() if mm == m and cc == c
                        for s, r in zip(sc, [r for r in data if r["model"] == m and r["context"] == c])
                        if r["difficulty"] == "hard"]
                # Recalculate hard from raw data
                hard_scores = [r["score"] for r in data if r["model"] == m and r["context"] == c and r["difficulty"] == "hard"]
                h = accuracy(hard_scores)
                best[m] = {**a, "context": c, "h_strict": h["strict"], "h_lenient": h["lenient"]}

        headers = ["Model", "Context", "Avg", "Strict", "Lenient", "H.Str", "H.Len"]
        rows = []
        for m in sorted(best):
            b = best[m]
            rows.append([m, b["context"], b["avg"], fmt_pct(b["strict"]), fmt_pct(b["lenient"]),
                          fmt_pct(b["h_strict"]), fmt_pct(b["h_lenient"])])
        print_table(headers, rows)
        print()


def compare_single_model(runs_data, run_ids, model_name):
    print(f"\n=== {model_name} — ALL CONFIGS ACROSS RUNS ===\n")

    for rid in run_ids:
        data = [r for r in runs_data[rid] if r["model"] == model_name]
        if not data:
            print(f"--- {rid}: model not found ---\n")
            continue

        print(f"--- {rid} ---")
        by_c = defaultdict(list)
        by_c_hard = defaultdict(list)
        for r in data:
            by_c[r["context"]].append(r["score"])
            if r["difficulty"] == "hard":
                by_c_hard[r["context"]].append(r["score"])

        headers = ["Context", "N", "Avg", "Strict", "Lenient", "H.Str", "H.Len"]
        rows = []
        for c in sorted(by_c):
            a = accuracy(by_c[c])
            h = accuracy(by_c_hard.get(c, []))
            rows.append([c, a["n"], a["avg"], fmt_pct(a["strict"]), fmt_pct(a["lenient"]),
                          fmt_pct(h["strict"]), fmt_pct(h["lenient"])])
        print_table(headers, rows)
        print()


def compare_by_question(runs_data, run_ids, model_filter=None, context_filter=None):
    print("\n=== BY QUESTION ===\n")
    filters = []
    if model_filter:
        filters.append(f"model={model_filter}")
    if context_filter:
        filters.append(f"context={context_filter}")
    if filters:
        print(f"  [filtered: {', '.join(filters)}]\n")

    headers = ["Q_ID", "Diff", "Cat"] + run_ids
    rows = []

    scores_by_q = {rid: defaultdict(list) for rid in run_ids}
    meta = {}
    for rid in run_ids:
        for r in runs_data[rid]:
            if model_filter and r["model"] != model_filter:
                continue
            if context_filter and r["context"] != context_filter:
                continue
            scores_by_q[rid][r["question_id"]].append(r["score"])
            meta[r["question_id"]] = (r["difficulty"], r["category"])

    all_qids = sorted(set().union(*(scores_by_q[rid].keys() for rid in run_ids)), key=lambda x: int(x))
    for qid in all_qids:
        diff, cat = meta.get(qid, ("?", "?"))
        row = [f"Q{qid}", diff, cat]
        for rid in run_ids:
            scores = scores_by_q[rid].get(qid, [])
            if scores:
                avg = sum(scores) / len(scores)
                row.append(f"{avg:.2f} (n={len(scores)})")
            else:
                row.append("—")
        rows.append(row)
    print_table(headers, rows)


def compare_by_category(runs_data, run_ids, model_filter=None):
    print("\n=== BY CATEGORY ===\n")
    if model_filter:
        print(f"  [filtered to model: {model_filter}]\n")

    scores_by_cat = {rid: defaultdict(list) for rid in run_ids}
    for rid in run_ids:
        for r in runs_data[rid]:
            if model_filter and r["model"] != model_filter:
                continue
            scores_by_cat[rid][r["category"]].append(r["score"])

    all_cats = sorted(set().union(*(scores_by_cat[rid].keys() for rid in run_ids)))
    headers = ["Category"] + [f"{rid} (avg)" for rid in run_ids] + [f"{rid} (len)" for rid in run_ids]
    rows = []
    for cat in all_cats:
        row = [cat]
        for rid in run_ids:
            scores = scores_by_cat[rid].get(cat, [])
            row.append(f"{sum(scores)/len(scores):.2f}" if scores else "—")
        for rid in run_ids:
            scores = scores_by_cat[rid].get(cat, [])
            row.append(fmt_pct(sum(1 for s in scores if s >= 2) / len(scores)) if scores else "—")
        rows.append(row)
    print_table(headers, rows)


def compare_by_difficulty(runs_data, run_ids, model_filter=None):
    print("\n=== BY DIFFICULTY ===\n")
    if model_filter:
        print(f"  [filtered to model: {model_filter}]\n")

    scores_by_diff = {rid: defaultdict(list) for rid in run_ids}
    for rid in run_ids:
        for r in runs_data[rid]:
            if model_filter and r["model"] != model_filter:
                continue
            scores_by_diff[rid][r["difficulty"]].append(r["score"])

    headers = ["Difficulty"] + [f"{rid} (avg)" for rid in run_ids] + [f"{rid} (strict)" for rid in run_ids] + [f"{rid} (lenient)" for rid in run_ids]
    rows = []
    for diff in ["easy", "medium", "hard"]:
        row = [diff]
        for rid in run_ids:
            scores = scores_by_diff[rid].get(diff, [])
            row.append(f"{sum(scores)/len(scores):.2f}" if scores else "—")
        for rid in run_ids:
            scores = scores_by_diff[rid].get(diff, [])
            row.append(fmt_pct(sum(1 for s in scores if s == 3) / len(scores)) if scores else "—")
        for rid in run_ids:
            scores = scores_by_diff[rid].get(diff, [])
            row.append(fmt_pct(sum(1 for s in scores if s >= 2) / len(scores)) if scores else "—")
        rows.append(row)
    print_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across runs")
    parser.add_argument("runs", nargs="*", help="Run IDs to compare (default: all)")
    parser.add_argument("--model", help="Filter to a specific model (also shows all configs)")
    parser.add_argument("--context", help="Filter to a specific context")
    parser.add_argument("--by-question", action="store_true", help="Show per-question breakdown")
    parser.add_argument("--by-category", action="store_true", help="Show per-category breakdown")
    parser.add_argument("--by-difficulty", action="store_true", help="Show per-difficulty breakdown")
    args = parser.parse_args()

    available = discover_runs()
    if not available:
        print("No evaluated_results_*.csv files found in results/")
        return

    if args.runs:
        run_ids = args.runs
        missing = [r for r in run_ids if r not in available]
        if missing:
            print(f"Runs not found: {missing}")
            print(f"Available: {list(available.keys())}")
            return
    else:
        run_ids = list(available.keys())

    print(f"Comparing {len(run_ids)} runs: {', '.join(run_ids)}")

    runs_data = {rid: load_run(available[rid]) for rid in run_ids}

    compare_overall(runs_data, run_ids)

    if args.model:
        compare_single_model(runs_data, run_ids, args.model)
    else:
        compare_by_model(runs_data, run_ids, context_filter=args.context)

    if args.by_difficulty or not (args.by_question or args.by_category):
        compare_by_difficulty(runs_data, run_ids, model_filter=args.model)

    if args.by_category:
        compare_by_category(runs_data, run_ids, model_filter=args.model)

    if args.by_question:
        compare_by_question(runs_data, run_ids, model_filter=args.model, context_filter=args.context)


if __name__ == "__main__":
    main()
