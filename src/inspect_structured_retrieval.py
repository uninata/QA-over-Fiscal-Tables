"""
Inspect what the structured retriever pulls for a given question.

Examples:
  python3 src/inspect_structured_retrieval.py --language ru --question-id 1
  python3 src/inspect_structured_retrieval.py --language en --question-id 6 --budget 16000
  python3 src/inspect_structured_retrieval.py --language ru --question-id 4 --budget full
  python3 src/inspect_structured_retrieval.py --language ru --all --budget full --output results/retrieval_structured_all.csv
"""

import argparse
import csv

import run_experiments as experiments
from structured_retriever import StructuredRetriever, load_structured_data
from evaluate_tfidf_retrieval import contains_exact_reference, number_coverage, is_substantive_number


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect structured retrieval for a question")
    parser.add_argument("--language", choices=["en", "ru"], default="ru")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question-id", type=int, help="Single question ID")
    group.add_argument("--all", action="store_true", help="Run all questions")
    parser.add_argument("--budget", type=str, default="8000",
                        help="Token budget: 2000, 8000, 16000, or full")
    parser.add_argument("--output", default="", help="Optional CSV output path")
    return parser.parse_args()


def parse_budget(value):
    if value == "full":
        return None
    return int(value)


def inspect_one(qa, retriever, rows, char_budget, budget_label, language, verbose=True):
    category = qa.get("category", "")
    context = retriever.retrieve(qa["question"], category=category, char_budget=char_budget)

    reference = qa["answer"]
    exact_match = contains_exact_reference(context, reference)
    ref_numbers, found_numbers = number_coverage(context, reference)
    substantive_ref = [n for n in ref_numbers if is_substantive_number(n)]
    all_numbers_found = bool(substantive_ref) and all(n in found_numbers for n in substantive_ref)
    contains_answer = exact_match or all_numbers_found

    if verbose:
        print(f"{'='*70}")
        print(f"  Language: {language}")
        print(f"  Question {qa['id']} [{qa['difficulty']}, {category}]:")
        print(f"    {qa['question']}")
        print(f"  Reference answer: {reference}")
        print(f"  Budget: {budget_label} ({char_budget or 'unlimited'} chars)")
        print(f"{'='*70}")
        print()

        scored = retriever._score_query(qa["question"], top_k=10)
        print("TF-IDF top matches (before expansion):")
        for idx, score in scored[:10]:
            r = rows[idx]
            print(f"  score={score:.3f}  [{r['level']:7s}] {r['ministry'][:35]} / {r['name'][:40]}")
        print()

        if category not in {"count", "filter", "indicator", "multi-step"}:
            ministry_codes_in_context = set()
            for line in context.split("\n")[1:]:
                parts = line.split(" | ")
                if parts:
                    ministry_codes_in_context.add(parts[0].strip())
            print(f"Ministries expanded: {len(ministry_codes_in_context)} codes: {sorted(ministry_codes_in_context)}")
            print()

        print("Retrieval evaluation:")
        print(f"  Exact reference match: {'YES' if exact_match else 'NO'}")
        print(f"  Reference numbers: {ref_numbers}")
        print(f"  Found numbers:     {found_numbers}")
        print(f"  All numbers found: {'YES' if all_numbers_found else 'NO'}")
        print(f"  Contains answer:   {'YES' if contains_answer else 'NO'}")
        print()

        lines = context.split("\n")
        print(f"Retrieved context ({len(context)} chars, {len(lines)} lines):")
        print("-" * 70)
        print(context)
        print("-" * 70)

    return {
        "question_id": qa["id"],
        "difficulty": qa["difficulty"],
        "category": category,
        "question": qa["question"],
        "reference_answer": reference,
        "exact_match": exact_match,
        "all_numbers_found": all_numbers_found,
        "contains_answer": contains_answer,
        "context_chars": len(context),
        "context_lines": len(context.split("\n")),
        "context": context,
    }


def main():
    args = parse_args()
    token_budget = parse_budget(args.budget)
    char_budget = token_budget * experiments.CHARS_PER_TOKEN if token_budget else None

    qa_pairs = experiments.load_qa(args.language)
    rows = load_structured_data()
    retriever = StructuredRetriever(rows)

    if args.all:
        results = []
        for qa in qa_pairs:
            result = inspect_one(qa, retriever, rows, char_budget, args.budget, args.language, verbose=False)
            hit = "HIT" if result["contains_answer"] else "MISS"
            print(f"  Q{result['question_id']:>2} [{result['difficulty']:6s}, {result['category']:11s}] "
                  f"{result['context_chars']:>6} chars  {hit}")
            results.append(result)

        hits = sum(1 for r in results if r["contains_answer"])
        print(f"\nHit rate: {hits}/{len(results)} ({hits/len(results):.0%})")

        if args.output:
            fieldnames = ["question_id", "difficulty", "category", "question",
                          "reference_answer", "exact_match", "all_numbers_found",
                          "contains_answer", "context_chars", "context_lines", "context"]
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved to {args.output}")
    else:
        qa = next((row for row in qa_pairs if int(row["id"]) == args.question_id), None)
        if not qa:
            raise SystemExit(f"No question id {args.question_id} for language {args.language}")

        result = inspect_one(qa, retriever, rows, char_budget, args.budget, args.language, verbose=True)

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["line_num", "content"])
                for i, line in enumerate(result["context"].split("\n")):
                    writer.writerow([i, line])
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
