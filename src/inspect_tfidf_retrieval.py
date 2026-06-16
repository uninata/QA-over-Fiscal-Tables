"""
Inspect exactly what TF-IDF retrieves for one question.

Examples:
  python3 src/inspect_tfidf_retrieval.py --language ru --question-id 1
  STATEMENTS_FILE=data/statements/statements_2026_diverse.json \
    python3 src/inspect_tfidf_retrieval.py --language en --question-id 3 --budget 8000 --top-k 20
  python3 src/inspect_tfidf_retrieval.py --language ru --all --budget full --output results/retrieval_statements_all.csv
"""

import argparse
import csv
import re

import run_experiments as experiments
from evaluate_tfidf_retrieval import contains_exact_reference, number_coverage, is_substantive_number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "ru"], default="en")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question-id", type=int, help="Single question ID")
    group.add_argument("--all", action="store_true", help="Run all questions")
    parser.add_argument("--budget", type=str, default="2000", help="Token budget: 2000, 8000, 16000, or full")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional CSV output path")
    return parser.parse_args()


def parse_budget(value):
    if value == "full":
        return None
    return int(value)


def highlight_numbers(statement, reference_numbers):
    highlighted = statement
    for number in sorted(set(reference_numbers), key=len, reverse=True):
        if not number:
            continue
        pattern = re.escape(number)
        highlighted = re.sub(pattern, f">>{number}<<", highlighted)
    return highlighted


def inspect_one(qa, retriever, char_budget, budget_label, language, top_k=20, verbose=True):
    ranked = retriever.ranked(qa["question"], top_k=top_k)

    selected = []
    total_chars = 0
    context_parts = []
    for rank, (score, statement) in enumerate(ranked, start=1):
        line_chars = len(f"- {statement}\n")
        included = char_budget is None or total_chars + line_chars <= char_budget
        if included:
            total_chars += line_chars
            context_parts.append(f"- {statement}")

        exact_match = contains_exact_reference(statement, qa["answer"])
        reference_numbers, found_numbers = number_coverage(statement, qa["answer"])
        substantive_numbers = [n for n in reference_numbers if is_substantive_number(n)]
        all_numbers_found = bool(substantive_numbers) and all(n in found_numbers for n in substantive_numbers)

        selected.append({
            "rank": rank,
            "score": score,
            "included_in_context": int(included),
            "exact_reference_match": int(exact_match),
            "all_reference_numbers_found": int(all_numbers_found),
            "found_reference_numbers": "|".join(found_numbers),
            "statement": statement,
        })

    context = "\n".join(context_parts)
    reference = qa["answer"]
    ctx_exact = contains_exact_reference(context, reference)
    ref_numbers, ctx_found = number_coverage(context, reference)
    substantive_ref = [n for n in ref_numbers if is_substantive_number(n)]
    ctx_all_numbers = bool(substantive_ref) and all(n in ctx_found for n in substantive_ref)
    contains_answer = ctx_exact or ctx_all_numbers

    if verbose:
        print(f"Language: {language}")
        print(f"Question {qa['id']}: {qa['question']}")
        print(f"Reference: {reference}")
        print(f"Budget: {budget_label} ({char_budget or 'full'} chars)")
        print(f"Statements file: {experiments.STATEMENT_FILES[language]}")
        print()

        ref_nums_display, _ = number_coverage("", reference)
        for row in selected:
            flags = []
            if row["included_in_context"]:
                flags.append("IN")
            if row["exact_reference_match"]:
                flags.append("EXACT")
            if row["all_reference_numbers_found"]:
                flags.append("NUMBERS")
            flag_text = ",".join(flags) if flags else "-"
            statement = highlight_numbers(row["statement"], ref_nums_display)
            print(f"{row['rank']:>2}. score={row['score']:.4f} [{flag_text}]")
            print(f"    {statement}")
            print()

    return {
        "question_id": qa["id"],
        "difficulty": qa.get("difficulty", ""),
        "category": qa.get("category", ""),
        "question": qa["question"],
        "reference_answer": reference,
        "exact_match": ctx_exact,
        "all_numbers_found": ctx_all_numbers,
        "contains_answer": contains_answer,
        "context_chars": len(context),
        "context": context,
        "_selected": selected,
    }


def main():
    args = parse_args()
    token_budget = parse_budget(args.budget)
    char_budget = token_budget * experiments.CHARS_PER_TOKEN if token_budget else None

    qa_pairs = experiments.load_qa(args.language)
    statements = experiments.load_all_statements(args.language)
    retriever = experiments.StatementRetriever(statements)

    if args.all:
        results = []
        for qa in qa_pairs:
            result = inspect_one(qa, retriever, char_budget, args.budget, args.language,
                                 top_k=args.top_k, verbose=False)
            hit = "HIT" if result["contains_answer"] else "MISS"
            print(f"  Q{result['question_id']:>2} [{result['difficulty']:6s}, {result['category']:11s}] "
                  f"{result['context_chars']:>6} chars  {hit}")
            results.append(result)

        hits = sum(1 for r in results if r["contains_answer"])
        print(f"\nHit rate: {hits}/{len(results)} ({hits/len(results):.0%})")

        if args.output:
            fieldnames = ["question_id", "difficulty", "category", "question",
                          "reference_answer", "exact_match", "all_numbers_found",
                          "contains_answer", "context_chars", "context"]
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow({k: r[k] for k in fieldnames})
            print(f"Saved to {args.output}")
    else:
        qa = next((row for row in qa_pairs if int(row["id"]) == args.question_id), None)
        if not qa:
            raise SystemExit(f"No question id {args.question_id} for language {args.language}")

        result = inspect_one(qa, retriever, char_budget, args.budget, args.language,
                             top_k=args.top_k, verbose=True)

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(result["_selected"][0].keys()))
                writer.writeheader()
                writer.writerows(result["_selected"])
            print(f"Saved inspection CSV to {args.output}")


if __name__ == "__main__":
    main()
