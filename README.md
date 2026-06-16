# Evidence-Grounded QA over the Kyrgyz Republic Program-Based Budget

NLP course project (CTU FEE). The project evaluates how well LLMs answer
evidence-grounded questions over the Kyrgyz Republic's 2026 program-based
budget (Приложение 14), comparing three context representations —
natural-language **statements**, raw **markdown** tables, and **structured**
JSON with category-aware retrieval — across four iterative experiment rounds
and ~5,160 API calls.

**Final deliverables:** [`deliverables/report_v2.md`](deliverables/report_v2.md)
(also `.docx`) and [`deliverables/presentation_v2.pptx`](deliverables/presentation_v2.pptx).

## Project Layout

```
├── data/
│   ├── raw/               Original budget Excel files (2023–2026) + annotated workbooks
│   ├── processed/         budget_2026_full.json / .md (561 rows: 168 programs, 393 measures)
│   ├── statements/        Generated natural-language statements (RU/EN, plain + diverse)
│   └── qa/                20-question benchmark with verified reference answers (RU/EN)
├── src/                   Core pipeline (see table below)
├── scripts/
│   ├── patches/           One-off corrections (reference fixes, Q20 re-judging)
│   └── presentation/      Slide-deck build/patch scripts + reusable slides_kit.py
├── results/
│   ├── raw/               Raw model outputs (all_results_*.csv/.json)
│   ├── evaluated/         LLM-as-Judge per-row scores (evaluated_results_*.csv)
│   ├── summaries/         Per-run accuracy summaries (evaluation_summary_*.csv)
│   ├── retrieval/         Retrieval inspection + quality (retrieval_*_all, tfidf_retrieval_*)
│   ├── human_validation/  Annotation sheet, key, and Q20-rejudged key for judge validation
│   ├── failure_analysis/  2×2 pipeline failure analysis + summary (Round 1, EN statements)
│   ├── logs/              Experiment run logs
│   └── archive/           Smoke tests and superseded runs
│   (Analysis scripts write to results/ root by default; files are sorted into these
│    subfolders for navigation — see "_q20_rejudged" files for the canonical numbers.)
├── notebooks/             Error-analysis notebook
├── deliverables/          Final report, presentation, speaker notes (archive/ = drafts)
└── reports/               Early project documents (proposal, feasibility test, annotations)
```

## Core Pipeline (`src/`)

| Stage | Script | Purpose |
|---|---|---|
| Data prep | `generate_structured_json.py` | Excel → structured JSON (incl. orphan-program promotion) |
| | `generate_statements.py` | Template-based RU/EN statements from budget rows |
| | `generate_diverse_statements.py` | Paraphrased statement variants |
| | `split_qa_pairs.py` | Split combined QA file into per-language files |
| Retrieval | `structured_retriever.py` | Category-aware TF-IDF with ministry expansion |
| | `evaluate_tfidf_retrieval.py` | Retrieval quality (contains-reference rate) |
| | `inspect_tfidf_retrieval.py` / `inspect_structured_retrieval.py` | Per-question retrieval debugging (`--all` for CSV export) |
| Experiments | `run_experiments.py` | Main loop: models × languages × contexts (Rounds 1–3) |
| | `run_statements_experiment.py` | Round 4: expanded statement retrieval (top-500 / all) |
| Evaluation | `evaluate_results.py` | LLM-as-Judge scoring (gpt-oss-120b, 0–3 scale) |
| | `compare_runs.py` | Cross-run comparison (by model / category / difficulty) |
| | `analyze_pipeline_failures.py` | 2×2 failure matrix: retrieval hit/miss × model right/wrong |
| Validation | `generate_validation_sample.py` | Stratified sample for human annotation |
| | `compute_judge_agreement.py` | Human–judge agreement (Cohen's κ, confusion matrix) |
| Shared | `utils.py`, `budget_common.py` | Paths, env loading, shared helpers |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root (not committed):

```
METACENTRUM_BASE_URL=<OpenAI-compatible endpoint>
METACENTRUM_API_KEY=<key>
```

## Reproducing the Pipeline

```bash
# 1. Data preparation (Excel → JSON/markdown/statements)
python src/generate_structured_json.py
python src/generate_statements.py

# 2. Run experiments (writes results/all_results_<run_id>.csv)
python src/run_experiments.py                 # Rounds 1–3 style grid
python src/run_statements_experiment.py       # Round 4 statement ablation

# 3. Judge the answers
EVALUATION_INPUT_CSV=results/all_results_v5_statements.csv \
EXPERIMENT_RUN_ID=v5_statements python src/evaluate_results.py

# 4. Compare runs
python src/compare_runs.py v4_round3 v5_statements --by-category

# 5. Human validation of the judge
python src/generate_validation_sample.py --run v4_round3 --n 60 --seed 42
# ... fill in human_score in results/human_validation_v4_round3.csv ...
python src/compute_judge_agreement.py --run v4_round3
# (scripts read/write results/ root; the saved sheets live in results/human_validation/)
```

One-off historical corrections (already applied; kept for provenance) live in
`scripts/patches/` and are run from the project root, e.g.
`python scripts/patches/rejudge_q20.py`.

## Key Results

| Round | Change | Qwen 3.5 best config | Strict | Lenient |
|---|---|---|---|---|
| 1 | Baseline (statements + markdown, EN) | markdown_full | 35% | 80% |
| 2 | + structured retrieval, bug fixes | ru_markdown_full | 39% | 83% |
| 3 | + new models, scratchpad CoT | ru_structured_full | 50% | 90% |
| 4 | + expanded statement top-k | ru_statements_all | 58% | 84% |

- Best overall: DeepSeek V4 Pro / Kimi K2.6 at 2.80 avg, 100% lenient (`ru_markdown_full`, Round 3)
- LLM-as-Judge validated against human annotation: κ = 0.763 (substantial), F1 = 0.87
- Multi-step questions remain unsolved (< 25% lenient in all configurations)

See `deliverables/report_v2.md` for the full analysis, limitations, and
per-round details.
