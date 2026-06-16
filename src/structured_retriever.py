"""
Structured retriever for analytical budget questions.

Instead of retrieving prose statements, this retriever:
1. Uses TF-IDF to identify which entities (ministries, programs) the question mentions
2. Expands to pull ALL structurally related rows from the JSON
3. Formats results as a compact table the LLM can count/sum over

For full-scan categories (count, filter, indicator, multi-step, aggregation over
the whole budget), it provides the complete structured table within the char budget.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

from utils import ROOT

DATA_FILE = ROOT / "data" / "processed" / "budget_2026_full.json"

FULL_SCAN_CATEGORIES = {"count", "filter", "indicator", "multi-step"}
ENTITY_CATEGORIES = {"lookup", "aggregation", "comparison"}


def load_structured_data(path=None):
    path = path or DATA_FILE
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class StructuredRetriever:
    """TF-IDF over structured rows, with hierarchical expansion."""

    def __init__(self, rows):
        self.rows = rows
        self._build_indices()
        self._build_tfidf()

    def _build_indices(self):
        self.by_ministry = defaultdict(list)
        self.by_program = defaultdict(list)
        for i, row in enumerate(self.rows):
            self.by_ministry[row["ministry_code"]].append(i)
            prog_key = (row["ministry_code"], row["code_prog"])
            self.by_program[prog_key].append(i)

    def _build_tfidf(self):
        self.doc_texts = []
        for row in self.rows:
            text = " ".join(filter(None, [
                row.get("ministry", ""),
                row.get("name", ""),
                row.get("indicator", ""),
                row.get("ministry_code", ""),
                row.get("code_prog", ""),
            ]))
            self.doc_texts.append(text.lower())

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.backend = "sklearn"
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5),
                max_features=50000, sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(self.doc_texts)
        except ModuleNotFoundError:
            self.backend = "simple"
            self._build_simple_index()

    def _char_ngrams(self, text, min_n=3, max_n=5):
        padded = f" {text.lower()} "
        grams = []
        for n in range(min_n, max_n + 1):
            grams.extend(padded[i:i+n] for i in range(max(0, len(padded) - n + 1)))
        return grams

    def _build_simple_index(self):
        from collections import Counter
        doc_counts = []
        df = defaultdict(int)
        for text in self.doc_texts:
            counts = Counter(self._char_ngrams(text))
            doc_counts.append(counts)
            for gram in counts:
                df[gram] += 1
        n_docs = len(self.doc_texts)
        self.idf = {g: math.log((1 + n_docs) / (1 + f)) + 1 for g, f in df.items()}
        self.doc_vectors = []
        for counts in doc_counts:
            vec = {g: (1 + math.log(c)) * self.idf[g] for g, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self.doc_vectors.append((vec, norm))

    def _score_query(self, question, top_k=20):
        question_lower = question.lower()
        if self.backend == "sklearn":
            q_vec = self.vectorizer.transform([question_lower])
            scores = (self.matrix @ q_vec.T).toarray().squeeze()
            top_idx = scores.argsort()[-top_k:][::-1]
            return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]
        else:
            from collections import Counter
            q_counts = Counter(self._char_ngrams(question_lower))
            q_vec = {g: (1 + math.log(c)) * self.idf.get(g, 0.0)
                     for g, c in q_counts.items() if g in self.idf}
            q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0
            scores = []
            for idx, (doc_vec, doc_norm) in enumerate(self.doc_vectors):
                score = sum(w * doc_vec.get(g, 0.0) for g, w in q_vec.items())
                if score > 0:
                    scores.append((idx, score / (doc_norm * q_norm)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    def _expand_indices(self, scored_indices):
        """Expand scored row indices to include all rows in the same ministries.
        Orders results by best TF-IDF score per ministry (most relevant ministry first).
        """
        ministry_best_score = {}
        for idx, score in scored_indices:
            code = self.rows[idx]["ministry_code"]
            if code not in ministry_best_score or score > ministry_best_score[code]:
                ministry_best_score[code] = score

        ranked_ministries = sorted(ministry_best_score.keys(),
                                   key=lambda c: ministry_best_score[c], reverse=True)
        expanded = []
        for code in ranked_ministries:
            expanded.extend(sorted(self.by_ministry[code]))
        return expanded

    def retrieve(self, question, category=None, char_budget=None):
        """Retrieve structured rows based on question category.

        For full-scan categories: return the entire table (truncated to budget).
        For entity categories: TF-IDF match → expand to full ministry → format.
        """
        if category in FULL_SCAN_CATEGORIES:
            indices = list(range(len(self.rows)))
            level_filter = self._level_filter_for_category(category, question)
        else:
            scored = self._score_query(question, top_k=10)
            if not scored:
                indices = list(range(len(self.rows)))
            else:
                indices = self._expand_indices(scored[:5])
            level_filter = None

        if level_filter:
            indices = [i for i in indices if self.rows[i]["level"] in level_filter]

        return self._format_rows(indices, char_budget)

    def _level_filter_for_category(self, category, question):
        """Determine if we should filter to programs-only for compactness."""
        question_lower = question.lower()
        if "мер" in question_lower or "measure" in question_lower:
            return None  # need measures too
        if category in ("count", "filter", "comparison", "aggregation"):
            return {"program"}
        return None

    def _format_rows(self, indices, char_budget=None):
        """Format selected rows as a compact table."""
        header = "ministry_code | level | ministry | program | name | funding_2026 | funding_2027 | funding_2028 | indicator | unit"
        lines = [header]
        for idx in indices:
            row = self.rows[idx]
            line = " | ".join([
                row.get("ministry_code", ""),
                row.get("level", ""),
                row.get("ministry", "")[:40],
                row.get("code_prog", "") or "",
                row.get("name", "")[:50],
                f"{row['funding_2026']:,.1f}" if row.get("funding_2026") else "",
                f"{row['funding_2027']:,.1f}" if row.get("funding_2027") else "",
                f"{row['funding_2028']:,.1f}" if row.get("funding_2028") else "",
                (row.get("indicator") or "")[:40],
                row.get("indicator_unit") or "",
            ])
            if char_budget and sum(len(l) + 1 for l in lines) + len(line) + 1 > char_budget:
                break
            lines.append(line)
        return "\n".join(lines)
