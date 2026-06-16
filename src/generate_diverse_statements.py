"""
Diversified Statement Generator for Kyrgyz Budget Data (2026)
Produces 3-5 varied phrasings per budget row in both English and Russian.

Variation dimensions:
  - Word order (fronting amount vs. fronting entity)
  - Detail level (full context chain vs. abbreviated)
  - Number format (thousands som vs. billions som vs. raw)
  - Sentence structure (active vs. passive vs. nominal)
  - Language (EN / RU)
"""

import pandas as pd
import json
import math
from pathlib import Path

from budget_common import (
    COL_GRBS, COL_PROG, COL_MEASURE, COL_NAME,
    COL_FUND_2026, COL_FUND_2027, COL_FUND_2028,
    COL_INDICATOR, COL_UNIT, COL_BASELINE, COL_TARGET_2027, COL_TARGET_2028,
    COL_ROW_TYPE,
    safe_str, fmt_amount, detect_row_type, language_specific_results,
)

ROOT = Path(__file__).resolve().parents[1]
STATEMENTS_DIR = ROOT / "data" / "statements"

EXCEL_FILE = ROOT / "reports" / "Annotated_Appendix14_2026.xlsx"
SHEET_NAME = "prilojenie_11_4"
OUTPUT_FILE = STATEMENTS_DIR / "statements_2026_diverse.json"
OUTPUT_FILE_EN = STATEMENTS_DIR / "statements_en_diverse_2026.json"
OUTPUT_FILE_RU = STATEMENTS_DIR / "statements_ru_diverse_2026.json"


def fmt_thou(val):
    """Format as thousands som with comma separator."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return f"{val:,.1f}"


def fmt_bln(val):
    """Format as billions som (divide by 1_000_000)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    bln = val / 1_000_000
    if bln >= 1:
        return f"{bln:.1f} млрд сом"
    elif bln >= 0.001:
        return f"{val/1000:.1f} млн сом"
    else:
        return f"{val:,.1f} тыс. сом"


def fmt_bln_en(val):
    """Format as billions/millions som in English."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    bln = val / 1_000_000
    if bln >= 1:
        return f"{bln:.1f} billion som"
    elif bln >= 0.001:
        return f"{val/1000:.1f} million som"
    else:
        return f"{val:,.1f} thousand som"


def get_val(r, col):
    v = r[col]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v



# ---------------------------------------------------------------------------
# TEMPLATE FUNCTIONS — each returns a list of statement strings
# ---------------------------------------------------------------------------

def gen_ministry_statements(ministry_name, ministry_code):
    """Generate 3-4 diverse statements for a MINISTRY row."""
    en = [
        f"Ministry '{ministry_name}' (code {ministry_code}) is allocated budget programs in the 2026 republican budget.",
        f"The 2026 program-based budget includes allocations for '{ministry_name}' (GRBS code {ministry_code}).",
        f"Budget agency code {ministry_code} corresponds to '{ministry_name}' in the 2026 fiscal year.",
    ]
    ru = [
        f"Министерство/ведомство «{ministry_name}» (код ГРБС {ministry_code}) включено в программный бюджет на 2026 год.",
        f"В бюджете 2026 года предусмотрены программы для «{ministry_name}» (код {ministry_code}).",
        f"Код ГРБС {ministry_code} соответствует ведомству «{ministry_name}» в республиканском бюджете на 2026 год.",
    ]
    return en, ru


def gen_program_statements(ministry_name, ministry_code, program_name, program_code, fund_2026, fund_2027, fund_2028, indicator_text=None):
    """Generate 4-5 diverse statements for a PROGRAM row."""
    t26 = fmt_thou(fund_2026)
    t27 = fmt_thou(fund_2027)
    t28 = fmt_thou(fund_2028)
    b26 = fmt_bln_en(fund_2026)
    b26r = fmt_bln(fund_2026)

    en = []
    ru = []

    # Pattern 1: Full context, thousands
    if t26:
        en.append(f"Under ministry '{ministry_name}' (code {ministry_code}), the program '{program_name}' (code {program_code}) is allocated {t26} thousand som in 2026, {t27} thousand som in 2027, {t28} thousand som in 2028.")
    # Pattern 2: Amount-fronted, billions
    if b26:
        en.append(f"In 2026, {b26} is allocated to the program '{program_name}' under '{ministry_name}'.")
    # Pattern 3: Abbreviated, code-heavy
    if t26:
        en.append(f"Program {ministry_code}/{program_code} '{program_name}': 2026 funding = {t26} тыс. сом; 2027 = {t27}; 2028 = {t28}.")
    # Pattern 4: Growth-oriented (if multi-year data)
    if fund_2026 and fund_2027 and fund_2026 != 0:
        pct = (fund_2027 - fund_2026) / fund_2026 * 100
        direction = "increases" if pct > 0 else "decreases"
        en.append(f"Funding for '{program_name}' ({ministry_name}) {direction} by {abs(pct):.1f}% from 2026 to 2027.")

    # Russian variants
    if t26:
        ru.append(f"Программа «{program_name}» (код {program_code}) ведомства «{ministry_name}» (код {ministry_code}) получает {t26} тыс. сом на 2026 год, {t27} — на 2027, {t28} — на 2028.")
    if b26r:
        ru.append(f"На программу «{program_name}» в 2026 году выделено {b26r} в рамках бюджета «{ministry_name}».")
    if t26:
        ru.append(f"Финансирование программы {ministry_code}/{program_code} «{program_name}»: {t26} тыс. сом (2026), {t27} (2027), {t28} (2028).")
    if fund_2026 and fund_2027 and fund_2026 != 0:
        pct = (fund_2027 - fund_2026) / fund_2026 * 100
        direction_ru = "увеличивается" if pct > 0 else "уменьшается"
        ru.append(f"Финансирование «{program_name}» ({ministry_name}) {direction_ru} на {abs(pct):.1f}% с 2026 по 2027 год.")

    # Indicator statement (if present)
    if indicator_text:
        en.append(f"The performance indicator for program '{program_name}' is: {indicator_text}.")
        ru.append(f"Показатель результативности программы «{program_name}»: {indicator_text}.")

    return en, ru


def gen_measure_statements(ministry_name, program_name, measure_name, measure_code, fund_2026, fund_2027, fund_2028, indicator_text=None):
    """Generate 3-5 diverse statements for a MEASURE row."""
    t26 = fmt_thou(fund_2026)
    t27 = fmt_thou(fund_2027)
    t28 = fmt_thou(fund_2028)
    b26 = fmt_bln_en(fund_2026)
    b26r = fmt_bln(fund_2026)

    en = []
    ru = []

    # Pattern 1: Full hierarchy
    if t26:
        en.append(f"Under '{ministry_name}', program '{program_name}', the budget measure '{measure_name}' (code {measure_code}) is allocated {t26} thousand som in 2026, {t27} in 2027, {t28} in 2028.")
    # Pattern 2: Measure-focused
    if b26:
        en.append(f"Budget measure '{measure_name}' receives {b26} in 2026 (part of program '{program_name}').")
    # Pattern 3: Compact tabular
    if t26:
        en.append(f"Measure {measure_code} '{measure_name}' [{program_name}]: {t26} / {t27} / {t28} тыс. сом (2026/2027/2028).")

    # Russian
    if t26:
        ru.append(f"Бюджетная мера «{measure_name}» (код {measure_code}) программы «{program_name}» ведомства «{ministry_name}» финансируется в размере {t26} тыс. сом на 2026, {t27} — на 2027, {t28} — на 2028.")
    if b26r:
        ru.append(f"На меру «{measure_name}» в рамках программы «{program_name}» выделено {b26r} в 2026 году.")
    if t26:
        ru.append(f"Мера {measure_code} «{measure_name}» [{program_name}]: {t26} / {t27} / {t28} тыс. сом (2026/2027/2028).")

    if indicator_text:
        en.append(f"Performance indicator for measure '{measure_name}': {indicator_text}.")
        ru.append(f"Показатель результативности меры «{measure_name}»: {indicator_text}.")

    return en, ru


def gen_indicator_statements(indicator_name, unit, baseline, target_2027, target_2028, parent_context_en, parent_context_ru):
    """Generate 3-4 diverse statements for an INDICATOR row."""
    unit = unit or "units"

    en = []
    ru = []

    # Pattern 1: Full
    parts_en = [f"Indicator '{indicator_name}' (unit: {unit})"]
    parts_ru = [f"Показатель «{indicator_name}» (ед. изм.: {unit})"]
    if baseline:
        parts_en.append(f"baseline: {baseline}")
        parts_ru.append(f"базовое значение: {baseline}")
    if target_2027:
        parts_en.append(f"target 2027: {target_2027}")
        parts_ru.append(f"целевое 2027: {target_2027}")
    if target_2028:
        parts_en.append(f"target 2028: {target_2028}")
        parts_ru.append(f"целевое 2028: {target_2028}")

    en.append(f"For {parent_context_en}: {'; '.join(parts_en)}.")
    ru.append(f"Для {parent_context_ru}: {'; '.join(parts_ru)}.")

    # Pattern 2: Target-focused
    if target_2027 and baseline:
        en.append(f"The target for '{indicator_name}' changes from {baseline} (baseline) to {target_2027} by 2027 ({parent_context_en}).")
        ru.append(f"Целевое значение «{indicator_name}» меняется с {baseline} (базовое) до {target_2027} к 2027 году ({parent_context_ru}).")

    # Pattern 3: Compact
    targets_str = f"{target_2027 or '—'}/{target_2028 or '—'}"
    en.append(f"KPI: '{indicator_name}' [{unit}] — base {baseline or '—'}, targets {targets_str} (2027/2028).")
    ru.append(f"КПЭ: «{indicator_name}» [{unit}] — база {baseline or '—'}, цели {targets_str} (2027/2028).")

    return en, ru


def gen_total_statements(ministry_name, ministry_code, fund_2026, fund_2027, fund_2028):
    """Generate 3 diverse statements for a TOTAL row."""
    t26 = fmt_thou(fund_2026)
    t27 = fmt_thou(fund_2027)
    t28 = fmt_thou(fund_2028)
    b26 = fmt_bln_en(fund_2026)
    b26r = fmt_bln(fund_2026)

    en = [
        f"Total for ministry '{ministry_name}' (code {ministry_code}): {t26} thousand som in 2026, {t27} in 2027, {t28} in 2028.",
    ]
    ru = [
        f"Итого по ведомству «{ministry_name}» (код {ministry_code}): {t26} тыс. сом в 2026, {t27} — в 2027, {t28} — в 2028.",
    ]
    if b26:
        en.append(f"The total 2026 allocation for '{ministry_name}' is {b26}.")
        ru.append(f"Общий объём финансирования «{ministry_name}» на 2026 год составляет {b26r}.")
    if fund_2026 and fund_2027 and fund_2026 != 0:
        pct = (fund_2027 - fund_2026) / fund_2026 * 100
        direction = "grows" if pct > 0 else "shrinks"
        direction_ru = "растёт" if pct > 0 else "сокращается"
        en.append(f"Total budget for '{ministry_name}' {direction} by {abs(pct):.1f}% from 2026 to 2027.")
        ru.append(f"Общий бюджет «{ministry_name}» {direction_ru} на {abs(pct):.1f}% с 2026 по 2027 год.")

    return en, ru


def build_indicator_text(r):
    """Build a compact indicator string from a row's indicator columns."""
    indicator = safe_str(r[COL_INDICATOR])
    if not indicator:
        return None
    unit = safe_str(r[COL_UNIT]) or "units"
    baseline = safe_str(r[COL_BASELINE])
    t2027 = safe_str(r[COL_TARGET_2027])
    t2028 = safe_str(r[COL_TARGET_2028])
    parts = [f"'{indicator}' ({unit})"]
    if baseline:
        parts.append(f"base={baseline}")
    if t2027:
        parts.append(f"2027={t2027}")
    if t2028:
        parts.append(f"2028={t2028}")
    return ", ".join(parts)


def generate_all(df):
    ministry_name = None
    ministry_code = None
    program_name = None
    program_code = None
    measure_name = None
    parent_context_en = "the budget item"
    parent_context_ru = "бюджетной статьи"

    results = []

    for idx in range(6, len(df)):
        r = df.iloc[idx]
        annotated_type = safe_str(r[COL_ROW_TYPE])
        if annotated_type in ("EMPTY", "OTHER", "Row Type"):
            continue

        row_type = detect_row_type(r)
        en_stmts = []
        ru_stmts = []

        if row_type == "MINISTRY":
            ministry_code = safe_str(r[COL_GRBS])
            ministry_name = safe_str(r[COL_PROG]) or safe_str(r[COL_NAME])
            program_name = None
            program_code = None
            measure_name = None
            en_stmts, ru_stmts = gen_ministry_statements(ministry_name, ministry_code)
            parent_context_en = f"ministry '{ministry_name}'"
            parent_context_ru = f"ведомства «{ministry_name}»"

        elif row_type == "PROGRAM":
            program_code = safe_str(r[COL_PROG])
            program_name = safe_str(r[COL_NAME])
            measure_name = None
            fund_2026 = get_val(r, COL_FUND_2026)
            fund_2027 = get_val(r, COL_FUND_2027)
            fund_2028 = get_val(r, COL_FUND_2028)
            indicator_text = build_indicator_text(r)
            en_stmts, ru_stmts = gen_program_statements(
                ministry_name, ministry_code, program_name, program_code,
                fund_2026, fund_2027, fund_2028, indicator_text
            )
            parent_context_en = f"program '{program_name}' under '{ministry_name}'"
            parent_context_ru = f"программы «{program_name}» ведомства «{ministry_name}»"

        elif row_type == "MEASURE":
            measure_code = safe_str(r[COL_MEASURE])
            measure_name = safe_str(r[COL_NAME])
            fund_2026 = get_val(r, COL_FUND_2026)
            fund_2027 = get_val(r, COL_FUND_2027)
            fund_2028 = get_val(r, COL_FUND_2028)
            indicator_text = build_indicator_text(r)
            en_stmts, ru_stmts = gen_measure_statements(
                ministry_name, program_name, measure_name, measure_code,
                fund_2026, fund_2027, fund_2028, indicator_text
            )
            parent_context_en = f"measure '{measure_name}' under program '{program_name}'"
            parent_context_ru = f"меры «{measure_name}» программы «{program_name}»"

        elif row_type == "INDICATOR":
            indicator_name = safe_str(r[COL_INDICATOR])
            unit = safe_str(r[COL_UNIT])
            baseline = safe_str(r[COL_BASELINE])
            t2027 = safe_str(r[COL_TARGET_2027])
            t2028 = safe_str(r[COL_TARGET_2028])
            if indicator_name:
                en_stmts, ru_stmts = gen_indicator_statements(
                    indicator_name, unit, baseline, t2027, t2028,
                    parent_context_en, parent_context_ru
                )

        elif row_type == "TOTAL":
            fund_2026 = get_val(r, COL_FUND_2026)
            fund_2027 = get_val(r, COL_FUND_2027)
            fund_2028 = get_val(r, COL_FUND_2028)
            en_stmts, ru_stmts = gen_total_statements(
                ministry_name, ministry_code, fund_2026, fund_2027, fund_2028
            )

        if en_stmts or ru_stmts:
            results.append({
                "row_index": idx,
                "row_type": row_type,
                "ministry_code": ministry_code,
                "ministry_name": ministry_name,
                "program_code": program_code if row_type in ("PROGRAM", "MEASURE", "INDICATOR") else None,
                "program_name": program_name if row_type in ("PROGRAM", "MEASURE", "INDICATOR") else None,
                "statements_en": en_stmts,
                "statements_ru": ru_stmts,
                "original_values": {
                    "name": safe_str(r[COL_NAME]),
                    "fund_2026": get_val(r, COL_FUND_2026),
                    "fund_2027": get_val(r, COL_FUND_2027),
                    "fund_2028": get_val(r, COL_FUND_2028),
                    "indicator": safe_str(r[COL_INDICATOR]),
                    "unit": safe_str(r[COL_UNIT]),
                    "baseline": safe_str(r[COL_BASELINE]),
                    "target_2027": safe_str(r[COL_TARGET_2027]),
                    "target_2028": safe_str(r[COL_TARGET_2028]),
                },
            })

    return results


def print_stats(results):
    from collections import Counter
    types = Counter(r["row_type"] for r in results)
    total_en = sum(len(r["statements_en"]) for r in results)
    total_ru = sum(len(r["statements_ru"]) for r in results)
    avg_en = total_en / len(results) if results else 0
    avg_ru = total_ru / len(results) if results else 0

    print(f"Rows processed: {len(results)}")
    print(f"Row types: {dict(types)}")
    print(f"Total EN statements: {total_en} (avg {avg_en:.1f}/row)")
    print(f"Total RU statements: {total_ru} (avg {avg_ru:.1f}/row)")
    print(f"Combined: {total_en + total_ru} statements")



def main():
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None)
    results = generate_all(df)
    print_stats(results)

    # Show samples
    print("\n--- Sample outputs ---")
    for r in results[:3]:
        print(f"\n[Row {r['row_index']}] ({r['row_type']}) {r.get('ministry_name', '')}")
        print("  EN:")
        for s in r["statements_en"]:
            print(f"    • {s}")
        print("  RU:")
        for s in r["statements_ru"]:
            print(f"    • {s}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")

    for language, output_file in [("en", OUTPUT_FILE_EN), ("ru", OUTPUT_FILE_RU)]:
        language_results = language_specific_results(results, language)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(language_results, f, ensure_ascii=False, indent=2)
        statement_count = sum(len(r["statements"]) for r in language_results)
        print(f"{language.upper()} output saved to {output_file} ({statement_count} statements)")


if __name__ == "__main__":
    main()
