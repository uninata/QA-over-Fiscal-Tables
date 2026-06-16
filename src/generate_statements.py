import pandas as pd
import json
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
OUTPUT_FILE = STATEMENTS_DIR / "statements_2026.json"
OUTPUT_FILE_EN = STATEMENTS_DIR / "statements_en_2026.json"
OUTPUT_FILE_RU = STATEMENTS_DIR / "statements_ru_2026.json"


def build_funding_phrase(r):
    amounts = []
    for year, col in [("2026", COL_FUND_2026), ("2027", COL_FUND_2027), ("2028", COL_FUND_2028)]:
        amt = fmt_amount(r[col])
        if amt:
            amounts.append(f"{amt} thousand som in {year}")
    return ", ".join(amounts) if amounts else None


def build_funding_phrase_ru(r):
    amounts = []
    for year, col in [("2026", COL_FUND_2026), ("2027", COL_FUND_2027), ("2028", COL_FUND_2028)]:
        amt = fmt_amount(r[col])
        if amt:
            amounts.append(f"{amt} тыс. сом в {year} году")
    return ", ".join(amounts) if amounts else None


def build_indicator_phrase(r):
    indicator = safe_str(r[COL_INDICATOR])
    if not indicator:
        return None
    unit = safe_str(r[COL_UNIT]) or "units"
    baseline = safe_str(r[COL_BASELINE])
    t2027 = safe_str(r[COL_TARGET_2027])
    t2028 = safe_str(r[COL_TARGET_2028])
    parts = [f"Performance indicator: '{indicator}', measured in {unit}"]
    if baseline:
        parts.append(f"baseline value: {baseline}")
    targets = []
    if t2027:
        targets.append(f"{t2027} (2027)")
    if t2028:
        targets.append(f"{t2028} (2028)")
    if targets:
        parts.append(f"targets: {', '.join(targets)}")
    return "; ".join(parts) + "."


def build_indicator_phrase_ru(r):
    indicator = safe_str(r[COL_INDICATOR])
    if not indicator:
        return None
    unit = safe_str(r[COL_UNIT]) or "ед."
    baseline = safe_str(r[COL_BASELINE])
    t2027 = safe_str(r[COL_TARGET_2027])
    t2028 = safe_str(r[COL_TARGET_2028])
    parts = [f"Показатель результативности: «{indicator}», единица измерения: {unit}"]
    if baseline:
        parts.append(f"базовое значение: {baseline}")
    targets = []
    if t2027:
        targets.append(f"{t2027} (2027)")
    if t2028:
        targets.append(f"{t2028} (2028)")
    if targets:
        parts.append(f"целевые значения: {', '.join(targets)}")
    return "; ".join(parts) + "."



def generate_statements(df):
    ministry_name = None
    ministry_code = None
    program_name = None
    program_code = None
    measure_name = None
    last_parent_context = None
    last_parent_context_ru = None

    results = []

    for idx in range(6, len(df)):
        r = df.iloc[idx]
        annotated_type = safe_str(r[COL_ROW_TYPE])
        if annotated_type in ("EMPTY", "OTHER", "Row Type"):
            continue

        # Derive row type from code columns; fall back to annotation column
        row_type = detect_row_type(r)
        # If annotation exists and disagrees, log it but trust the code-column rule
        if annotated_type and annotated_type not in ("EMPTY", "OTHER", "Row Type") and annotated_type != row_type:
            row_type_source = "detected"
        else:
            row_type_source = "annotated"

        statements_en = []
        statements_ru = []

        if row_type == "MINISTRY":
            ministry_code = safe_str(r[COL_GRBS])
            ministry_name = safe_str(r[COL_PROG]) or safe_str(r[COL_NAME])
            program_name = None
            program_code = None
            measure_name = None
            statements_en.append(
                f"Ministry '{ministry_name}' (code {ministry_code}) is allocated budget programs in the 2026 republican budget."
            )
            statements_ru.append(
                f"Министерство/ведомство «{ministry_name}» (код {ministry_code}) включено в программный бюджет на 2026 год."
            )
            last_parent_context = f"ministry '{ministry_name}'"
            last_parent_context_ru = f"ведомства «{ministry_name}»"

        elif row_type == "PROGRAM":
            program_code = safe_str(r[COL_PROG])
            program_name = safe_str(r[COL_NAME])
            measure_name = None
            funding = build_funding_phrase(r)
            funding_ru = build_funding_phrase_ru(r)
            if funding:
                statements_en.append(
                    f"Under ministry '{ministry_name}' (code {ministry_code}), the program '{program_name}' (program code {program_code}) is allocated {funding}."
                )
            if funding_ru:
                statements_ru.append(
                    f"По ведомству «{ministry_name}» (код {ministry_code}) программа «{program_name}» (код программы {program_code}) получает {funding_ru}."
                )
            indicator = build_indicator_phrase(r)
            indicator_ru = build_indicator_phrase_ru(r)
            if indicator:
                statements_en.append(indicator)
            if indicator_ru:
                statements_ru.append(indicator_ru)
            last_parent_context = f"program '{program_name}' under ministry '{ministry_name}'"
            last_parent_context_ru = f"программы «{program_name}» ведомства «{ministry_name}»"

        elif row_type == "MEASURE":
            measure_code = safe_str(r[COL_MEASURE])
            measure_name = safe_str(r[COL_NAME])
            funding = build_funding_phrase(r)
            funding_ru = build_funding_phrase_ru(r)
            if funding:
                statements_en.append(
                    f"Under ministry '{ministry_name}', program '{program_name}', the budget measure '{measure_name}' (measure code {measure_code}) is allocated {funding}."
                )
            if funding_ru:
                statements_ru.append(
                    f"По ведомству «{ministry_name}», в программе «{program_name}», бюджетная мера «{measure_name}» (код меры {measure_code}) получает {funding_ru}."
                )
            indicator = build_indicator_phrase(r)
            indicator_ru = build_indicator_phrase_ru(r)
            if indicator:
                statements_en.append(indicator)
            if indicator_ru:
                statements_ru.append(indicator_ru)
            last_parent_context = f"measure '{measure_name}' under program '{program_name}'"
            last_parent_context_ru = f"меры «{measure_name}» программы «{program_name}»"

        elif row_type == "INDICATOR":
            indicator = build_indicator_phrase(r)
            indicator_ru = build_indicator_phrase_ru(r)
            if indicator:
                context = last_parent_context or "the previous budget item"
                statements_en.append(f"For {context}: {indicator}")
            if indicator_ru:
                context_ru = last_parent_context_ru or "предыдущей бюджетной статьи"
                statements_ru.append(f"Для {context_ru}: {indicator_ru}")

        elif row_type == "TOTAL":
            funding = build_funding_phrase(r)
            funding_ru = build_funding_phrase_ru(r)
            if funding:
                statements_en.append(f"Total for ministry '{ministry_name}' (code {ministry_code}): {funding}.")
            if funding_ru:
                statements_ru.append(f"Итого по ведомству «{ministry_name}» (код {ministry_code}): {funding_ru}.")

        if statements_en or statements_ru:
            results.append({
                "row_index": idx,
                "row_type": row_type,
                "ministry_code": ministry_code,
                "ministry_name": ministry_name,
                "program_code": program_code if row_type in ("PROGRAM", "MEASURE", "INDICATOR") else None,
                "program_name": program_name if row_type in ("PROGRAM", "MEASURE", "INDICATOR") else None,
                "statements": statements_en,
                "statements_en": statements_en,
                "statements_ru": statements_ru,
            })

    return results



def main():
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None)
    results = generate_statements(df)

    type_counts = {}
    for r in results:
        type_counts[r["row_type"]] = type_counts.get(r["row_type"], 0) + 1

    print(f"Generated statements for {len(results)} rows:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print()

    for r in results[:10]:
        print(f"[Row {r['row_index']}] ({r['row_type']})")
        for s in r["statements_en"]:
            print(f"  → {s}")
        for s in r["statements_ru"]:
            print(f"  → {s}")
        print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Full output saved to {OUTPUT_FILE}")

    for language, output_file in [("en", OUTPUT_FILE_EN), ("ru", OUTPUT_FILE_RU)]:
        language_results = language_specific_results(results, language)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(language_results, f, ensure_ascii=False, indent=2)
        statement_count = sum(len(r["statements"]) for r in language_results)
        print(f"{language.upper()} output saved to {output_file} ({statement_count} statements)")


if __name__ == "__main__":
    main()
