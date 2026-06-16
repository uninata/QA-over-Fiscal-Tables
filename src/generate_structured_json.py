"""
Generate budget_2026_full.json from the source Excel (Приложение 14).

Reads the raw Excel, classifies each row as ministry/program/measure/indicator/total,
and outputs a structured JSON with one entry per program and measure.

Orphan programs (e.g., program 992 "State Investment Projects" which appears only
as a measure with no parent program row) are promoted to program level.

Usage:
    python src/generate_structured_json.py
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

from utils import ROOT

INPUT_FILE = ROOT / "data" / "raw" / "2026" / "русс" / "Приложение 14 (2026) ver4.xlsx"
OUTPUT_JSON = ROOT / "data" / "processed" / "budget_2026_full.json"
OUTPUT_MD = ROOT / "data" / "processed" / "budget_2026_full.md"

# Column indices (0-based, after skipping header rows)
COL_GRBS = 0       # Ministry code
COL_PROG = 1       # Program code
COL_MEASURE = 2    # Measure code
COL_INIT = 3       # Initiative code
COL_NAME = 4       # Program/measure name
COL_FUND_2026 = 5  # Funding 2026
COL_FUND_2027 = 6  # Funding 2027
COL_FUND_2028 = 7  # Funding 2028
COL_INDICATOR = 8  # Performance indicator
COL_UNIT = 9       # Unit of measurement
COL_BASELINE = 10  # Baseline value


def safe_str(val):
    """Convert value to stripped string, or None if empty/NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    return s if s else None


def safe_float(val):
    """Convert value to float, or None if empty/NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def is_numeric_code(val):
    """Check if value looks like a numeric code (e.g., '001', '01', '1.0')."""
    s = safe_str(val)
    if s is None:
        return False
    return s.replace(".", "").replace("-", "").isdigit()


def clean_code(val):
    """Clean a code value: '1.0' -> '01', '001' -> '001'."""
    s = safe_str(val)
    if s is None:
        return None
    # Handle float codes like '1.0' -> '01'
    if "." in s:
        try:
            s = f"{int(float(s)):02d}"
        except ValueError:
            pass
    return s


def classify_row(row, current_ministry_code):
    """Classify a row as MINISTRY, PROGRAM, MEASURE, INDICATOR, or TOTAL.

    The Excel layout:
      - Ministry row:  col0 = ministry_code (e.g. "34"), col1 = ministry name text
      - Program row:   col0 = ministry_code (e.g. "34"), col1 = program code (e.g. "001")
      - Measure row:   col0 = measure_num (e.g. "01"),   col1 = measure_num, col2 = float
      - Total row:     col0 = "Итог по ведомству:"
      - Indicator row: col0 empty, continuation of previous row

    Key distinction: program rows have the current ministry code in col0,
    while measure rows have a small sequential number (the measure's own index).
    """
    grbs = safe_str(row[COL_GRBS])
    has_grbs = grbs is not None
    has_prog = is_numeric_code(row[COL_PROG])
    has_measure = is_numeric_code(row[COL_MEASURE])

    if has_grbs and "Итог" in grbs:
        return "TOTAL"

    if not has_grbs:
        return "INDICATOR"

    # If col0 matches current ministry code, this is a ministry or program row
    if grbs == current_ministry_code:
        if has_prog and not has_measure:
            return "PROGRAM"
        if has_prog and has_measure:
            # Rare: ministry code in col0 + prog code + measure code
            # This can happen; treat as MEASURE
            return "MEASURE"
        if not has_prog:
            # Ministry code with text name in col1 = ministry header
            return "MINISTRY"

    # col0 is NOT the current ministry code
    # Could be: (a) a new ministry, or (b) a measure row with sequential numbering
    if has_prog and not has_measure:
        # Check if col1 contains text (ministry name) vs numeric (program code)
        prog_val = safe_str(row[COL_PROG])
        if prog_val and not is_numeric_code(prog_val):
            # col1 is text = new ministry
            return "MINISTRY"
        else:
            # col0 is small number, col1 is number = measure row
            # (measure rows have their own numbering in col0/col1)
            return "MEASURE"

    if has_prog and has_measure:
        return "MEASURE"

    if not has_prog:
        # New ministry: col0 has code, col1 has name text
        return "MINISTRY"

    return "INDICATOR"


def parse_excel(path):
    """Parse the Excel file and return structured rows."""
    df = pd.read_excel(path, header=None, skiprows=6)
    print(f"Read {len(df)} rows from {path.name}")

    rows = []
    current_ministry = None
    current_ministry_code = None
    current_program_code = None

    for i, row in df.iterrows():
        row_type = classify_row(row, current_ministry_code)

        if row_type == "MINISTRY":
            current_ministry_code = safe_str(row[COL_GRBS])
            current_ministry = safe_str(row[COL_PROG])
            if current_ministry is None or is_numeric_code(current_ministry):
                current_ministry = safe_str(row[COL_NAME])
            current_program_code = None
            continue

        if row_type in ("TOTAL", "INDICATOR"):
            continue

        if row_type == "PROGRAM":
            current_program_code = safe_str(row[COL_PROG])
            rows.append({
                "ministry": current_ministry,
                "ministry_code": current_ministry_code,
                "code_prog": current_program_code,
                "code_measure": None,
                "level": "program",
                "name": safe_str(row[COL_NAME]) or "",
                "funding_2026": safe_float(row[COL_FUND_2026]),
                "funding_2027": safe_float(row[COL_FUND_2027]),
                "funding_2028": safe_float(row[COL_FUND_2028]),
                "indicator": safe_str(row[COL_INDICATOR]),
                "indicator_unit": safe_str(row[COL_UNIT]),
                "baseline_value": safe_float(row[COL_BASELINE]),
            })

        elif row_type == "MEASURE":
            grbs_val = safe_str(row[COL_GRBS])
            if grbs_val == current_ministry_code:
                # Ministry code in col0 means col1 is the program code
                # e.g., col0=34, col1=992, col2=1.0
                prog_code = safe_str(row[COL_PROG])
                measure_code = clean_code(row[COL_MEASURE])
            else:
                # Measure's own sequential numbering in col0/col1
                prog_code = current_program_code
                measure_code = clean_code(row[COL_MEASURE])
                if measure_code is None:
                    measure_code = clean_code(row[COL_GRBS])
            rows.append({
                "ministry": current_ministry,
                "ministry_code": current_ministry_code,
                "code_prog": prog_code,
                "code_measure": measure_code,
                "level": "measure",
                "name": safe_str(row[COL_NAME]) or "",
                "funding_2026": safe_float(row[COL_FUND_2026]),
                "funding_2027": safe_float(row[COL_FUND_2027]),
                "funding_2028": safe_float(row[COL_FUND_2028]),
                "indicator": safe_str(row[COL_INDICATOR]),
                "indicator_unit": safe_str(row[COL_UNIT]),
                "baseline_value": safe_float(row[COL_BASELINE]),
            })

    return rows


def promote_orphan_programs(rows):
    """Promote orphan measures to program level.

    An orphan is a (ministry_code, code_prog) pair that has measure rows
    but no program-level row. This happens with program 992
    ("State Investment Projects") which appears only as measures.

    For each orphan, create a synthetic program-level row by summing
    the measure funding and taking the first measure's metadata.
    """
    # Find which (ministry, program) pairs have a program-level row
    program_keys = set()
    for r in rows:
        if r["level"] == "program":
            program_keys.add((r["ministry_code"], r["code_prog"]))

    # Find orphan (ministry, program) pairs
    from collections import defaultdict
    orphan_measures = defaultdict(list)
    for r in rows:
        if r["level"] == "measure":
            key = (r["ministry_code"], r["code_prog"])
            if key not in program_keys:
                orphan_measures[key].append(r)

    if not orphan_measures:
        return rows

    # Create synthetic program rows
    new_programs = []
    for (mc, cp), measures in orphan_measures.items():
        # Sum funding across all measures for this orphan program
        def sum_field(field):
            vals = [m[field] for m in measures if m[field] is not None]
            return sum(vals) if vals else None

        first = measures[0]
        new_programs.append({
            "ministry": first["ministry"],
            "ministry_code": mc,
            "code_prog": cp,
            "code_measure": None,
            "level": "program",
            "name": first["name"],
            "funding_2026": sum_field("funding_2026"),
            "funding_2027": sum_field("funding_2027"),
            "funding_2028": sum_field("funding_2028"),
            "indicator": first["indicator"],
            "indicator_unit": first["indicator_unit"],
            "baseline_value": first["baseline_value"],
        })

    # Insert each synthetic program row before its first measure
    result = []
    inserted = set()
    for r in rows:
        key = (r["ministry_code"], r["code_prog"])
        if key in orphan_measures and key not in inserted:
            # Find and insert the synthetic program row
            for np in new_programs:
                if (np["ministry_code"], np["code_prog"]) == key:
                    result.append(np)
                    inserted.add(key)
                    break
        result.append(r)

    return result


def generate_markdown(rows):
    """Generate a markdown table from structured rows."""
    header = "| ministry_code | level | ministry | program | name | funding_2026 | funding_2027 | funding_2028 | indicator | unit |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        line = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("ministry_code", ""),
            r.get("level", ""),
            (r.get("ministry") or "")[:40],
            r.get("code_prog") or "",
            (r.get("name") or "")[:50],
            f"{r['funding_2026']:,.1f}" if r.get("funding_2026") else "",
            f"{r['funding_2027']:,.1f}" if r.get("funding_2027") else "",
            f"{r['funding_2028']:,.1f}" if r.get("funding_2028") else "",
            (r.get("indicator") or "")[:40],
            r.get("indicator_unit") or "",
        )
        lines.append(line)
    return "\n".join(lines)


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        sys.exit(1)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Parse
    rows = parse_excel(INPUT_FILE)
    programs = sum(1 for r in rows if r["level"] == "program")
    measures = sum(1 for r in rows if r["level"] == "measure")
    print(f"Parsed: {programs} programs, {measures} measures, {len(rows)} total")

    # Promote orphans
    rows = promote_orphan_programs(rows)
    programs_after = sum(1 for r in rows if r["level"] == "program")
    promoted = programs_after - programs
    print(f"Promoted {promoted} orphan measures to program level")
    print(f"Final: {programs_after} programs, {sum(1 for r in rows if r['level'] == 'measure')} measures, {len(rows)} total")

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT_JSON}")

    # Write markdown
    md = generate_markdown(rows)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {OUTPUT_MD}")

    # Verify: check Ministry of Education total
    edu_total = sum(
        r["funding_2026"] or 0
        for r in rows
        if r["ministry_code"] == "34" and r["level"] == "program"
    )
    print(f"\nVerification — Ministry of Education (34) program total: {edu_total:,.1f}")


if __name__ == "__main__":
    main()
