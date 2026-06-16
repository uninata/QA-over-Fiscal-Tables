"""Shared constants and utilities for budget statement generation."""

import math

# Column indices for the Annotated Appendix Excel
COL_GRBS = 0
COL_PROG = 1
COL_MEASURE = 2
COL_INIT = 3
COL_NAME = 4
COL_FUND_2026 = 5
COL_FUND_2027 = 6
COL_FUND_2028 = 7
COL_INDICATOR = 8
COL_UNIT = 9
COL_BASELINE = 10
COL_TARGET_2027 = 11
COL_TARGET_2028 = 12
COL_ROW_TYPE = 26


def safe_str(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val).strip()


def fmt_amount(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return f"{val:,.1f}" if isinstance(val, float) else str(val)


def is_numeric_code(val):
    s = safe_str(val)
    if s is None:
        return False
    return s.replace(".", "").replace("-", "").isdigit()


def detect_row_type(r):
    grbs = safe_str(r[COL_GRBS])
    has_grbs = grbs is not None
    has_prog_code = is_numeric_code(r[COL_PROG])
    has_measure_code = is_numeric_code(r[COL_MEASURE])
    if has_grbs and "Итог" in grbs:
        return "TOTAL"
    if has_grbs and has_prog_code and has_measure_code:
        return "MEASURE"
    elif has_grbs and has_prog_code and not has_measure_code:
        return "PROGRAM"
    elif has_grbs and not has_prog_code:
        return "MINISTRY"
    else:
        return "INDICATOR"


def language_specific_results(results, language):
    statement_key = f"statements_{language}"
    output = []
    for row in results:
        output.append({
            "row_index": row["row_index"],
            "row_type": row["row_type"],
            "ministry_code": row["ministry_code"],
            "ministry_name": row["ministry_name"],
            "program_code": row["program_code"],
            "program_name": row["program_name"],
            "statements": row[statement_key],
        })
    return output
