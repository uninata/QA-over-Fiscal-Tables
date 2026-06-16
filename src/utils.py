"""Shared pipeline utilities: env loading, path resolution, result file discovery."""

import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load_env():
    if not ENV_FILE.exists():
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_project_path(value, default=None):
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def latest_result_file(results_dir, pattern):
    files = [p for p in results_dir.glob(pattern) if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def get_results_dir():
    subdir = os.environ.get("RESULTS_SUBDIR", "").strip()
    return ROOT / "results" / subdir if subdir else ROOT / "results"


def get_run_id(fallback_env="EXPERIMENT_RUN_ID"):
    return os.environ.get(fallback_env) or datetime.now().strftime("%Y%m%d_%H%M%S")
