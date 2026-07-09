"""API比較GUIで使うパス定義。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX_PATH = PROJECT_ROOT / "web" / "api_comparator_gui.html"
COMPARE_INDEX_PATH = PROJECT_ROOT / "web" / "api_comparator_compare.html"
TARGET_CONFIG_PATH = PROJECT_ROOT / "config" / "api_targets.json"
RESULTS_DIR = PROJECT_ROOT / "results"
EXECUTION_LOG_PATH = RESULTS_DIR / "api_comparator.log"
EXECUTION_HISTORY_PATH = RESULTS_DIR / "api_comparator_history.jsonl"
