#!/usr/bin/env python3
"""API比較GUIサーバーの起動エントリーポイント。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_comparator.server import run_server  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_server())
