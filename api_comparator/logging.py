"""API比較の実行ログを安全に記録する機能。"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import EXECUTION_LOG_PATH


class ExecutionLogger:
    """APIキーを含めず、比較実行の事実だけをJSON Linesへ記録するクラス。"""

    def __init__(self, log_path: Path = EXECUTION_LOG_PATH) -> None:
        """ロガーを初期化する。

        Args:
            log_path: ログファイルの出力先。
        """

        self.log_path = log_path
        self._lock = threading.Lock()

    def log_run_started(self, *, request_id: str, target_count: int, max_tokens: int) -> None:
        """比較実行の開始を記録する。

        Args:
            request_id: 比較実行ごとのID。
            target_count: 実行対象数。
            max_tokens: 最大出力トークン数。
        """

        self._write(
            {
                "event": "run_started",
                "request_id": request_id,
                "target_count": target_count,
                "max_tokens": max_tokens,
            }
        )

    def log_run_finished(
        self,
        *,
        request_id: str,
        elapsed_sec: float,
        success_count: int,
        error_count: int,
    ) -> None:
        """比較実行の終了を記録する。

        Args:
            request_id: 比較実行ごとのID。
            elapsed_sec: 全体の実行時間。
            success_count: 成功したターゲット数。
            error_count: エラーになったターゲット数。
        """

        self._write(
            {
                "event": "run_finished",
                "request_id": request_id,
                "elapsed_sec": elapsed_sec,
                "success_count": success_count,
                "error_count": error_count,
            }
        )

    def log_result(self, *, request_id: str, target: dict[str, Any], result: dict[str, Any]) -> None:
        """比較ターゲット1件の結果を記録する。

        Args:
            request_id: 比較実行ごとのID。
            target: 比較ターゲット。APIキーは記録しない。
            result: 比較結果。
        """

        self._write(
            {
                "event": "target_result",
                "request_id": request_id,
                "target_id": target.get("id"),
                "label": target.get("label"),
                "provider": target.get("provider"),
                "model": target.get("model"),
                "thinking": target.get("thinking"),
                "latency_sec": result.get("latency_sec"),
                "total_tokens": result.get("total_tokens"),
                "estimated_cost_usd": result.get("estimated_cost_usd"),
                "finish_reason": result.get("finish_reason"),
                "warning": result.get("warning"),
                "error": result.get("error"),
            }
        )

    def _write(self, entry: dict[str, Any]) -> None:
        """ログ1行を書き込む。

        Args:
            entry: 記録する値。
        """

        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
