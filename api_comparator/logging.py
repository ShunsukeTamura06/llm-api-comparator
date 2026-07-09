"""API比較の実行ログを安全に記録する機能。"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import EXECUTION_HISTORY_PATH, EXECUTION_LOG_PATH


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


class ExecutionHistoryStore:
    """比較結果を後からGUIで再表示できる形式で保存するクラス。"""

    def __init__(self, history_path: Path = EXECUTION_HISTORY_PATH) -> None:
        """履歴ストアを初期化する。

        Args:
            history_path: 履歴JSON Linesファイルの出力先。
        """

        self.history_path = history_path
        self._lock = threading.Lock()

    def record(self, *, request: dict[str, Any], result: dict[str, Any]) -> None:
        """比較実行1件を履歴として保存する。

        Args:
            request: 比較リクエスト。
            result: 比較結果。
        """

        entry = self._build_entry(request=request, result=result)
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(line + "\n")

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """保存済み履歴の概要を新しい順に返す。

        Args:
            limit: 最大件数。

        Returns:
            履歴概要の配列。
        """

        entries = self._read_entries()
        summaries = [self._summary_for(entry) for entry in entries]
        return list(reversed(summaries))[: max(0, limit)]

    def get_run(self, request_id: str) -> dict[str, Any] | None:
        """Run IDから履歴詳細を返す。

        Args:
            request_id: 比較実行ごとのID。

        Returns:
            履歴詳細。見つからない場合はNone。
        """

        for entry in reversed(self._read_entries()):
            if entry.get("request_id") == request_id:
                return entry
        return None

    def _build_entry(self, *, request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """履歴エントリを作成する。

        Args:
            request: 比較リクエスト。
            result: 比較結果。

        Returns:
            保存可能な履歴エントリ。
        """

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": result.get("request_id"),
            "elapsed_sec": result.get("elapsed_sec"),
            "request": self._sanitize_request(request),
            "summary": result.get("summary") if isinstance(result.get("summary"), dict) else {},
            "results": result.get("results") if isinstance(result.get("results"), list) else [],
            "log_path": result.get("log_path"),
        }

    def _sanitize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """APIキーを含まないリクエスト情報へ変換する。

        Args:
            request: 比較リクエスト。

        Returns:
            保存用リクエスト。
        """

        targets = []
        for target in request.get("targets") or []:
            if not isinstance(target, dict):
                continue
            targets.append(
                {
                    "id": target.get("id"),
                    "label": target.get("label"),
                    "provider": target.get("provider"),
                    "model": target.get("model"),
                    "thinking": target.get("thinking"),
                    "reasoning_effort": target.get("reasoning_effort"),
                    "pricing": target.get("pricing") if isinstance(target.get("pricing"), dict) else {},
                }
            )
        return {
            "system_prompt": str(request.get("system_prompt") or ""),
            "user_prompt": str(request.get("user_prompt") or ""),
            "max_tokens": request.get("max_tokens"),
            "json_mode": bool(request.get("json_mode")),
            "effort": str(request.get("effort") or ""),
            "targets": targets,
        }

    def _summary_for(self, entry: dict[str, Any]) -> dict[str, Any]:
        """履歴一覧向けの概要へ変換する。

        Args:
            entry: 履歴エントリ。

        Returns:
            概要。
        """

        results = entry.get("results") if isinstance(entry.get("results"), list) else []
        request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        return {
            "timestamp": entry.get("timestamp"),
            "request_id": entry.get("request_id"),
            "elapsed_sec": entry.get("elapsed_sec"),
            "target_count": len(results),
            "success_count": len([result for result in results if isinstance(result, dict) and not result.get("error")]),
            "error_count": len([result for result in results if isinstance(result, dict) and result.get("error")]),
            "total_estimated_cost_usd": summary.get("total_estimated_cost_usd"),
            "prompt_preview": str(request.get("user_prompt") or "").replace("\n", " ")[:120],
        }

    def _read_entries(self) -> list[dict[str, Any]]:
        """履歴ファイルからエントリ配列を読み込む。

        Returns:
            履歴エントリ配列。
        """

        if not self.history_path.exists():
            return []
        entries = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries
