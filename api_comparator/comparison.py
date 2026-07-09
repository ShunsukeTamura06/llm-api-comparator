"""比較結果の集約戦略。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SummaryBuilder(ABC):
    """比較サマリー生成の共通インターフェース。"""

    @abstractmethod
    def build(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """比較サマリーを生成する。

        Args:
            results: 成功した比較結果。

        Returns:
            サマリー。
        """


class DefaultSummaryBuilder(SummaryBuilder):
    """速度、費用、出力量を中心にした標準サマリー生成。"""

    def build(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """比較サマリーを生成する。

        Args:
            results: 成功した比較結果。

        Returns:
            サマリー。
        """

        if not results:
            return {}

        fastest = min(results, key=lambda item: item["latency_sec"])
        cheapest = min(
            [result for result in results if result.get("estimated_cost_usd") is not None],
            key=lambda item: item["estimated_cost_usd"],
            default=None,
        )
        longest = max(results, key=lambda item: item.get("completion_tokens") or 0)
        total_cost = sum(result.get("estimated_cost_usd") or 0 for result in results)
        return {
            "fastest_target_id": fastest.get("target_id"),
            "fastest_label": fastest.get("label"),
            "cheapest_target_id": cheapest.get("target_id") if cheapest else None,
            "cheapest_label": cheapest.get("label") if cheapest else None,
            "longest_target_id": longest.get("target_id"),
            "longest_label": longest.get("label"),
            "total_estimated_cost_usd": round(total_cost, 8),
        }


class ComparisonRequestValidator:
    """比較リクエストの検証を担当するクラス。"""

    def __init__(self, *, max_targets: int = 8, max_tokens_limit: int = 200000) -> None:
        """バリデータを初期化する。

        Args:
            max_targets: 一度に比較できる最大ターゲット数。
            max_tokens_limit: max_tokensの上限。
        """

        self.max_targets = max_targets
        self.max_tokens_limit = max_tokens_limit

    def validate(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """比較リクエストを検証する。

        Args:
            request: 比較リクエスト。

        Returns:
            検証済みターゲット配列。

        Raises:
            ValueError: 入力値が不正な場合。
        """

        requested_targets = request.get("targets")
        if not isinstance(requested_targets, list) or not requested_targets:
            raise ValueError("比較対象を1つ以上選択してください。")
        if len(requested_targets) > self.max_targets:
            raise ValueError(f"比較対象は最大{self.max_targets}件までです。")

        user_prompt = str(request.get("user_prompt") or "").strip()
        if not user_prompt:
            raise ValueError("Userプロンプトを入力してください。")

        max_tokens = int(request.get("max_tokens") or 900)
        if max_tokens < 1 or max_tokens > self.max_tokens_limit:
            raise ValueError(f"max_tokensは1から{self.max_tokens_limit}の範囲で指定してください。")

        invalid_targets = [target for target in requested_targets if not isinstance(target, dict)]
        if invalid_targets:
            raise ValueError("比較対象の形式が不正です。")
        return requested_targets
