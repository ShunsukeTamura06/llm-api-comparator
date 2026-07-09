"""API比較のアプリケーションサービス。"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .comparison import ComparisonRequestValidator, DefaultSummaryBuilder, SummaryBuilder
from .config import TargetNormalizer, TargetRepository
from .logging import ExecutionLogger
from .provider_registry import ProviderRegistry, build_default_provider_registry


class TokenCostEstimator:
    """ターゲット設定の単価から概算費用を計算するクラス。"""

    def estimate(
        self,
        target: dict[str, Any],
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        """概算費用を計算する。

        Args:
            target: 比較ターゲット。
            prompt_tokens: 入力トークン数。
            completion_tokens: 出力トークン数。

        Returns:
            概算費用。必要な値がない場合はNone。
        """

        pricing = target.get("pricing") if isinstance(target.get("pricing"), dict) else {}
        input_price = pricing.get("input_per_1m")
        output_price = pricing.get("output_per_1m")
        if input_price is None or output_price is None or prompt_tokens is None or completion_tokens is None:
            return None
        return round((prompt_tokens * float(input_price) + completion_tokens * float(output_price)) / 1_000_000, 8)


class TargetMerger:
    """GUIから来たターゲット指定と設定ファイルのターゲットを統合するクラス。"""

    def __init__(self, repository: TargetRepository, normalizer: TargetNormalizer | None = None) -> None:
        """マージャを初期化する。

        Args:
            repository: ターゲットリポジトリ。
            normalizer: ターゲット正規化クラス。
        """

        self.repository = repository
        self.normalizer = normalizer or TargetNormalizer()

    def merge(self, requested_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """リクエストターゲットへ設定ファイル側の秘密情報を補う。

        Args:
            requested_targets: GUIから送られたターゲット配列。

        Returns:
            正規化済みターゲット配列。
        """

        configured_targets = {str(target.get("id")): target for target in self.repository.load()}
        merged_targets = []
        for target in requested_targets:
            configured_target = configured_targets.get(str(target.get("id")), {})
            merged_target = {**configured_target, **target}
            if configured_target.get("api_key") and not target.get("api_key"):
                merged_target["api_key"] = configured_target["api_key"]
            merged_targets.append(self.normalizer.normalize(merged_target))
        return merged_targets


class ResultAdvisor:
    """比較結果に対するユーザー向け補足警告を生成するクラス。"""

    def warning_for(self, result: dict[str, Any], *, max_tokens: int) -> str:
        """結果に応じた補足警告を返す。

        Args:
            result: 比較結果。
            max_tokens: リクエストの最大出力トークン数。

        Returns:
            警告文。不要な場合は空文字。
        """

        finish_reason = str(result.get("finish_reason") or "")
        content = str(result.get("content") or "")
        if finish_reason != "length" or content:
            return ""

        thinking = str(result.get("thinking") or "")
        model = str(result.get("model") or "")
        if thinking == "enabled":
            return (
                f"{model} は最大出力 {max_tokens} tokens に達し、thinkingで上限を使い切った可能性があります。"
                " 最大出力を2048〜4096以上に増やすか、thinking disabledのターゲットも比較してください。"
            )
        return f"{model} は最大出力 {max_tokens} tokens に達しました。最大出力を増やして再実行してください。"


class ComparisonService:
    """複数APIの並行実行と結果集約を担当するサービス。"""

    def __init__(
        self,
        repository: TargetRepository | None = None,
        provider_registry: ProviderRegistry | None = None,
        cost_estimator: TokenCostEstimator | None = None,
        execution_logger: ExecutionLogger | None = None,
        result_advisor: ResultAdvisor | None = None,
        summary_builder: SummaryBuilder | None = None,
        request_validator: ComparisonRequestValidator | None = None,
    ) -> None:
        """サービスを初期化する。

        Args:
            repository: ターゲットリポジトリ。
            provider_registry: プロバイダクライアントレジストリ。
            cost_estimator: 概算費用計算クラス。
            execution_logger: 実行ログ記録クラス。
            result_advisor: 結果補足警告の生成クラス。
            summary_builder: 比較サマリー生成戦略。
            request_validator: 比較リクエストの検証クラス。
        """

        self.repository = repository or TargetRepository()
        self.target_merger = TargetMerger(self.repository)
        self.provider_registry = provider_registry or build_default_provider_registry()
        self.cost_estimator = cost_estimator or TokenCostEstimator()
        self.execution_logger = execution_logger or ExecutionLogger()
        self.result_advisor = result_advisor or ResultAdvisor()
        self.summary_builder = summary_builder or DefaultSummaryBuilder()
        self.request_validator = request_validator or ComparisonRequestValidator()

    def compare(self, request: dict[str, Any]) -> dict[str, Any]:
        """複数ターゲットを並行実行して比較結果を返す。

        Args:
            request: 比較リクエスト。

        Returns:
            比較結果。

        Raises:
            ValueError: 入力値が不正な場合。
        """

        requested_targets = self.request_validator.validate(request)
        targets = self.target_merger.merge(requested_targets)
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        self.execution_logger.log_run_started(
            request_id=request_id,
            target_count=len(targets),
            max_tokens=int(request.get("max_tokens") or 900),
        )
        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as executor:
            futures = [executor.submit(self._call_target, request, target, request_id) for target in targets]
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda item: str(item.get("target_id") or ""))
        successful_results = [result for result in results if not result.get("error")]
        elapsed_sec = round(time.perf_counter() - started, 3)
        self.execution_logger.log_run_finished(
            request_id=request_id,
            elapsed_sec=elapsed_sec,
            success_count=len(successful_results),
            error_count=len(results) - len(successful_results),
        )
        return {
            "request_id": request_id,
            "elapsed_sec": elapsed_sec,
            "results": results,
            "summary": self.summary_builder.build(successful_results),
            "log_path": str(self.execution_logger.log_path),
        }

    def _call_target(self, request: dict[str, Any], target: dict[str, Any], request_id: str) -> dict[str, Any]:
        """比較対象1件を実行する。

        Args:
            request: 共通リクエスト。
            target: 比較ターゲット。
            request_id: 比較実行ごとのID。

        Returns:
            成功またはエラーを表す結果辞書。
        """

        try:
            provider = str(target.get("provider") or "openai_compatible")
            client = self.provider_registry.client_for(provider)
            started = time.perf_counter()
            parsed = client.complete(target=target, request=request)
            latency = time.perf_counter() - started
            prompt_tokens = parsed.get("prompt_tokens")
            completion_tokens = parsed.get("completion_tokens")
            result = {
                "target_id": target.get("id"),
                "label": target.get("label"),
                "provider": provider,
                "model": target.get("model"),
                "thinking": target.get("thinking"),
                "latency_sec": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": parsed.get("total_tokens"),
                "estimated_cost_usd": self.cost_estimator.estimate(target, prompt_tokens, completion_tokens),
                "finish_reason": parsed.get("finish_reason"),
                "content": parsed.get("content") or "",
                "reasoning_content": parsed.get("reasoning_content") or "",
                "error": None,
            }
            result["warning"] = self.result_advisor.warning_for(
                result,
                max_tokens=int(request.get("max_tokens") or 900),
            )
            self.execution_logger.log_result(request_id=request_id, target=target, result=result)
            return result
        except Exception as exc:  # noqa: BLE001
            result = {
                "target_id": target.get("id"),
                "label": target.get("label"),
                "provider": target.get("provider"),
                "model": target.get("model"),
                "thinking": target.get("thinking"),
                "latency_sec": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost_usd": None,
                "finish_reason": None,
                "content": "",
                "reasoning_content": "",
                "warning": "",
                "error": str(exc),
            }
            self.execution_logger.log_result(request_id=request_id, target=target, result=result)
            return result
