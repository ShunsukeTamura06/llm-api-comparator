"""比較対象APIの設定読み込みと正規化。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_SETTINGS,
    MODEL_PRICING_USD_PER_1M_TOKENS,
    ProviderDefaults,
)
from .paths import TARGET_CONFIG_PATH


DEFAULT_COMPARE_TARGETS: list[dict[str, Any]] = [
    {
        "id": "deepseek-v4-flash-thinking",
        "label": "DeepSeek V4 Flash / thinking",
        "provider": "openai_compatible",
        "base_url": DEFAULT_BASE_URL,
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": "enabled",
        "reasoning_effort": "high",
        "selected": True,
        "pricing": {"input_per_1m": 0.14, "output_per_1m": 0.28},
    },
    {
        "id": "deepseek-v4-flash-direct",
        "label": "DeepSeek V4 Flash / direct",
        "provider": "openai_compatible",
        "base_url": DEFAULT_BASE_URL,
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": "disabled",
        "selected": True,
        "pricing": {"input_per_1m": 0.14, "output_per_1m": 0.28},
    },
    {
        "id": "deepseek-v4-pro-thinking",
        "label": "DeepSeek V4 Pro / thinking",
        "provider": "openai_compatible",
        "base_url": DEFAULT_BASE_URL,
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": "enabled",
        "reasoning_effort": "high",
        "selected": True,
        "pricing": {"input_per_1m": 0.435, "output_per_1m": 0.87},
    },
    {
        "id": "deepseek-v4-pro-direct",
        "label": "DeepSeek V4 Pro / direct",
        "provider": "openai_compatible",
        "base_url": DEFAULT_BASE_URL,
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": "disabled",
        "selected": True,
        "pricing": {"input_per_1m": 0.435, "output_per_1m": 0.87},
    },
    {
        "id": "openai-gpt-5-5",
        "label": "OpenAI GPT-5.5",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.5",
        "api_key_env": "OPENAI_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 5.0, "output_per_1m": 30.0},
    },
    {
        "id": "openai-gpt-5-4-mini",
        "label": "OpenAI GPT-5.4 Mini",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4-mini",
        "api_key_env": "OPENAI_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 0.75, "output_per_1m": 4.5},
    },
    {
        "id": "anthropic-claude-fable-5",
        "label": "Anthropic Claude Fable 5",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-fable-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_version": "2023-06-01",
        "selected": False,
        "pricing": {"input_per_1m": 10.0, "output_per_1m": 50.0},
    },
    {
        "id": "anthropic-claude-opus-4-8",
        "label": "Anthropic Claude Opus 4.8",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-opus-4-8",
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_version": "2023-06-01",
        "selected": False,
        "pricing": {"input_per_1m": 5.0, "output_per_1m": 25.0},
    },
    {
        "id": "anthropic-claude-sonnet-5",
        "label": "Anthropic Claude Sonnet 5",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_version": "2023-06-01",
        "selected": False,
        "pricing": {"input_per_1m": 2.0, "output_per_1m": 10.0},
        "pricing_note": "Introductory pricing through 2026-08-31; standard pricing is $3/$15 per 1M tokens after that date.",
    },
    {
        "id": "anthropic-claude-haiku-4-5",
        "label": "Anthropic Claude Haiku 4.5",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_version": "2023-06-01",
        "selected": False,
        "pricing": {"input_per_1m": 1.0, "output_per_1m": 5.0},
    },
    {
        "id": "gemini-3-1-pro-via-litellm",
        "label": "Gemini 3.1 Pro Preview via LiteLLM",
        "provider": "litellm",
        "model": "gemini/gemini-3.1-pro-preview",
        "api_key_env": "GEMINI_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 2.0, "output_per_1m": 12.0},
        "pricing_note": "Standard text pricing for prompts <= 200k tokens.",
    },
    {
        "id": "gemini-3-flash-via-litellm",
        "label": "Gemini 3 Flash Preview via LiteLLM",
        "provider": "litellm",
        "model": "gemini/gemini-3-flash-preview",
        "api_key_env": "GEMINI_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 0.5, "output_per_1m": 3.0},
    },
    {
        "id": "gemini-3-1-flash-lite-via-litellm",
        "label": "Gemini 3.1 Flash-Lite via LiteLLM",
        "provider": "litellm",
        "model": "gemini/gemini-3.1-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 0.25, "output_per_1m": 1.5},
    },
    {
        "id": "minimax-m3-via-litellm",
        "label": "MiniMax M3 via LiteLLM",
        "provider": "litellm",
        "model": "minimax/MiniMax-M3",
        "api_key_env": "MINIMAX_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 0.3, "output_per_1m": 1.2},
        "pricing_note": "Standard pricing for <= 512k input tokens during the permanent 50% off period.",
    },
    {
        "id": "minimax-m2-7-via-litellm",
        "label": "MiniMax M2.7 via LiteLLM",
        "provider": "litellm",
        "model": "minimax/MiniMax-M2.7",
        "api_key_env": "MINIMAX_API_KEY",
        "selected": False,
        "pricing": {"input_per_1m": 0.3, "output_per_1m": 1.2},
    },
]


class TargetNormalizer:
    """比較ターゲット定義を正規化する責務を持つクラス。"""

    def normalize(self, target: dict[str, Any]) -> dict[str, Any]:
        """ターゲット定義に既定値を補う。

        Args:
            target: 入力ターゲット。

        Returns:
            正規化済みターゲット。
        """

        provider = str(target.get("provider") or "openai_compatible")
        model = str(target.get("model") or "")
        defaults = self.provider_defaults(provider)
        target_id = str(target.get("id") or f"{provider}-{model}-{time.time_ns()}")
        pricing = target.get("pricing") if isinstance(target.get("pricing"), dict) else {}
        return {
            "id": target_id,
            "label": str(target.get("label") or target_id),
            "provider": provider,
            "base_url": str(target.get("base_url") or defaults.base_url),
            "endpoint": str(target.get("endpoint") or defaults.endpoint),
            "model": model,
            "api_key_env": str(target.get("api_key_env") or defaults.api_key_env),
            "api_key": str(target.get("api_key") or ""),
            "thinking": str(target.get("thinking") or "disabled"),
            "reasoning_effort": str(target.get("reasoning_effort") or "high"),
            "anthropic_version": str(target.get("anthropic_version") or "2023-06-01"),
            "selected": bool(target.get("selected", True)),
            "pricing": {
                "input_per_1m": pricing.get("input_per_1m"),
                "output_per_1m": pricing.get("output_per_1m"),
            },
            "pricing_note": str(target.get("pricing_note") or ""),
            "extra_body": target.get("extra_body") if isinstance(target.get("extra_body"), dict) else {},
        }

    def provider_defaults(self, provider: str) -> ProviderDefaults:
        """プロバイダ別の既定値を返す。

        Args:
            provider: プロバイダ種別。

        Returns:
            既定値。
        """

        return DEFAULT_PROVIDER_SETTINGS.get(provider) or ProviderDefaults(
            DEFAULT_BASE_URL,
            "/chat/completions",
            "DEEPSEEK_API_KEY",
        )


class ApiKeyResolver:
    """ターゲットごとのAPIキー解決を担当するクラス。"""

    def resolve(self, target: dict[str, Any], *, allow_missing: bool = False) -> str:
        """ターゲットのAPIキーを解決する。

        Args:
            target: 比較ターゲット。
            allow_missing: 未設定時に例外でなく空文字を返すか。

        Returns:
            APIキー。

        Raises:
            ValueError: APIキーを解決できない場合。
        """

        api_key = str(target.get("api_key") or "").strip()
        if api_key:
            return api_key

        env_name = str(target.get("api_key_env") or "").strip()
        if env_name:
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                return env_value

        if allow_missing:
            return ""
        label = target.get("label") or target.get("id") or target.get("model")
        raise ValueError(f"{label} のAPIキーが未設定です。api_key_envまたは外部設定ファイルを確認してください。")


class TargetRepository:
    """比較ターゲット設定の読み込みを担当するリポジトリ。"""

    def __init__(
        self,
        config_path: Path = TARGET_CONFIG_PATH,
        normalizer: TargetNormalizer | None = None,
        api_key_resolver: ApiKeyResolver | None = None,
    ) -> None:
        """リポジトリを初期化する。

        Args:
            config_path: ターゲット設定ファイルのパス。
            normalizer: ターゲット正規化クラス。
            api_key_resolver: APIキー解決クラス。
        """

        self.config_path = config_path
        self.normalizer = normalizer or TargetNormalizer()
        self.api_key_resolver = api_key_resolver or ApiKeyResolver()

    def load(self) -> list[dict[str, Any]]:
        """比較ターゲット設定を読み込む。

        Returns:
            比較ターゲット配列。

        Raises:
            RuntimeError: 設定ファイルのJSON形式が不正な場合。
        """

        if not self.config_path.exists():
            return [self.normalizer.normalize(target) for target in DEFAULT_COMPARE_TARGETS]

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"設定ファイルのJSONが不正です: {self.config_path}") from exc

        targets = config.get("targets")
        if not isinstance(targets, list):
            raise RuntimeError("設定ファイルにはtargets配列が必要です。")
        return [self.normalizer.normalize(target) for target in targets if isinstance(target, dict)]

    def public_targets(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """ブラウザへ返してよいターゲット情報へ変換する。

        Args:
            targets: サーバー側ターゲット配列。

        Returns:
            APIキー本体を除いたターゲット配列。
        """

        public_values = []
        for target in targets:
            public_target = dict(target)
            public_target.pop("api_key", None)
            public_target["has_inline_api_key"] = bool(target.get("api_key"))
            public_target["has_resolved_api_key"] = bool(
                self.api_key_resolver.resolve(target, allow_missing=True)
            )
            public_values.append(public_target)
        return public_values
