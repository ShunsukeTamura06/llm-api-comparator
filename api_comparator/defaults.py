"""プロバイダとモデルの既定設定。"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
MODEL_PRICING_USD_PER_1M_TOKENS = {
    "deepseek-v4-flash": {"input_cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input_cache_miss": 0.435, "output": 0.87},
    "gpt-5.5": {"input_cache_miss": 5.0, "output": 30.0},
    "gpt-5.4-mini": {"input_cache_miss": 0.75, "output": 4.5},
    "claude-fable-5": {"input_cache_miss": 10.0, "output": 50.0},
    "claude-opus-4-8": {"input_cache_miss": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input_cache_miss": 2.0, "output": 10.0},
    "claude-haiku-4-5-20251001": {"input_cache_miss": 1.0, "output": 5.0},
    "gemini/gemini-3.1-pro-preview": {"input_cache_miss": 2.0, "output": 12.0},
    "gemini/gemini-3-flash-preview": {"input_cache_miss": 0.5, "output": 3.0},
    "gemini/gemini-3.1-flash-lite": {"input_cache_miss": 0.25, "output": 1.5},
    "minimax/MiniMax-M3": {"input_cache_miss": 0.3, "output": 1.2},
    "minimax/MiniMax-M2.7": {"input_cache_miss": 0.3, "output": 1.2},
}


@dataclass(frozen=True)
class ProviderDefaults:
    """プロバイダ別の既定値。

    Attributes:
        base_url: APIのBase URL。
        endpoint: APIエンドポイント。
        api_key_env: APIキーを読む環境変数名。
    """

    base_url: str
    endpoint: str
    api_key_env: str


DEFAULT_PROVIDER_SETTINGS = {
    "openai_compatible": ProviderDefaults(DEFAULT_BASE_URL, "/chat/completions", "DEEPSEEK_API_KEY"),
    "openai": ProviderDefaults("https://api.openai.com/v1", "/chat/completions", "OPENAI_API_KEY"),
    "anthropic": ProviderDefaults("https://api.anthropic.com/v1", "/messages", "ANTHROPIC_API_KEY"),
    "litellm": ProviderDefaults("", "", "LITELLM_API_KEY"),
}
