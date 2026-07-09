"""プロバイダクライアントのレジストリ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ApiKeyResolver
from .defaults import DEFAULT_PROVIDER_SETTINGS
from .providers import AnthropicMessagesClient, JsonHttpClient, LiteLLMClient, OpenAIChatClient, ProviderClient


@dataclass(frozen=True)
class ProviderDefinition:
    """プロバイダの既定値と実行クライアントをまとめた定義。

    Attributes:
        name: provider名。
        client: 実行クライアント。
        default_base_url: GUIや設定正規化で使う既定Base URL。
        default_endpoint: 既定エンドポイント。
        default_api_key_env: APIキー環境変数名。
    """

    name: str
    client: ProviderClient
    default_base_url: str
    default_endpoint: str
    default_api_key_env: str


class ProviderRegistry:
    """プロバイダ定義を名前で引けるレジストリ。"""

    def __init__(self) -> None:
        """レジストリを初期化する。"""

        self._definitions: dict[str, ProviderDefinition] = {}

    def register(self, definition: ProviderDefinition) -> None:
        """プロバイダ定義を登録する。

        Args:
            definition: プロバイダ定義。
        """

        self._definitions[definition.name] = definition

    def client_for(self, provider: str) -> ProviderClient:
        """プロバイダ名に対応するクライアントを返す。

        Args:
            provider: プロバイダ名。

        Returns:
            プロバイダクライアント。

        Raises:
            ValueError: 未対応プロバイダの場合。
        """

        definition = self._definitions.get(provider)
        if definition is None:
            raise ValueError(f"未対応のproviderです: {provider}")
        return definition.client

    def defaults_for(self, provider: str) -> dict[str, Any] | None:
        """プロバイダ名に対応する既定値を返す。

        Args:
            provider: プロバイダ名。

        Returns:
            既定値。未登録の場合はNone。
        """

        definition = self._definitions.get(provider)
        if definition is None:
            return None
        return {
            "base_url": definition.default_base_url,
            "endpoint": definition.default_endpoint,
            "api_key_env": definition.default_api_key_env,
        }


def build_default_provider_registry(api_key_resolver: ApiKeyResolver | None = None) -> ProviderRegistry:
    """標準プロバイダレジストリを構築する。

    Args:
        api_key_resolver: APIキー解決クラス。

    Returns:
        標準プロバイダが登録されたレジストリ。
    """

    resolver = api_key_resolver or ApiKeyResolver()
    http_client = JsonHttpClient()
    openai_client = OpenAIChatClient(http_client, resolver)
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            name="openai_compatible",
            client=openai_client,
            default_base_url=DEFAULT_PROVIDER_SETTINGS["openai_compatible"].base_url,
            default_endpoint=DEFAULT_PROVIDER_SETTINGS["openai_compatible"].endpoint,
            default_api_key_env=DEFAULT_PROVIDER_SETTINGS["openai_compatible"].api_key_env,
        )
    )
    registry.register(
        ProviderDefinition(
            name="openai",
            client=openai_client,
            default_base_url=DEFAULT_PROVIDER_SETTINGS["openai"].base_url,
            default_endpoint=DEFAULT_PROVIDER_SETTINGS["openai"].endpoint,
            default_api_key_env=DEFAULT_PROVIDER_SETTINGS["openai"].api_key_env,
        )
    )
    registry.register(
        ProviderDefinition(
            name="anthropic",
            client=AnthropicMessagesClient(http_client, resolver),
            default_base_url=DEFAULT_PROVIDER_SETTINGS["anthropic"].base_url,
            default_endpoint=DEFAULT_PROVIDER_SETTINGS["anthropic"].endpoint,
            default_api_key_env=DEFAULT_PROVIDER_SETTINGS["anthropic"].api_key_env,
        )
    )
    registry.register(
        ProviderDefinition(
            name="litellm",
            client=LiteLLMClient(resolver),
            default_base_url=DEFAULT_PROVIDER_SETTINGS["litellm"].base_url,
            default_endpoint=DEFAULT_PROVIDER_SETTINGS["litellm"].endpoint,
            default_api_key_env=DEFAULT_PROVIDER_SETTINGS["litellm"].api_key_env,
        )
    )
    return registry
