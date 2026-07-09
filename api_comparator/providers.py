"""各LLM APIプロバイダの呼び出し実装。"""

from __future__ import annotations

import json
import importlib
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .config import ApiKeyResolver


class JsonHttpClient:
    """JSON POSTだけを担当する薄いHTTPクライアント。"""

    def __init__(self, ssl_context: ssl.SSLContext | None = None) -> None:
        """HTTPクライアントを初期化する。

        Args:
            ssl_context: HTTPS接続時に使うSSL context。未指定なら安全な既定値。
        """

        self.ssl_context = ssl_context or self._default_ssl_context()

    def post(self, *, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        """JSONをPOSTし、レスポンスJSONを返す。

        Args:
            url: 送信先URL。
            headers: HTTPヘッダー。
            payload: JSONペイロード。

        Returns:
            レスポンスJSON。

        Raises:
            RuntimeError: HTTPエラー、接続エラー、JSON解析エラーが発生した場合。
        """

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url=url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180, context=self.ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(self.describe_url_error(exc.reason)) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("APIのレスポンスJSONを解析できません。") from exc

    def describe_url_error(self, reason: Any) -> str:
        """URL接続エラーをユーザーが対処しやすいメッセージへ変換する。

        Args:
            reason: urllibが返したエラー理由。

        Returns:
            表示用エラーメッセージ。
        """

        if isinstance(reason, ssl.SSLCertVerificationError):
            return (
                "APIへ接続できません: SSL証明書の検証に失敗しました。"
                "PythonがCA証明書を見つけられていない可能性があります。"
                " macOSのpython.org版なら「Install Certificates.command」を実行するか、"
                "`python3 -m pip install certifi` 後に `SSL_CERT_FILE=$(python3 -m certifi)` を設定して再起動してください。"
                " 証明書検証の無効化は推奨しません。"
            )
        return f"APIへ接続できません: {reason}"

    def _default_ssl_context(self) -> ssl.SSLContext:
        """証明書検証を有効にしたSSL contextを作成する。

        Returns:
            SSL context。
        """

        certifi_path = self._certifi_path()
        if certifi_path:
            return ssl.create_default_context(cafile=certifi_path)
        return ssl.create_default_context()

    def _certifi_path(self) -> str | None:
        """certifiが利用可能ならCA bundleパスを返す。

        Returns:
            certifiのCA bundleパス。未導入ならNone。
        """

        try:
            certifi = importlib.import_module("certifi")
        except ModuleNotFoundError:
            return None
        where = getattr(certifi, "where", None)
        if not callable(where):
            return None
        return str(where())


class ProviderClient(ABC):
    """プロバイダ呼び出しの共通インターフェース。"""

    @abstractmethod
    def complete(self, *, target: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """モデルAPIを呼び出して比較結果を返す。

        Args:
            target: 比較ターゲット。
            request: 共通リクエスト。

        Returns:
            比較表示用の結果辞書。
        """


class OpenAIChatClient(ProviderClient):
    """OpenAI Chat Completions互換APIのクライアント。"""

    def __init__(self, http_client: JsonHttpClient, api_key_resolver: ApiKeyResolver) -> None:
        """クライアントを初期化する。

        Args:
            http_client: JSON HTTPクライアント。
            api_key_resolver: APIキー解決クラス。
        """

        self.http_client = http_client
        self.api_key_resolver = api_key_resolver

    def complete(self, *, target: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """OpenAI Chat Completions互換APIを呼び出す。

        Args:
            target: 比較ターゲット。
            request: 共通リクエスト。

        Returns:
            比較表示用の結果辞書。
        """

        system_prompt = str(request.get("system_prompt") or "").strip()
        user_prompt = str(request.get("user_prompt") or "").strip()
        max_tokens = int(request.get("max_tokens") or 900)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": target["model"],
            "messages": messages,
            "stream": False,
        }
        if target.get("provider") == "openai":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if bool(request.get("json_mode")):
            payload["response_format"] = {"type": "json_object"}

        if target.get("provider") == "openai_compatible":
            thinking = str(target.get("thinking") or "disabled")
            if thinking in {"enabled", "disabled"}:
                payload["thinking"] = {"type": thinking}
            if thinking == "enabled":
                payload["reasoning_effort"] = str(target.get("reasoning_effort") or "high")

        extra_body = target.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        response = self.http_client.post(
            url=self._url(target),
            headers={
                "Authorization": f"Bearer {self.api_key_resolver.resolve(target)}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        return self._parse_response(response)

    def _url(self, target: dict[str, Any]) -> str:
        """ターゲットの呼び出しURLを組み立てる。

        Args:
            target: 比較ターゲット。

        Returns:
            呼び出しURL。
        """

        return str(target["base_url"]).rstrip("/") + "/" + str(target["endpoint"]).lstrip("/")

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """OpenAI互換レスポンスを比較結果へ変換する。

        Args:
            response: APIレスポンスJSON。

        Returns:
            抽出済みフィールド。
        """

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("APIレスポンスにchoicesが含まれていません。")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
        }


class AnthropicMessagesClient(ProviderClient):
    """Anthropic Messages APIのクライアント。"""

    def __init__(self, http_client: JsonHttpClient, api_key_resolver: ApiKeyResolver) -> None:
        """クライアントを初期化する。

        Args:
            http_client: JSON HTTPクライアント。
            api_key_resolver: APIキー解決クラス。
        """

        self.http_client = http_client
        self.api_key_resolver = api_key_resolver

    def complete(self, *, target: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """Anthropic Messages APIを呼び出す。

        Args:
            target: 比較ターゲット。
            request: 共通リクエスト。

        Returns:
            比較表示用の結果辞書。
        """

        system_prompt = str(request.get("system_prompt") or "").strip()
        user_prompt = str(request.get("user_prompt") or "").strip()
        max_tokens = int(request.get("max_tokens") or 900)
        payload: dict[str, Any] = {
            "model": target["model"],
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        extra_body = target.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        response = self.http_client.post(
            url=self._url(target),
            headers={
                "x-api-key": self.api_key_resolver.resolve(target),
                "anthropic-version": str(target.get("anthropic_version") or "2023-06-01"),
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        return self._parse_response(response)

    def _url(self, target: dict[str, Any]) -> str:
        """ターゲットの呼び出しURLを組み立てる。

        Args:
            target: 比較ターゲット。

        Returns:
            呼び出しURL。
        """

        return str(target["base_url"]).rstrip("/") + "/" + str(target["endpoint"]).lstrip("/")

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Anthropicレスポンスを比較結果へ変換する。

        Args:
            response: APIレスポンスJSON。

        Returns:
            抽出済みフィールド。
        """

        text_parts = []
        for block in response.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))

        usage = response.get("usage") or {}
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": response.get("stop_reason"),
            "content": "".join(text_parts),
            "reasoning_content": "",
        }


class LiteLLMClient(ProviderClient):
    """LiteLLM Python SDKを任意依存として使うクライアント。"""

    def __init__(
        self,
        api_key_resolver: ApiKeyResolver,
        litellm_module: Any | None = None,
    ) -> None:
        """クライアントを初期化する。

        Args:
            api_key_resolver: APIキー解決クラス。
            litellm_module: テスト用に注入するLiteLLM互換モジュール。
        """

        self.api_key_resolver = api_key_resolver
        self.litellm_module = litellm_module

    def complete(self, *, target: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """LiteLLM経由でモデルAPIを呼び出す。

        Args:
            target: 比較ターゲット。
            request: 共通リクエスト。

        Returns:
            比較表示用の結果辞書。

        Raises:
            RuntimeError: LiteLLMが未インストールの場合。
        """

        litellm = self._load_litellm()
        system_prompt = str(request.get("system_prompt") or "").strip()
        user_prompt = str(request.get("user_prompt") or "").strip()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": target["model"],
            "messages": messages,
            "max_tokens": int(request.get("max_tokens") or 900),
            "api_key": self.api_key_resolver.resolve(target),
            "drop_params": True,
        }
        base_url = str(target.get("base_url") or "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        if bool(request.get("json_mode")):
            kwargs["response_format"] = {"type": "json_object"}

        extra_body = target.get("extra_body")
        if isinstance(extra_body, dict):
            kwargs.update(extra_body)

        response = litellm.completion(**kwargs)
        return self._parse_response(response)

    def _load_litellm(self) -> Any:
        """LiteLLMモジュールを遅延ロードする。

        Returns:
            LiteLLMモジュール。

        Raises:
            RuntimeError: LiteLLMが未インストールの場合。
        """

        if self.litellm_module is not None:
            return self.litellm_module
        try:
            self.litellm_module = importlib.import_module("litellm")
            return self.litellm_module
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM providerを使うにはlitellmパッケージが必要です。"
                "供給網リスクを確認し、信頼できるバージョンを固定してから導入してください。"
            ) from exc

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """LiteLLMレスポンスを比較結果へ変換する。

        Args:
            response: LiteLLM ModelResponseまたはdict。

        Returns:
            抽出済みフィールド。
        """

        choices = self._get(response, "choices") or []
        if not choices:
            raise RuntimeError("LiteLLMレスポンスにchoicesが含まれていません。")
        choice = choices[0]
        message = self._get(choice, "message") or {}
        usage = self._get(response, "usage") or {}
        prompt_tokens = self._get(usage, "prompt_tokens")
        completion_tokens = self._get(usage, "completion_tokens")
        total_tokens = self._get(usage, "total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": self._get(choice, "finish_reason"),
            "content": self._get(message, "content") or "",
            "reasoning_content": self._get(message, "reasoning_content") or "",
        }

    def _get(self, value: Any, key: str) -> Any:
        """dict/オブジェクトのどちらからでも値を取得する。

        Args:
            value: 取得元。
            key: キーまたは属性名。

        Returns:
            取得値。存在しない場合はNone。
        """

        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
