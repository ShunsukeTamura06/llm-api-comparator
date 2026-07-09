"""API比較バックエンドの単体テスト。"""

from __future__ import annotations

import unittest
import json
import ssl
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from api_comparator.comparison import ComparisonRequestValidator, DefaultSummaryBuilder
from api_comparator.config import TargetNormalizer, TargetRepository
from api_comparator.logging import ExecutionHistoryStore, ExecutionLogger
from api_comparator.paths import COMPARE_INDEX_PATH
from api_comparator.provider_registry import ProviderRegistry, build_default_provider_registry
from api_comparator.providers import JsonHttpClient, LiteLLMClient, OpenAIChatClient
from api_comparator.services import ResultAdvisor


class DefaultSummaryBuilderTest(unittest.TestCase):
    """標準サマリー生成のテスト。"""

    def test_build_selects_fastest_and_cheapest(self) -> None:
        """最速・最安・合計費用を集計できる。"""

        summary = DefaultSummaryBuilder().build(
            [
                {
                    "target_id": "slow-cheap",
                    "label": "Slow Cheap",
                    "latency_sec": 2.0,
                    "estimated_cost_usd": 0.01,
                    "completion_tokens": 100,
                },
                {
                    "target_id": "fast-expensive",
                    "label": "Fast Expensive",
                    "latency_sec": 1.0,
                    "estimated_cost_usd": 0.03,
                    "completion_tokens": 200,
                },
            ]
        )

        self.assertEqual(summary["fastest_target_id"], "fast-expensive")
        self.assertEqual(summary["cheapest_target_id"], "slow-cheap")
        self.assertEqual(summary["longest_target_id"], "fast-expensive")
        self.assertEqual(summary["total_estimated_cost_usd"], 0.04)


class ComparisonRequestValidatorTest(unittest.TestCase):
    """比較リクエスト検証のテスト。"""

    def test_validate_rejects_empty_prompt(self) -> None:
        """Userプロンプトが空なら拒否する。"""

        validator = ComparisonRequestValidator()

        with self.assertRaises(ValueError):
            validator.validate({"targets": [{"id": "a"}], "user_prompt": ""})


class ResultAdvisorTest(unittest.TestCase):
    """比較結果への補足警告生成のテスト。"""

    def test_warns_when_length_finished_without_content(self) -> None:
        """length終了かつ本文なしなら最大出力不足を説明する。"""

        warning = ResultAdvisor().warning_for(
            {
                "finish_reason": "length",
                "content": "",
                "thinking": "enabled",
                "model": "deepseek-v4-pro",
            },
            max_tokens=700,
        )

        self.assertIn("最大出力", warning)
        self.assertIn("thinking", warning)
        self.assertIn("2048", warning)

    def test_does_not_warn_for_normal_stop(self) -> None:
        """通常終了なら警告しない。"""

        warning = ResultAdvisor().warning_for(
            {
                "finish_reason": "stop",
                "content": "OK",
                "thinking": "disabled",
                "model": "deepseek-v4-flash",
            },
            max_tokens=700,
        )

        self.assertEqual(warning, "")


class TargetNormalizerTest(unittest.TestCase):
    """ターゲット正規化のテスト。"""

    def test_normalize_allows_openai_compatible_custom_provider_target(self) -> None:
        """OpenAI互換APIは設定だけで既定値を補完できる。"""

        target = TargetNormalizer().normalize(
            {
                "id": "minimax-example",
                "label": "Minimax Example",
                "provider": "openai_compatible",
                "base_url": "https://example.test/v1",
                "model": "example-model",
                "api_key_env": "MINIMAX_API_KEY",
            }
        )

        self.assertEqual(target["endpoint"], "/chat/completions")
        self.assertEqual(target["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(target["base_url"], "https://example.test/v1")

    def test_normalize_litellm_provider_defaults(self) -> None:
        """LiteLLM providerは専用の既定値を補完できる。"""

        target = TargetNormalizer().normalize(
            {
                "id": "gemini-via-litellm",
                "label": "Gemini via LiteLLM",
                "provider": "litellm",
                "model": "gemini/gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
            }
        )

        self.assertEqual(target["provider"], "litellm")
        self.assertEqual(target["endpoint"], "")
        self.assertEqual(target["api_key_env"], "GEMINI_API_KEY")


class ProviderRegistryTest(unittest.TestCase):
    """プロバイダレジストリのテスト。"""

    def test_client_for_rejects_unknown_provider(self) -> None:
        """未登録providerなら明確なエラーを返す。"""

        registry = ProviderRegistry()

        with self.assertRaises(ValueError):
            registry.client_for("unknown")

    def test_default_registry_contains_litellm_provider(self) -> None:
        """標準レジストリにLiteLLM providerが登録されている。"""

        registry = build_default_provider_registry()

        self.assertIsNotNone(registry.client_for("litellm"))


class LiteLLMClientTest(unittest.TestCase):
    """LiteLLMクライアントのテスト。"""

    def test_complete_uses_injected_litellm_module(self) -> None:
        """LiteLLM moduleを注入してcompletion結果を共通形式へ変換できる。"""

        litellm_module = Mock()
        litellm_module.completion.return_value = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        resolver = Mock()
        resolver.resolve.return_value = "test-key"
        client = LiteLLMClient(api_key_resolver=resolver, litellm_module=litellm_module)

        result = client.complete(
            target={
                "provider": "litellm",
                "model": "gemini/gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "extra_body": {},
            },
            request={"system_prompt": "sys", "user_prompt": "hi", "max_tokens": 32, "json_mode": False},
        )

        litellm_module.completion.assert_called_once()
        call_kwargs = litellm_module.completion.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(call_kwargs["api_key"], "test-key")
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["total_tokens"], 7)


class JsonHttpClientTest(unittest.TestCase):
    """JSON HTTPクライアントのテスト。"""

    def test_ssl_error_message_includes_certificate_fix(self) -> None:
        """証明書検証エラーは復旧手順つきのメッセージに変換する。"""

        client = JsonHttpClient()
        certificate_error = ssl.SSLCertVerificationError("unable to get local issuer certificate")

        message = client.describe_url_error(certificate_error)

        self.assertIn("SSL証明書の検証に失敗しました", message)
        self.assertIn("SSL_CERT_FILE", message)
        self.assertIn("certifi", message)


class OpenAIChatClientTest(unittest.TestCase):
    """OpenAI Chat Completions互換クライアントのテスト。"""

    def test_openai_provider_uses_max_completion_tokens(self) -> None:
        """OpenAI providerでは最新モデル向けのmax_completion_tokensを送る。"""

        http_client = Mock()
        http_client.post.return_value = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        resolver = Mock()
        resolver.resolve.return_value = "test-key"
        client = OpenAIChatClient(http_client=http_client, api_key_resolver=resolver)

        client.complete(
            target={
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "/chat/completions",
                "model": "gpt-5.5",
                "extra_body": {},
            },
            request={"system_prompt": "", "user_prompt": "hi", "max_tokens": 16, "json_mode": False},
        )

        payload = http_client.post.call_args.kwargs["payload"]
        self.assertEqual(payload["max_completion_tokens"], 16)
        self.assertNotIn("max_tokens", payload)


class ExecutionLoggerTest(unittest.TestCase):
    """実行ログのテスト。"""

    def test_log_result_writes_sanitized_jsonl(self) -> None:
        """APIキーを含めず実行結果をJSON Linesで記録する。"""

        with TemporaryDirectory() as temp_dir:
            logger = ExecutionLogger(log_path=Path(temp_dir) / "api_comparator.log")

            logger.log_result(
                request_id="req-1",
                target={"id": "openai", "label": "OpenAI", "api_key": "dummy-secret", "api_key_env": "OPENAI_API_KEY"},
                result={"error": None, "latency_sec": 1.23, "total_tokens": 42, "estimated_cost_usd": 0.001},
            )

            log_entry = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(log_entry["event"], "target_result")
            self.assertEqual(log_entry["target_id"], "openai")
            self.assertNotIn("api_key", log_entry)
            self.assertNotIn("dummy-secret", json.dumps(log_entry))


class ExecutionHistoryStoreTest(unittest.TestCase):
    """実行履歴保存のテスト。"""

    def test_record_saves_replayable_history_without_api_keys(self) -> None:
        """質問と結果を再表示可能な形式で保存し、APIキーは含めない。"""

        with TemporaryDirectory() as temp_dir:
            store = ExecutionHistoryStore(history_path=Path(temp_dir) / "history.jsonl")

            store.record(
                request={
                    "system_prompt": "sys",
                    "user_prompt": "hi",
                    "max_tokens": 32,
                    "json_mode": False,
                    "targets": [
                        {
                            "id": "openai",
                            "label": "OpenAI",
                            "model": "gpt",
                            "provider": "openai",
                            "api_key": "dummy-secret",
                            "api_key_env": "OPENAI_API_KEY",
                        }
                    ],
                },
                result={
                    "request_id": "req-1",
                    "elapsed_sec": 1.2,
                    "summary": {"total_estimated_cost_usd": 0.01},
                    "results": [{"target_id": "openai", "content": "hello", "error": None}],
                },
            )

            history = store.list_runs(limit=10)
            detail = store.get_run("req-1")
            serialized_detail = json.dumps(detail, ensure_ascii=False)

            self.assertEqual(history[0]["request_id"], "req-1")
            self.assertEqual(history[0]["target_count"], 1)
            self.assertEqual(detail["request"]["user_prompt"], "hi")
            self.assertEqual(detail["results"][0]["content"], "hello")
            self.assertNotIn("api_key", serialized_detail)
            self.assertNotIn("dummy-secret", serialized_detail)


class StaticPageTest(unittest.TestCase):
    """静的ページのテスト。"""

    def test_compare_page_exists_with_visual_sections(self) -> None:
        """比較専用ページにはグラフと表の表示領域がある。"""

        html = COMPARE_INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn("実行ページへ戻る", html)
        self.assertIn("latencyChart", html)
        self.assertIn("costChart", html)
        self.assertIn("tokenChart", html)
        self.assertIn("matrixBody", html)


class ExampleConfigTest(unittest.TestCase):
    """設定例ファイルのテスト。"""

    def test_example_config_contains_litellm_targets(self) -> None:
        """設定例にGemini/Minimax向けLiteLLMターゲットが含まれている。"""

        config_path = Path(__file__).resolve().parent.parent / "config" / "api_targets.example.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        providers = {target["id"]: target["provider"] for target in config["targets"]}

        self.assertEqual(providers["gemini-3-1-pro-via-litellm"], "litellm")
        self.assertEqual(providers["minimax-m3-via-litellm"], "litellm")

    def test_example_config_contains_representative_pricing(self) -> None:
        """設定例には代表モデルの単価が事前入力されている。"""

        config_path = Path(__file__).resolve().parent.parent / "config" / "api_targets.example.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pricing_by_id = {target["id"]: target["pricing"] for target in config["targets"]}

        self.assertEqual(pricing_by_id["openai-gpt-5-5"], {"input_per_1m": 5.0, "output_per_1m": 30.0})
        self.assertEqual(pricing_by_id["anthropic-claude-fable-5"], {"input_per_1m": 10.0, "output_per_1m": 50.0})
        self.assertEqual(pricing_by_id["gemini-3-1-pro-via-litellm"], {"input_per_1m": 2.0, "output_per_1m": 12.0})
        self.assertEqual(pricing_by_id["minimax-m3-via-litellm"], {"input_per_1m": 0.3, "output_per_1m": 1.2})


class TargetRepositoryTest(unittest.TestCase):
    """ターゲットリポジトリのテスト。"""

    def test_default_targets_include_litellm_examples(self) -> None:
        """設定ファイルなしでもLiteLLM例が初期表示できる。"""

        repository = TargetRepository(config_path=Path("/tmp/does-not-exist-api-targets.json"))
        targets = repository.load()
        providers = {target["id"]: target["provider"] for target in targets}

        self.assertEqual(providers["gemini-3-1-pro-via-litellm"], "litellm")
        self.assertEqual(providers["minimax-m3-via-litellm"], "litellm")

    def test_default_targets_include_deepseek_pro_direct(self) -> None:
        """標準比較対象にDeepSeek Proのnon-thinking設定が含まれている。"""

        repository = TargetRepository(config_path=Path("/tmp/does-not-exist-api-targets.json"))
        targets = repository.load()
        target_ids = {target["id"] for target in targets}

        self.assertIn("deepseek-v4-pro-direct", target_ids)

    def test_default_targets_include_representative_current_models_with_prices(self) -> None:
        """標準比較対象には主要プロバイダの代表モデルと単価が含まれる。"""

        repository = TargetRepository(config_path=Path("/tmp/does-not-exist-api-targets.json"))
        targets = repository.load()
        targets_by_id = {target["id"]: target for target in targets}

        self.assertEqual(targets_by_id["openai-gpt-5-5"]["model"], "gpt-5.5")
        self.assertEqual(targets_by_id["openai-gpt-5-5"]["pricing"]["output_per_1m"], 30.0)
        self.assertEqual(targets_by_id["anthropic-claude-sonnet-5"]["model"], "claude-sonnet-5")
        self.assertEqual(targets_by_id["anthropic-claude-sonnet-5"]["pricing"]["input_per_1m"], 2.0)
        self.assertEqual(targets_by_id["gemini-3-flash-via-litellm"]["model"], "gemini/gemini-3-flash-preview")
        self.assertEqual(targets_by_id["gemini-3-flash-via-litellm"]["pricing"]["output_per_1m"], 3.0)


if __name__ == "__main__":
    unittest.main()
