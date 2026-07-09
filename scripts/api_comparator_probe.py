#!/usr/bin/env python3
"""LLM API Comparator向けのDeepSeek V4検証CLI。

APIキーを環境変数から読み取り、代表的な日本語・コーディング・推論タスクを
DeepSeek V4モデルへ送信します。結果は標準出力とJSONLファイルへ保存します。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
DEFAULT_OUTPUT_DIR = Path("results")

MODEL_PRICING_USD_PER_1M_TOKENS = {
    "deepseek-v4-flash": {"input_cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input_cache_miss": 0.435, "output": 0.87},
}


@dataclass(frozen=True)
class ProbeCase:
    """検証ケースを表す値オブジェクト。

    Attributes:
        name: ケース名。
        description: 評価観点の説明。
        messages: Chat Completions APIに渡すメッセージ配列。
        max_tokens: 最大出力トークン数。
        response_format: 構造化出力などで指定するレスポンス形式。
    """

    name: str
    description: str
    messages: list[dict[str, str]]
    max_tokens: int
    response_format: dict[str, str] | None = None


@dataclass(frozen=True)
class ProbeResult:
    """API検証結果を表す値オブジェクト。

    Attributes:
        timestamp: UTCのISO 8601タイムスタンプ。
        case_name: 検証ケース名。
        model: 使用モデル名。
        thinking: Thinking modeの指定。
        stream: ストリーミング有無。
        latency_sec: リクエスト全体の所要秒数。
        ttft_sec: 初回テキスト受信までの秒数。非ストリーミング時はNone。
        prompt_tokens: 入力トークン数。
        completion_tokens: 出力トークン数。
        total_tokens: 総トークン数。
        estimated_cost_usd: キャッシュミス前提の概算費用。
        content: 最終回答本文。
        reasoning_content: Thinking modeで返る推論テキスト。
        error: エラー内容。成功時はNone。
    """

    timestamp: str
    case_name: str
    model: str
    thinking: str
    stream: bool
    latency_sec: float
    ttft_sec: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    content: str
    reasoning_content: str
    error: str | None


def build_cases() -> dict[str, ProbeCase]:
    """標準の検証ケースを構築する。

    Returns:
        ケース名をキーにした検証ケース辞書。
    """

    return {
        "jp_summary": ProbeCase(
            name="jp_summary",
            description="日本語の要約品質、論点整理、自然さを確認する。",
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは日本語で簡潔かつ実務的に回答するAIアシスタントです。",
                },
                {
                    "role": "user",
                    "content": (
                        "以下の状況を、経営会議向けに「要点」「懸念」「次のアクション」の"
                        "3見出しで200字程度に要約してください。\n\n"
                        "新規SaaSのβ版は想定より早く利用企業が増えた一方、オンボーディングに"
                        "時間がかかり、サポート工数が週30時間を超えています。解約率は低いものの、"
                        "管理者向け権限設定と監査ログへの要望が強く、営業は大企業向け提案を急ぎたい"
                        "と言っています。開発チームは既存のデータ同期処理の遅延も抱えています。"
                    ),
                },
            ],
        ),
        "coding": ProbeCase(
            name="coding",
            description="コード読解、バグ修正、説明の具体性を確認する。",
            max_tokens=900,
            messages=[
                {
                    "role": "system",
                    "content": "あなたはPythonに強いシニアエンジニアです。日本語で回答してください。",
                },
                {
                    "role": "user",
                    "content": (
                        "次の関数は、空配列や負数を含む入力で意図しない結果になります。"
                        "バグを説明し、型ヒント付きで修正版を書いてください。\n\n"
                        "```python\n"
                        "def average_positive(nums):\n"
                        "    total = 0\n"
                        "    count = 0\n"
                        "    for n in nums:\n"
                        "        if n > 0:\n"
                        "            total += n\n"
                        "        count += 1\n"
                        "    return total / count\n"
                        "```"
                    ),
                },
            ],
        ),
        "reasoning": ProbeCase(
            name="reasoning",
            description="推論問題での正確性と説明のわかりやすさを確認する。",
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは推論過程を簡潔に整理し、最終結論を明確に示すアシスタントです。",
                },
                {
                    "role": "user",
                    "content": (
                        "A、B、C、Dの4人がいます。AはBより速く、CはDより遅い。"
                        "BはDより速く、CはAより遅い。この条件から確実に言える順位関係を説明してください。"
                    ),
                },
            ],
        ),
        "json": ProbeCase(
            name="json",
            description="構造化出力の安定性を確認する。",
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "必ずjsonだけを返してください。Markdownや説明文は不要です。\n"
                        "例: {\"category\":\"billing\",\"priority\":\"high\",\"reply_summary\":\"請求書の差し替え依頼\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "次の問い合わせを分類し、JSONで返してください。"
                        "スキーマは {\"category\": string, \"priority\": \"low\"|\"medium\"|\"high\", "
                        "\"reply_summary\": string} です。\n\n"
                        "問い合わせ: 請求書の金額が契約書と違います。今月中に監査があるため、"
                        "本日中に正しい請求書へ差し替えてください。"
                    ),
                },
            ],
        ),
    }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        解析済み引数。
    """

    parser = argparse.ArgumentParser(description="LLM API Comparator DeepSeek検証CLI")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cases", nargs="+", default=list(build_cases().keys()))
    parser.add_argument("--thinking", choices=["enabled", "disabled", "both"], default="both")
    parser.add_argument("--effort", choices=["high", "max"], default="high")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def create_payload(
    *,
    model: str,
    case: ProbeCase,
    thinking: Literal["enabled", "disabled"],
    effort: str,
    stream: bool,
) -> dict[str, Any]:
    """Chat Completions APIのペイロードを作成する。

    Args:
        model: 使用モデル名。
        case: 検証ケース。
        thinking: Thinking modeの指定。
        effort: 推論努力量。
        stream: ストリーミング有無。

    Returns:
        APIへ送信するJSONペイロード。
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": case.messages,
        "max_tokens": case.max_tokens,
        "stream": stream,
        "thinking": {"type": thinking},
    }
    if thinking == "enabled":
        payload["reasoning_effort"] = effort
    if case.response_format:
        payload["response_format"] = case.response_format
    return payload


def post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """JSON POSTでAPIを呼び出す。

    Args:
        url: APIエンドポイントURL。
        api_key: DeepSeek APIキー。
        payload: 送信ペイロード。
        timeout: タイムアウト秒数。

    Returns:
        レスポンスJSON。

    Raises:
        RuntimeError: APIがエラーを返した場合。
    """

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def iter_stream_events(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> Iterable[dict[str, Any]]:
    """Server-Sent Events形式のストリームを逐次JSONとして返す。

    Args:
        url: APIエンドポイントURL。
        api_key: DeepSeek APIキー。
        payload: 送信ペイロード。
        timeout: タイムアウト秒数。

    Yields:
        SSEのdata行から復元したJSONイベント。

    Raises:
        RuntimeError: APIがエラーを返した場合。
    """

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_line = line.removeprefix("data: ").strip()
                if data_line == "[DONE]":
                    break
                yield json.loads(data_line)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def estimate_cost_usd(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    """公式価格からキャッシュミス前提の概算費用を計算する。

    Args:
        model: 使用モデル名。
        prompt_tokens: 入力トークン数。
        completion_tokens: 出力トークン数。

    Returns:
        概算費用。価格またはトークン数が不明な場合はNone。
    """

    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None
    input_cost = prompt_tokens * pricing["input_cache_miss"] / 1_000_000
    output_cost = completion_tokens * pricing["output"] / 1_000_000
    return round(input_cost + output_cost, 8)


def run_non_streaming_probe(
    *,
    url: str,
    api_key: str,
    model: str,
    case: ProbeCase,
    thinking: Literal["enabled", "disabled"],
    effort: str,
    timeout: float,
) -> ProbeResult:
    """非ストリーミングで1件の検証を実行する。

    Args:
        url: APIエンドポイントURL。
        api_key: DeepSeek APIキー。
        model: 使用モデル名。
        case: 検証ケース。
        thinking: Thinking modeの指定。
        effort: 推論努力量。
        timeout: タイムアウト秒数。

    Returns:
        検証結果。
    """

    started = time.perf_counter()
    try:
        response = post_json(
            url=url,
            api_key=api_key,
            payload=create_payload(model=model, case=case, thinking=thinking, effort=effort, stream=False),
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        message = response["choices"][0]["message"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return ProbeResult(
            timestamp=utc_now(),
            case_name=case.name,
            model=model,
            thinking=thinking,
            stream=False,
            latency_sec=round(latency, 3),
            ttft_sec=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens"),
            estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return error_result(model=model, case=case, thinking=thinking, stream=False, started=started, error=exc)


def run_streaming_probe(
    *,
    url: str,
    api_key: str,
    model: str,
    case: ProbeCase,
    thinking: Literal["enabled", "disabled"],
    effort: str,
    timeout: float,
) -> ProbeResult:
    """ストリーミングで1件の検証を実行する。

    Args:
        url: APIエンドポイントURL。
        api_key: DeepSeek APIキー。
        model: 使用モデル名。
        case: 検証ケース。
        thinking: Thinking modeの指定。
        effort: 推論努力量。
        timeout: タイムアウト秒数。

    Returns:
        検証結果。
    """

    started = time.perf_counter()
    first_text_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, int] = {}
    try:
        events = iter_stream_events(
            url=url,
            api_key=api_key,
            payload=create_payload(model=model, case=case, thinking=thinking, effort=effort, stream=True),
            timeout=timeout,
        )
        for event in events:
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content_piece = delta.get("content") or ""
            reasoning_piece = delta.get("reasoning_content") or ""
            if first_text_at is None and (content_piece or reasoning_piece):
                first_text_at = time.perf_counter()
            if content_piece:
                content_parts.append(content_piece)
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
        latency = time.perf_counter() - started
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return ProbeResult(
            timestamp=utc_now(),
            case_name=case.name,
            model=model,
            thinking=thinking,
            stream=True,
            latency_sec=round(latency, 3),
            ttft_sec=round(first_text_at - started, 3) if first_text_at else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens"),
            estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return error_result(model=model, case=case, thinking=thinking, stream=True, started=started, error=exc)


def error_result(
    *,
    model: str,
    case: ProbeCase,
    thinking: str,
    stream: bool,
    started: float,
    error: Exception,
) -> ProbeResult:
    """例外を検証結果として整形する。

    Args:
        model: 使用モデル名。
        case: 検証ケース。
        thinking: Thinking modeの指定。
        stream: ストリーミング有無。
        started: 開始時刻。
        error: 発生した例外。

    Returns:
        エラー情報を含む検証結果。
    """

    return ProbeResult(
        timestamp=utc_now(),
        case_name=case.name,
        model=model,
        thinking=thinking,
        stream=stream,
        latency_sec=round(time.perf_counter() - started, 3),
        ttft_sec=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
        content="",
        reasoning_content="",
        error=str(error),
    )


def utc_now() -> str:
    """現在時刻をUTCのISO 8601形式で返す。

    Returns:
        UTCタイムスタンプ。
    """

    return datetime.now(timezone.utc).isoformat()


def selected_thinking_modes(value: str) -> list[Literal["enabled", "disabled"]]:
    """Thinking mode指定を実行対象の配列に変換する。

    Args:
        value: CLIで指定された値。

    Returns:
        実行するThinking modeの配列。
    """

    if value == "both":
        return ["enabled", "disabled"]
    return [value]  # type: ignore[list-item]


def write_jsonl(path: Path, result: ProbeResult) -> None:
    """検証結果をJSONLへ追記する。

    Args:
        path: 出力先パス。
        result: 検証結果。
    """

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def print_result(result: ProbeResult) -> None:
    """検証結果の概要を標準出力へ表示する。

    Args:
        result: 検証結果。
    """

    header = f"{result.model} | {result.case_name} | thinking={result.thinking} | stream={result.stream}"
    if result.error:
        print(f"[ERROR] {header} | {result.error}", flush=True)
        return

    preview = result.content.replace("\n", " ")[:160]
    print(
        "[OK] "
        f"{header} | latency={result.latency_sec}s"
        f" | ttft={result.ttft_sec}s"
        f" | tokens={result.total_tokens}"
        f" | cost=${result.estimated_cost_usd}"
        f" | preview={preview}",
        flush=True,
    )


def validate_cases(selected_case_names: list[str], cases: dict[str, ProbeCase]) -> list[ProbeCase]:
    """指定されたケース名を検証し、実行対象ケースを返す。

    Args:
        selected_case_names: CLIで指定されたケース名。
        cases: 利用可能な検証ケース辞書。

    Returns:
        実行対象ケース配列。

    Raises:
        SystemExit: 存在しないケース名が指定された場合。
    """

    unknown = sorted(set(selected_case_names) - set(cases))
    if unknown:
        available = ", ".join(sorted(cases))
        raise SystemExit(f"Unknown case: {', '.join(unknown)}. Available cases: {available}")
    return [cases[name] for name in selected_case_names]


def main() -> int:
    """CLIのエントリーポイント。

    Returns:
        プロセス終了コード。
    """

    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set.", file=sys.stderr)
        return 2

    cases = validate_cases(args.cases, build_cases())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"api_comparator_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    url = args.base_url.rstrip("/") + "/chat/completions"

    for _ in range(args.repeat):
        for model in args.models:
            for thinking in selected_thinking_modes(args.thinking):
                for case in cases:
                    if args.stream:
                        result = run_streaming_probe(
                            url=url,
                            api_key=api_key,
                            model=model,
                            case=case,
                            thinking=thinking,
                            effort=args.effort,
                            timeout=args.timeout,
                        )
                    else:
                        result = run_non_streaming_probe(
                            url=url,
                            api_key=api_key,
                            model=model,
                            case=case,
                            thinking=thinking,
                            effort=args.effort,
                            timeout=args.timeout,
                        )
                    write_jsonl(output_path, result)
                    print_result(result)

    print(f"\nSaved results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
