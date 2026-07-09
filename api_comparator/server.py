"""API比較GUIのHTTPサーバー。"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import TargetRepository
from .defaults import DEFAULT_MODELS, MODEL_PRICING_USD_PER_1M_TOKENS
from .paths import STATIC_INDEX_PATH, TARGET_CONFIG_PATH
from .services import ComparisonService


class ApiComparatorHandler(BaseHTTPRequestHandler):
    """API比較GUI用のHTTPリクエストハンドラー。"""

    server_version = "ApiComparatorGui/2.0"
    repository = TargetRepository()
    comparison_service = ComparisonService(repository=repository)

    def do_HEAD(self) -> None:
        """HEADリクエストを処理する。"""

        if self.path in {"/", "/index.html", "/api/config"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        """GETリクエストを処理する。"""

        if self.path in {"/", "/index.html"}:
            self.send_text(self.load_index_html(), content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/config":
            self.handle_config()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """POSTリクエストを処理する。"""

        if self.path != "/api/compare":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            request_body = self.read_json_body()
            result = self.comparison_service.compare(request_body)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
            return

        self.send_json(result)

    def handle_config(self) -> None:
        """GUI設定レスポンスを返す。"""

        try:
            compare_targets = self.repository.load()
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_json(
            {
                "models": DEFAULT_MODELS,
                "pricing": MODEL_PRICING_USD_PER_1M_TOKENS,
                "compare_targets": self.repository.public_targets(compare_targets),
                "config_path": str(TARGET_CONFIG_PATH),
            }
        )

    def read_json_body(self) -> dict[str, Any]:
        """リクエストボディをJSONとして読み込む。

        Returns:
            JSONオブジェクト。

        Raises:
            ValueError: JSONが不正、またはオブジェクトでない場合。
        """

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is empty.")
        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body is not valid JSON.") from exc
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def send_text(self, text: str, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        """テキストレスポンスを送信する。

        Args:
            text: 送信する文字列。
            content_type: Content-Typeヘッダー。
            status: HTTPステータス。
        """

        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """JSONレスポンスを送信する。

        Args:
            value: JSON化する辞書。
            status: HTTPステータス。
        """

        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def load_index_html(self) -> str:
        """GUIのHTMLを読み込む。

        Returns:
            表示するHTML文字列。
        """

        return STATIC_INDEX_PATH.read_text(encoding="utf-8")

    def log_message(self, format: str, *args: Any) -> None:
        """APIキーをログへ出さない最小限のアクセスログを出力する。

        Args:
            format: ログ書式。
            args: ログ引数。
        """

        print(f"{self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        解析済み引数。
    """

    parser = argparse.ArgumentParser(description="API Comparator GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def run_server() -> int:
    """GUIサーバーを起動する。

    Returns:
        プロセス終了コード。
    """

    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ApiComparatorHandler)
    print(f"API Comparator GUI: http://{args.host}:{args.port}")
    print("APIキーは環境変数、またはconfig/api_targets.jsonで指定してください。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
