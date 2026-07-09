# LLM API Comparator

OpenAI、Anthropic、DeepSeek、LiteLLM経由の各種LLM APIを、同じプロンプトで並行比較するためのローカルGUIです。回答内容、実行時間、トークン数、概算費用、finish reasonを1画面で比較できます。

## DeepSeekターゲットの前提

- OpenAI互換の `base_url` は `https://api.deepseek.com`
- V4のモデル名は `deepseek-v4-flash` と `deepseek-v4-pro`
- `deepseek-chat` / `deepseek-reasoner` は 2026-07-24 15:59 UTC に非推奨予定
- Thinking mode はターゲットごとに有効・無効を切り替えられます。

## セットアップ

APIキーはファイルに保存せず、実行時の環境変数に設定してください。

```bash
export DEEPSEEK_API_KEY='<your DeepSeek API key>'
export OPENAI_API_KEY='<your OpenAI API key>'
export ANTHROPIC_API_KEY='<your Anthropic API key>'
```

## GUIで試す

ローカルGUIを起動します。

```bash
python3 scripts/api_comparator_gui.py
```

ブラウザで `http://127.0.0.1:8765` を開いてください。APIキー本体は画面には入力せず、環境変数または `config/api_targets.json` で管理します。

GUIでは任意の質問を入力し、複数の比較対象を並行実行できます。標準では以下を同時比較します。

- `deepseek-v4-flash` / thinking有効
- `deepseek-v4-flash` / thinking無効
- `deepseek-v4-pro` / thinking有効
- `deepseek-v4-pro` / thinking無効

OpenAI、Anthropic、Gemini、MiniMaxの代表モデルも価格付きで初期登録されていますが、APIキー未設定時のエラーを避けるため未選択です。チェックを入れるだけで比較対象へ追加できます。

比較画面には、回答内容、実行時間、トークン数、概算費用、finish reason、reasoning contentが表示されます。
各実行にはRun IDが付き、GUIの実行履歴から過去の質問・回答・メトリクスを再表示できます。

比較対象はGUI上で追加・編集・削除できます。画面上の編集はブラウザ内に保持されます。

## 比較専用ページ

`http://127.0.0.1:8765/compare` では、保存済みの実行履歴を選んで視覚的に比較できます。

- 実行時間、概算費用、総トークン数の横棒グラフ
- APIごとのメトリクス表
- 回答プレビューと回答全文
- 最速、最安、合計概算費用のサマリー

## 比較対象とAPIキーの外部設定

`config/api_targets.example.json` をコピーして `config/api_targets.json` を作成すると、比較対象を外部ファイルから読み込めます。

```bash
cp config/api_targets.example.json config/api_targets.json
```

APIキーは `api_key_env` に環境変数名を書き、実行前に環境変数へ設定するのが推奨です。

```bash
export DEEPSEEK_API_KEY='...'
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
export MINIMAX_API_KEY='...'
```

`config/api_targets.json` は `.gitignore` 済みです。代表モデルの価格は設定済みですが、各社の価格改定や長文・Priority・Batch・キャッシュ・リージョン課金は別単価になる場合があるため、必要に応じて各ターゲットの `pricing.input_per_1m` と `pricing.output_per_1m` を上書きしてください。

## 代表モデルと採用単価

概算費用の初期値は、公式価格ページの通常API利用・標準テキスト入出力・100万トークンあたりUSDを採用しています。レスポンスから返る金額ではなく、APIが返すトークン使用量にこの単価を掛けた推定値です。

| Provider | Model | Input / 1M | Output / 1M | 備考 |
| --- | --- | ---: | ---: | --- |
| DeepSeek | `deepseek-v4-flash` | $0.14 | $0.28 | Cache miss前提 |
| DeepSeek | `deepseek-v4-pro` | $0.435 | $0.87 | Cache miss前提 |
| OpenAI | `gpt-5.5` | $5.00 | $30.00 | Standard / short context |
| OpenAI | `gpt-5.4-mini` | $0.75 | $4.50 | Standard / short context |
| Anthropic | `claude-fable-5` | $10.00 | $50.00 | Claude API |
| Anthropic | `claude-opus-4-8` | $5.00 | $25.00 | Claude API |
| Anthropic | `claude-sonnet-5` | $2.00 | $10.00 | 2026-08-31までのintroductory pricing |
| Anthropic | `claude-haiku-4-5-20251001` | $1.00 | $5.00 | Claude API |
| Google Gemini | `gemini-3.1-pro-preview` | $2.00 | $12.00 | Standard、200k tokens以下 |
| Google Gemini | `gemini-3-flash-preview` | $0.50 | $3.00 | Standard |
| Google Gemini | `gemini-3.1-flash-lite` | $0.25 | $1.50 | Standard |
| MiniMax | `MiniMax-M3` | $0.30 | $1.20 | Standard、512k tokens以下、50% off適用後 |
| MiniMax | `MiniMax-M2.7` | $0.30 | $1.20 | Standard |

価格確認元: DeepSeek API Docs、OpenAI API Pricing、Anthropic Claude Platform Docs、Google Gemini API Pricing、MiniMax API Docs。
DeepSeekのピーク時間帯料金告知の控え: [docs/assets/deepseek-usage-pricing-notice-2026-07-09.png](docs/assets/deepseek-usage-pricing-notice-2026-07-09.png)

## 構成

GUIバックエンドは責務ごとに分割しています。

- `scripts/api_comparator_gui.py`: GUI起動用エントリーポイント
- `scripts/api_comparator_probe.py`: DeepSeekターゲット向けCLI検証ツール
- `api_comparator/server.py`: HTTPルーティング
- `api_comparator/config.py`: 比較ターゲット設定、APIキー解決、公開用設定変換
- `api_comparator/defaults.py`: プロバイダとモデルの既定値
- `api_comparator/provider_registry.py`: provider名とAPIクライアントの対応付け
- `api_comparator/providers.py`: OpenAI互換、OpenAI、AnthropicのAPI呼び出し
- `api_comparator/comparison.py`: 比較サマリー生成と入力検証
- `api_comparator/services.py`: 並行実行、結果集約、概算費用計算
- `api_comparator/paths.py`: パス定義
- `web/api_comparator_gui.html`: GUI本体

## 拡張方針

OpenAI互換APIなら、基本的には `config/api_targets.json` にターゲットを追加するだけで比較できます。MinimaxなどがOpenAI互換エンドポイントを提供する場合は `provider: "openai_compatible"`、`base_url`、`model`、`api_key_env` を指定してください。

GeminiやMinimaxなどをLiteLLM経由でまとめて扱いたい場合は、`provider: "litellm"` を指定できます。LiteLLMは任意依存です。未インストールの場合、そのターゲットだけエラーになります。

```json
{
  "id": "gemini-3-flash-via-litellm",
  "label": "Gemini via LiteLLM",
  "provider": "litellm",
  "model": "gemini/gemini-3-flash-preview",
  "api_key_env": "GEMINI_API_KEY",
  "pricing": {
    "input_per_1m": 0.5,
    "output_per_1m": 3.0
  }
}
```

LiteLLMを導入する場合は、供給網リスクを確認し、信頼できるバージョンに固定してください。現時点ではプロジェクトの必須依存にはしていません。

ネイティブ形式のAPIを追加する場合は、`api_comparator/providers.py` に `ProviderClient` 実装を追加し、`api_comparator/provider_registry.py` の `build_default_provider_registry()` で登録します。既存の比較サービスやHTTPルーティングは変更しない方針です。

比較方法を変える場合は、`api_comparator/comparison.py` に `SummaryBuilder` 実装を追加し、`ComparisonService` へ注入します。速度・費用・回答長以外の評価軸を追加しても、プロバイダ呼び出し側には影響させません。

## CLIで試す

Flash と Pro の両方を、thinking 有効・無効の両方で比較します。

```bash
python3 scripts/api_comparator_probe.py --models deepseek-v4-flash deepseek-v4-pro --thinking both
```

ストリーミングで初回トークン時間も確認します。

```bash
python3 scripts/api_comparator_probe.py --models deepseek-v4-flash --thinking enabled --stream
```

特定ケースだけ実行します。

```bash
python3 scripts/api_comparator_probe.py --cases jp_summary coding --models deepseek-v4-pro
```

## 結果の見方

標準出力に概要が表示され、詳細は `results/` 配下のJSONLに保存されます。

- `latency_sec`: API呼び出し全体の秒数
- `ttft_sec`: ストリーミング時の初回テキスト受信までの秒数
- `total_tokens`: APIが返した総トークン数
- `estimated_cost_usd`: 公式価格をもとにした概算費用
- `content_preview`: 回答の先頭部分

費用はキャッシュミス前提で概算しています。実請求はDeepSeek側の利用明細を確認してください。

## 実行履歴とログ

GUIで比較を実行すると、後から見返すための履歴が `results/api_comparator_history.jsonl` に保存されます。履歴にはRun ID、質問、回答、reasoning content、実行時間、トークン数、概算費用、サマリーが含まれます。APIキーやAuthorizationヘッダーは保存しません。

GUI上部の「実行履歴」から過去の実行を選ぶと、その時の質問と比較結果を再表示できます。

監査・デバッグ用の軽量ログは `results/api_comparator.log` に以下のイベントとして追記されます。

- `run_started`: 実行開始、Run ID、対象数、最大出力トークン数
- `target_result`: APIごとのモデル、実行時間、総トークン数、概算費用、エラー
- `run_finished`: 実行終了、成功数、エラー数、総実行時間

軽量ログにはAPIキー、Authorizationヘッダー、プロンプト全文、回答全文は出力しません。GitHub公開時に履歴とログを含めないため、`results/` は `.gitignore` 済みです。

## SSL証明書エラーへの対処

`[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate` が出る場合、PythonがCA証明書を見つけられていません。このアプリは `certifi` がインストール済みなら自動的にそのCA bundleを使います。

それでも失敗する場合は、macOSのpython.org版Pythonなら同梱の `Install Certificates.command` を実行してください。別環境では以下のようにCA bundleを明示できます。

```bash
python3 -m pip install certifi
export SSL_CERT_FILE="$(python3 -m certifi)"
python3 scripts/api_comparator_gui.py
```

証明書検証の無効化は推奨しません。
