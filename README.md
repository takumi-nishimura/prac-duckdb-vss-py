# Agent Conversation Simulation & Summarization with Vector Search

## 概要

このプロジェクトは、言語モデルを用いたエージェント間の会話シミュレーションを行い、その会話の要約と埋め込みベクトルをデータベースに保存します。さらに、会話のテーマに基づいて、過去の類似した会話要約をベクトル検索する機能を提供します。

## 主な機能

* **会話シミュレーション:** 2つのAIエージェントが、動的に生成されたユニークなテーマについて会話をシミュレートします。
* **要約とデータベース保存:** 生成された会話全体の要約が作成され、その要約テキストと埋め込みベクトルがタイムスタンプと共にデータベース (DuckDB) に保存されます。
* **ベクトル検索:** 会話のテーマをクエリとして使用し、データベースに蓄積された過去の会話要約の中から、ベクトル類似度が高いものを検索して表示します。

## 要件

* Python 3.x
* 必要なライブラリ ( `pyproject.toml` 参照):
    * `duckdb`
    * `litellm`
    * `transformers`
    * `torch` (Transformersのバックエンドとして)
    * `sentencepiece` (特定のTokenizerで必要)
* ローカルLLM環境 (例: Ollama): スクリプト内で指定されている以下のモデルが利用可能であること。
    * 会話生成モデル: `ollama/hf.co/mmnga/ArrowMint-Gemma3-4B-YUKI-v0.1-gguf:latest`
    * 要約生成モデル: `ollama/gemma3:27b`
    * *(注意: 上記モデル名は `main.py` 内の `CHAT_MODEL_NAME`, `SUMMARY_MODEL_NAME` で定義されています。環境に合わせて変更可能です)*
* 埋め込みモデル: `pfnet/plamo-embedding-1b`
    * *(注意: このモデルは初回実行時にHugging Face Hubから自動的にダウンロードされます)*

## セットアップ

1.  **リポジトリのクローン:**
    ```bash
    git clone https://github.com/takumi-nishimura/prac-duckdb-vss-py.git
    cd prac-duckdb-vss-py
    ```
2.  **依存関係のインストール:**
    `uv` (推奨) または `pip` を使用して依存関係をインストールします。
    ```bash
    # uv を使用する場合
    uv sync

    # pip を使用する場合 (uv がない場合)
    pip install duckdb litellm transformers torch sentencepiece
    ```

## 実行方法

以下のコマンドでスクリプトを実行します (`uv` を使用する場合)。

```bash
uv run main.py
```
