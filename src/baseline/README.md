# Baselines

`4.experiment` に掲載した baseline を公開用にまとめたディレクトリです。

## 収録内容

- `rulebase/`
  - 辞書ベース実装
  - 結果: `res/rulebase/ingredient_matching_rulebase.json`
- `bm25/`
  - 文字 2-gram BM25 実装
  - 結果: `res/bm25/ingredient_matching_bm25.json`
- `embedding/`
  - Embedding + cosine 実装
  - 結果:
    - `res/sentence_bert/ingredient_matching_sentence_bert.json`
    - `res/multilingual_e5_large/ingredient_matching_multilingual_e5_large.json`
    - `res/qwen3_8b/ingredient_matching_qwen3_8b.json`
- `rag/`
  - Top-k retrieval + LLM reranking 実装
  - `4.experiment` で使った `k=10` の結果のみを収録
- `llm/`
  - RAG 実装が参照する LLM client 定義

## 補足

- 大きな embedding cache 本体は含めていません。
- notebook と論文表で使う最終結果 JSON は含めています。
