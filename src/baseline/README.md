# Baselines

## Contents

- `rulebase/`
  - Dictionary-based implementation
  - Result: `res/rulebase/ingredient_matching_rulebase.json`
- `bm25/`
  - Character 2-gram BM25 implementation
  - Result: `res/bm25/ingredient_matching_bm25.json`
- `embedding/`
  - Embedding + cosine similarity implementation
  - Results:
    - `res/sentence_bert/ingredient_matching_sentence_bert.json`
    - `res/multilingual_e5_large/ingredient_matching_multilingual_e5_large.json`
    - `res/qwen3_8b/ingredient_matching_qwen3_8b.json`
- `rag/`
  - Top-k retrieval + LLM reranking implementation
- `llm/`
  - LLM client definitions referenced by the RAG implementation
