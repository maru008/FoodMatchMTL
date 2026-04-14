from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INGREDIENT_CSV = BASE_DIR / "data" / "ingredient_label.csv"
DEFAULT_FOODTABLE_CSV = (
    BASE_DIR / "data" / "data_nutrient_FoodTableJP" / "data_preprocessed" / "food_nutrition_all.csv"
)
DEFAULT_TOP_K = 10
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_ASCII_UNIT_PATTERN = re.compile(r"(?i)(?<=\d)\s*(kg|g|ml|cc)\b|\b(kg|g|ml|cc)\b")
_JP_UNIT_TERMS = ("大さじ", "小さじ", "適量", "少々")
_SYMBOLS_TO_SPACE = ",.・/()（）[]【】「」『』:：；;!！?？-ー_"


def _build_fullwidth_alnum_translation() -> Dict[int, str]:
    table: Dict[int, str] = {}
    for index in range(10):
        table[ord(chr(ord("０") + index))] = chr(ord("0") + index)
    for index in range(26):
        table[ord(chr(ord("Ａ") + index))] = chr(ord("A") + index)
        table[ord(chr(ord("ａ") + index))] = chr(ord("a") + index)
    return table


_FULLWIDTH_ALNUM_TRANSLATION = _build_fullwidth_alnum_translation()
_SYMBOL_TRANSLATION = str.maketrans({symbol: " " for symbol in _SYMBOLS_TO_SPACE})


def _to_id_str(value: Any) -> str:
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value)


def _to_json_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
        return float(value)
    return value


def _normalize_food_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        number = int(float(text))
        return f"{number:05d}"
    except ValueError:
        digits = re.sub(r"\D", "", text)
        if not digits:
            return None
        return digits.zfill(5)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    normalized = text.translate(_FULLWIDTH_ALNUM_TRANSLATION)
    normalized = normalized.lower()
    normalized = normalized.translate(_SYMBOL_TRANSLATION)

    for term in _JP_UNIT_TERMS:
        normalized = normalized.replace(term, " ")
    normalized = _ASCII_UNIT_PATTERN.sub(" ", normalized)
    normalized = _MULTI_SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def tokenize_char_bigrams(text: str) -> List[str]:
    if len(text) < 2:
        return []

    tokens: List[str] = []
    for index in range(len(text) - 1):
        token = text[index : index + 2]
        if " " in token:
            continue
        tokens.append(token)
    return tokens


def _load_ingredient_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = ["ingredient_name", "foodtable_id"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"ingredient csv missing columns: {missing}")
    return df


def _load_foodtable_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required_columns = ["food_code", "food_name"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"foodtable csv missing columns: {missing}")
    return df


def _pick_alias_columns(foodtable_df: pd.DataFrame) -> List[str]:
    exact_candidates = {
        "alias",
        "aliases",
        "food_alias",
        "food_aliases",
        "synonym",
        "synonyms",
        "別名",
        "別称",
    }
    columns: List[str] = []
    for column in foodtable_df.columns:
        lowered = column.lower()
        if lowered in exact_candidates:
            columns.append(column)
            continue
        if "alias" in lowered or "synonym" in lowered or "別名" in column or "別称" in column:
            columns.append(column)

    deduped: List[str] = []
    seen = set()
    for column in columns:
        if column in seen or column == "food_name":
            continue
        deduped.append(column)
        seen.add(column)
    return deduped


def _pick_food_group_columns(foodtable_df: pd.DataFrame) -> List[str]:
    exact_candidates = {"food_group", "food_group_name", "食品群", "食品群名"}

    columns: List[str] = []
    for column in foodtable_df.columns:
        lowered = column.lower()
        if lowered in exact_candidates:
            columns.append(column)
            continue
        if "group" in lowered or "群" in column:
            columns.append(column)

    deduped: List[str] = []
    seen = set()
    for column in columns:
        if column in seen:
            continue
        deduped.append(column)
        seen.add(column)
    return deduped


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _join_fields(values: Iterable[str]) -> str:
    merged: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        merged.append(value)
        seen.add(value)
    return " ".join(merged).strip()


def _create_ingredient_query_cache(
    ingredient_df: pd.DataFrame,
    output_path: Path,
    output_tag: str,
) -> Dict[str, List[str]]:
    ingredient_names = (
        ingredient_df["ingredient_name"].dropna().astype(str).drop_duplicates().tolist()
    )

    items = []
    query_token_map: Dict[str, List[str]] = {}
    for ingredient_name in ingredient_names:
        normalized = normalize_text(ingredient_name)
        tokens = tokenize_char_bigrams(normalized)
        query_token_map[ingredient_name] = tokens
        items.append(
            {
                "ingredient": ingredient_name,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

    payload = {
        "output_tag": output_tag,
        "tokenizer": "char_bigram",
        "items": items,
    }
    _save_json(output_path, payload)
    return query_token_map


def _load_ingredient_query_cache(path: Path) -> Dict[str, List[str]]:
    payload = _load_json(path)
    query_token_map: Dict[str, List[str]] = {}

    if isinstance(payload, dict) and "items" in payload:
        for item in payload.get("items", []):
            ingredient = str(item.get("ingredient", ""))
            tokens = item.get("tokens", [])
            if not ingredient or not isinstance(tokens, list):
                continue
            query_token_map[ingredient] = [str(token) for token in tokens]
        return query_token_map

    if isinstance(payload, dict):
        for ingredient, tokens in payload.items():
            if isinstance(tokens, list):
                query_token_map[str(ingredient)] = [str(token) for token in tokens]
        return query_token_map

    raise ValueError(f"Unsupported ingredient query cache format: {path}")


def _create_foodtable_index_cache(
    foodtable_df: pd.DataFrame,
    output_path: Path,
    output_tag: str,
) -> List[Dict[str, Any]]:
    alias_columns = _pick_alias_columns(foodtable_df)
    group_columns = _pick_food_group_columns(foodtable_df)

    columns = ["food_code", "food_name", *alias_columns, *group_columns]
    target_df = foodtable_df[columns].copy()

    items: List[Dict[str, Any]] = []
    for _, row in target_df.iterrows():
        food_code = _to_id_str(row["food_code"]).strip()
        food_name = _safe_text(row["food_name"])
        if not food_code or not food_name:
            continue

        text_fields = [_safe_text(row["food_name"])]
        text_fields.extend(_safe_text(row[column]) for column in alias_columns)
        text_fields.extend(_safe_text(row[column]) for column in group_columns)

        document = _join_fields(text_fields)
        normalized = normalize_text(document)
        tokens = tokenize_char_bigrams(normalized)
        items.append(
            {
                "food_code": food_code,
                "food_name": food_name,
                "document": document,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

    payload = {
        "output_tag": output_tag,
        "tokenizer": "char_bigram",
        "alias_columns": alias_columns,
        "group_columns": group_columns,
        "items": items,
    }
    _save_json(output_path, payload)
    return items


def _load_foodtable_index_cache(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items", [])
        if isinstance(items, list):
            return items

    if isinstance(payload, list):
        # Backward compatibility if only item list was saved.
        return payload

    raise ValueError(f"Unsupported foodtable index cache format: {path}")


class BM25Index:
    def __init__(self, items: Sequence[Dict[str, Any]]):
        self.food_codes: List[str] = []
        self.food_names: List[str] = []
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for item in items:
            food_code = _to_id_str(item.get("food_code", "")).strip()
            food_name = str(item.get("food_name", "")).strip()
            tokens_raw = item.get("tokens", [])
            if not food_code or not food_name or not isinstance(tokens_raw, list):
                continue

            tokens = [str(token) for token in tokens_raw if isinstance(token, str) and token]
            term_freq = Counter(tokens)

            self.food_codes.append(food_code)
            self.food_names.append(food_name)
            self.doc_lengths.append(len(tokens))

            current_index = len(self.food_codes) - 1
            for token, freq in term_freq.items():
                self.postings[token].append((current_index, int(freq)))

        self.document_count = len(self.food_codes)
        self.avgdl = (
            float(sum(self.doc_lengths)) / float(self.document_count)
            if self.document_count > 0
            else 0.0
        )

    def search(self, query_tokens: Sequence[str], top_k: int, k1: float, b: float) -> List[Dict[str, Any]]:
        if self.document_count == 0:
            return []
        if top_k <= 0:
            return []
        if not query_tokens:
            return []

        query_term_freq = Counter(query_tokens)
        score_map: Dict[int, float] = defaultdict(float)
        n_docs = float(self.document_count)

        for token, qtf in query_term_freq.items():
            posting = self.postings.get(token)
            if not posting:
                continue

            df = float(len(posting))
            idf = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)
            for doc_index, tf in posting:
                doc_length = float(self.doc_lengths[doc_index])
                length_ratio = (doc_length / self.avgdl) if self.avgdl > 0 else 0.0
                denominator = float(tf) + k1 * (1.0 - b + b * length_ratio)
                if denominator == 0.0:
                    continue
                score = idf * ((float(tf) * (k1 + 1.0)) / denominator)
                score_map[doc_index] += float(qtf) * score

        if not score_map:
            return []

        ranked = sorted(score_map.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        candidates: List[Dict[str, Any]] = []
        for doc_index, score in ranked:
            candidates.append(
                {
                    "id": self.food_codes[doc_index],
                    "food_name": self.food_names[doc_index],
                    "score": float(score),
                }
            )
        return candidates


def _run_matching(
    ingredient_df: pd.DataFrame,
    query_token_map: Dict[str, List[str]],
    bm25_index: BM25Index,
    top_k: int,
    k1: float,
    b: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    exact_match_count = 0
    category_match_count = 0
    effective_exact_match_count = 0
    effective_count = 0
    evaluable_count = 0

    for _, row in tqdm(ingredient_df.iterrows(), total=len(ingredient_df)):
        ingredient_name = str(row["ingredient_name"])
        actual_foodtable_id = _to_json_scalar(row["foodtable_id"])
        actual_foodtable_id_norm = _normalize_food_id(row["foodtable_id"])

        query_tokens = query_token_map.get(ingredient_name, [])
        top_candidates: List[Dict[str, Any]] = []
        predicted_id: Optional[str] = None

        if query_tokens:
            top_candidates = bm25_index.search(
                query_tokens=query_tokens,
                top_k=top_k,
                k1=k1,
                b=b,
            )
            if top_candidates:
                predicted_id = top_candidates[0]["id"]
                effective_count += 1

        predicted_id_norm = _normalize_food_id(predicted_id)

        if actual_foodtable_id_norm is not None:
            evaluable_count += 1

            if predicted_id_norm is not None:
                if predicted_id_norm == actual_foodtable_id_norm:
                    effective_exact_match_count += 1
                    exact_match_count += 1
                if predicted_id_norm[:2] == actual_foodtable_id_norm[:2]:
                    category_match_count += 1

        record = {
            "ingredient": ingredient_name,
            "foodtable_maping": actual_foodtable_id,
            "response": {"id": str(predicted_id)} if predicted_id is not None else None,
            "top10_candidates": top_candidates,
        }
        results.append(record)

    acc = (exact_match_count / evaluable_count) if evaluable_count > 0 else 0.0
    em = acc
    category_match = (category_match_count / evaluable_count) if evaluable_count > 0 else 0.0
    effective_acc = (
        effective_exact_match_count / effective_count if effective_count > 0 else 0.0
    )

    metrics = {
        "acc": float(acc),
        "em": float(em),
        "category_match": float(category_match),
        "effective_acc": float(effective_acc),
        "total_rows": int(len(ingredient_df)),
        "evaluable_rows": int(evaluable_count),
        "effective_rows": int(effective_count),
        "exact_match_count": int(exact_match_count),
        "category_match_count": int(category_match_count),
    }
    return results, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BM25 baseline matching runner with cache-aware tokenization.",
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "res")
    parser.add_argument("--output-tag", default="bm25")
    parser.add_argument("--ingredient-csv", type=Path, default=DEFAULT_INGREDIENT_CSV)
    parser.add_argument("--foodtable-csv", type=Path, default=DEFAULT_FOODTABLE_CSV)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--force-ingredient-query", action="store_true")
    parser.add_argument("--force-foodtable-index", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_tag = args.output_tag
    output_dir = args.output_root / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    ingredient_query_path = output_dir / f"ingredient_query_{output_tag}.json"
    foodtable_index_path = output_dir / f"foodtable_index_{output_tag}.json"
    matching_output_path = output_dir / f"ingredient_matching_{output_tag}.json"

    ingredient_df = _load_ingredient_df(args.ingredient_csv)
    foodtable_df = _load_foodtable_df(args.foodtable_csv)

    needs_ingredient_query = args.force_ingredient_query or not ingredient_query_path.exists()
    needs_foodtable_index = args.force_foodtable_index or not foodtable_index_path.exists()

    if needs_ingredient_query:
        print(f"[BUILD] ingredient query cache: {ingredient_query_path}")
        ingredient_query_map = _create_ingredient_query_cache(
            ingredient_df=ingredient_df,
            output_path=ingredient_query_path,
            output_tag=output_tag,
        )
    else:
        print(f"[SKIP] ingredient query cache already exists: {ingredient_query_path}")
        ingredient_query_map = _load_ingredient_query_cache(ingredient_query_path)

    if needs_foodtable_index:
        print(f"[BUILD] foodtable index cache: {foodtable_index_path}")
        foodtable_items = _create_foodtable_index_cache(
            foodtable_df=foodtable_df,
            output_path=foodtable_index_path,
            output_tag=output_tag,
        )
    else:
        print(f"[SKIP] foodtable index cache already exists: {foodtable_index_path}")
        foodtable_items = _load_foodtable_index_cache(foodtable_index_path)

    bm25_index = BM25Index(foodtable_items)

    print(f"Output directory: {output_dir}")
    print(f"Top-k: {args.top_k}")
    print(f"BM25 k1: {args.k1}")
    print(f"BM25 b: {args.b}")
    print(f"Documents indexed: {bm25_index.document_count}")
    print(f"Average document length: {bm25_index.avgdl:.4f}")

    results, metrics = _run_matching(
        ingredient_df=ingredient_df,
        query_token_map=ingredient_query_map,
        bm25_index=bm25_index,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
    )
    _save_json(matching_output_path, results)

    print(f"[DONE] matching results: {matching_output_path}")
    print(
        "Rows: total={total} evaluable={evaluable} effective={effective}".format(
            total=metrics["total_rows"],
            evaluable=metrics["evaluable_rows"],
            effective=metrics["effective_rows"],
        )
    )
    print(f"Acc: {metrics['acc']:.4f}")
    print(f"EM: {metrics['em']:.4f}")
    print(f"CategoryMatch: {metrics['category_match']:.4f}")


if __name__ == "__main__":
    main()
