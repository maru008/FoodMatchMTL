from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INGREDIENT_CSV = BASE_DIR / "data" / "ingredient_label.csv"
DEFAULT_FOODTABLE_CSV = (
    BASE_DIR / "data" / "data_nutrient_FoodTableJP" / "data_preprocessed" / "food_nutrition_all.csv"
)
DEFAULT_ALIAS_DICT_JSON = (
    BASE_DIR / "src" / "FoodMatchMTL" / "train_data" / "ingredient_alias_dict.json"
)
DEFAULT_FOODTABLE_NAME2ID_JSON = BASE_DIR / "data" / "foodtable_name2id.json"
DEFAULT_TOP_K = 10

_MULTI_SPACE_PATTERN = re.compile(r"\s+")


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


def _normalize_lookup_key(text: str) -> str:
    normalized = str(text).replace("\u3000", " ").strip().lower()
    normalized = _MULTI_SPACE_PATTERN.sub(" ", normalized)
    return normalized


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_alias_dict(path: Path) -> Dict[str, List[str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"alias dict must be JSON object: {path}")

    alias_dict: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            alias_dict[str(key)] = [str(item) for item in value]
            continue
        alias_dict[str(key)] = [str(value)]
    return alias_dict


def _load_foodtable_name2id(path: Path) -> Dict[str, str]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"foodtable_name2id must be JSON object: {path}")

    normalized: Dict[str, str] = {}
    for raw_name, raw_id in payload.items():
        key = _normalize_lookup_key(str(raw_name))
        food_id = _normalize_food_id(raw_id)
        if not key or food_id is None:
            continue
        normalized[key] = food_id
    return normalized


def _build_food_id_to_name(foodtable_df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for _, row in foodtable_df[["food_code", "food_name"]].dropna().iterrows():
        food_id = _normalize_food_id(row["food_code"])
        food_name = str(row["food_name"]).strip()
        if food_id is None or not food_name:
            continue
        if food_id in mapping:
            continue
        mapping[food_id] = food_name
    return mapping


def _iter_alias_terms(canonical_name: str, aliases: Sequence[str]) -> Iterable[str]:
    # Canonical name is also a valid alias lookup term.
    yield canonical_name
    for alias in aliases:
        yield alias


def _build_alias_to_food_ids(
    alias_dict: Dict[str, List[str]],
    foodtable_name2id: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], int]:
    alias_to_food_ids: Dict[str, Set[str]] = defaultdict(set)
    missing_key_count = 0

    for canonical_name, aliases in alias_dict.items():
        food_id = foodtable_name2id.get(_normalize_lookup_key(canonical_name))
        if food_id is None:
            missing_key_count += 1
            continue

        for term in _iter_alias_terms(canonical_name, aliases):
            key = _normalize_lookup_key(term)
            if not key:
                continue
            alias_to_food_ids[key].add(food_id)

    return alias_to_food_ids, missing_key_count


def _build_candidates(
    candidate_ids: Sequence[str],
    food_id_to_name: Dict[str, str],
    top_k: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for food_id in candidate_ids[:top_k]:
        candidates.append(
            {
                "id": food_id,
                "food_name": food_id_to_name.get(food_id, ""),
                "score": 1.0,
            }
        )
    return candidates


def _run_matching(
    ingredient_df: pd.DataFrame,
    alias_to_food_ids: Dict[str, Set[str]],
    food_id_to_name: Dict[str, str],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    exact_match_count = 0
    category_match_count = 0
    effective_count = 0
    evaluable_count = 0
    ambiguous_count = 0

    for _, row in tqdm(ingredient_df.iterrows(), total=len(ingredient_df)):
        ingredient_name = str(row["ingredient_name"])
        ingredient_key = _normalize_lookup_key(ingredient_name)
        actual_foodtable_id = _to_json_scalar(row["foodtable_id"])
        actual_foodtable_id_norm = _normalize_food_id(row["foodtable_id"])

        candidate_ids = sorted(alias_to_food_ids.get(ingredient_key, set()))
        top_candidates = _build_candidates(candidate_ids, food_id_to_name, top_k=top_k)

        predicted_id: Optional[str] = None
        if len(candidate_ids) == 1:
            predicted_id = candidate_ids[0]
            effective_count += 1
        elif len(candidate_ids) > 1:
            # User requirement: when multiple candidates exist, prediction is null.
            ambiguous_count += 1

        predicted_id_norm = _normalize_food_id(predicted_id)
        if actual_foodtable_id_norm is not None:
            evaluable_count += 1
            if predicted_id_norm is not None and predicted_id_norm == actual_foodtable_id_norm:
                exact_match_count += 1
            if (
                predicted_id_norm is not None
                and len(predicted_id_norm) >= 2
                and predicted_id_norm[:2] == actual_foodtable_id_norm[:2]
            ):
                category_match_count += 1

        results.append(
            {
                "ingredient": ingredient_name,
                "foodtable_maping": actual_foodtable_id,
                "response": {"id": predicted_id} if predicted_id is not None else None,
                "top10_candidates": top_candidates,
            }
        )

    acc = (exact_match_count / evaluable_count) if evaluable_count > 0 else 0.0
    em = acc
    category_match = (category_match_count / evaluable_count) if evaluable_count > 0 else 0.0

    metrics = {
        "acc": float(acc),
        "em": float(em),
        "category_match": float(category_match),
        "total_rows": int(len(ingredient_df)),
        "evaluable_rows": int(evaluable_count),
        "effective_rows": int(effective_count),
        "ambiguous_rows": int(ambiguous_count),
        "exact_match_count": int(exact_match_count),
        "category_match_count": int(category_match_count),
    }
    return results, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline matching runner aligned with BM25 output format.",
    )
    parser.add_argument("--ingredient-csv", type=Path, default=DEFAULT_INGREDIENT_CSV)
    parser.add_argument("--foodtable-csv", type=Path, default=DEFAULT_FOODTABLE_CSV)
    parser.add_argument("--alias-dict-json", type=Path, default=DEFAULT_ALIAS_DICT_JSON)
    parser.add_argument("--foodtable-name2id-json", type=Path, default=DEFAULT_FOODTABLE_NAME2ID_JSON)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "res")
    parser.add_argument("--output-tag", default="rulebase")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = args.output_root / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_output_path = output_dir / f"ingredient_matching_{args.output_tag}.json"

    ingredient_df = _load_ingredient_df(args.ingredient_csv)
    foodtable_df = _load_foodtable_df(args.foodtable_csv)
    alias_dict = _load_alias_dict(args.alias_dict_json)
    foodtable_name2id = _load_foodtable_name2id(args.foodtable_name2id_json)
    food_id_to_name = _build_food_id_to_name(foodtable_df)
    alias_to_food_ids, missing_key_count = _build_alias_to_food_ids(alias_dict, foodtable_name2id)

    results, metrics = _run_matching(
        ingredient_df=ingredient_df,
        alias_to_food_ids=alias_to_food_ids,
        food_id_to_name=food_id_to_name,
        top_k=args.top_k,
    )
    _save_json(matching_output_path, results)

    print(f"Output directory: {output_dir}")
    print(f"Alias dictionary: {args.alias_dict_json}")
    print(f"Alias keys missing in foodtable_name2id: {missing_key_count}")
    print(f"[DONE] matching results: {matching_output_path}")
    print(
        "Rows: total={total} evaluable={evaluable} effective={effective} ambiguous={ambiguous}".format(
            total=metrics["total_rows"],
            evaluable=metrics["evaluable_rows"],
            effective=metrics["effective_rows"],
            ambiguous=metrics["ambiguous_rows"],
        )
    )
    print(f"Acc: {metrics['acc']:.4f}")
    print(f"EM: {metrics['em']:.4f}")
    print(f"CategoryMatch: {metrics['category_match']:.4f}")


if __name__ == "__main__":
    main()
