from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from embedding_model_defs import build_embedder, list_model_keys, resolve_model_spec

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INGREDIENT_CSV = BASE_DIR / "data" / "ingredient_label.csv"
DEFAULT_FOODTABLE_CSV = (
    BASE_DIR / "data" / "data_nutrient_FoodTableJP" / "data_preprocessed" / "food_nutrition_all.csv"
)
TOP_K = 10


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


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


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


def _build_foodtable_name_to_id(foodtable_df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for _, row in foodtable_df[["food_name", "food_code"]].dropna().iterrows():
        food_name = str(row["food_name"])
        if food_name in mapping:
            continue
        mapping[food_name] = _to_id_str(row["food_code"])
    return mapping


def _load_ingredient_embedding_map(path: Path) -> Dict[str, np.ndarray]:
    payload = _load_json(path)
    embedding_map: Dict[str, np.ndarray] = {}

    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items", [])
        for item in items:
            ingredient_name = str(item.get("ingredient", ""))
            embedding = item.get("embedding")
            if not ingredient_name or embedding is None:
                continue
            embedding_map[ingredient_name] = np.asarray(embedding, dtype=np.float32)
        return embedding_map

    if isinstance(payload, dict):
        # backward compatibility with old format: {ingredient_name: embedding}
        for ingredient_name, embedding in payload.items():
            embedding_map[str(ingredient_name)] = np.asarray(embedding, dtype=np.float32)
        return embedding_map

    raise ValueError(f"Unsupported ingredient embedding format: {path}")


def _load_foodtable_embeddings(
    path: Path,
    foodtable_name_to_id: Dict[str, str],
) -> Tuple[List[str], List[str], np.ndarray]:
    payload = _load_json(path)

    food_codes: List[str] = []
    food_names: List[str] = []
    vectors: List[np.ndarray] = []

    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items", [])
        for item in items:
            food_code = _to_id_str(item.get("food_code", "")).strip()
            food_name = str(item.get("food_name", "")).strip()
            embedding = item.get("embedding")
            if not food_code or not food_name or embedding is None:
                continue
            food_codes.append(food_code)
            food_names.append(food_name)
            vectors.append(np.asarray(embedding, dtype=np.float32))
    elif isinstance(payload, dict):
        # backward compatibility with old format: {food_name: embedding}
        for food_name, embedding in payload.items():
            food_name_str = str(food_name)
            food_code = foodtable_name_to_id.get(food_name_str)
            if food_code is None:
                continue
            food_codes.append(food_code)
            food_names.append(food_name_str)
            vectors.append(np.asarray(embedding, dtype=np.float32))
    else:
        raise ValueError(f"Unsupported foodtable embedding format: {path}")

    if not vectors:
        raise ValueError(f"No foodtable embeddings loaded from: {path}")

    matrix = np.vstack(vectors).astype(np.float32)
    return food_codes, food_names, matrix


def _create_ingredient_embeddings(
    ingredient_df: pd.DataFrame,
    encode_texts: Callable[[List[str]], Any],
    output_path: Path,
    model_key: str,
    model_name: str,
) -> Dict[str, np.ndarray]:
    ingredient_names = (
        ingredient_df["ingredient_name"].dropna().astype(str).drop_duplicates().tolist()
    )
    embeddings = encode_texts(ingredient_names)

    items = []
    embedding_map: Dict[str, np.ndarray] = {}
    for ingredient_name, embedding in zip(ingredient_names, embeddings):
        embedding_np = np.asarray(embedding, dtype=np.float32)
        embedding_map[ingredient_name] = embedding_np
        items.append(
            {
                "ingredient": ingredient_name,
                "embedding": embedding_np.tolist(),
            }
        )

    payload = {
        "model_key": model_key,
        "model_name": model_name,
        "items": items,
    }
    _save_json(output_path, payload)
    return embedding_map


def _create_foodtable_embeddings(
    foodtable_df: pd.DataFrame,
    encode_texts: Callable[[List[str]], Any],
    output_path: Path,
    model_key: str,
    model_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    target_df = foodtable_df[["food_code", "food_name"]].dropna().copy()
    target_df["food_code"] = target_df["food_code"].map(_to_id_str)
    target_df["food_name"] = target_df["food_name"].astype(str)

    target_df = target_df[
        (target_df["food_code"].str.len() > 0) & (target_df["food_name"].str.len() > 0)
    ]

    food_codes = target_df["food_code"].tolist()
    food_names = target_df["food_name"].tolist()

    embeddings = encode_texts(food_names)
    vectors = np.asarray(embeddings, dtype=np.float32)

    items = []
    for food_code, food_name, embedding in zip(food_codes, food_names, vectors):
        items.append(
            {
                "food_code": food_code,
                "food_name": food_name,
                "embedding": embedding.tolist(),
            }
        )

    payload = {
        "model_key": model_key,
        "model_name": model_name,
        "items": items,
    }
    _save_json(output_path, payload)
    return food_codes, food_names, vectors


def _get_top_candidates(
    ingredient_embedding: np.ndarray,
    food_codes: List[str],
    food_names: List[str],
    normalized_food_matrix: np.ndarray,
) -> List[Dict[str, Any]]:
    scores = normalized_food_matrix.dot(ingredient_embedding)
    if scores.size == 0:
        return []

    k = min(TOP_K, scores.shape[0])
    if k == scores.shape[0]:
        sorted_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    candidates: List[Dict[str, Any]] = []
    for index in sorted_indices[:k]:
        candidates.append(
            {
                "id": food_codes[index],
                "food_name": food_names[index],
                "score": float(scores[index]),
            }
        )
    return candidates


def _run_matching(
    ingredient_df: pd.DataFrame,
    ingredient_embedding_map: Dict[str, np.ndarray],
    food_codes: List[str],
    food_names: List[str],
    food_matrix: np.ndarray,
) -> Tuple[List[Dict[str, Any]], float, int]:
    normalized_food_matrix = _normalize_rows(food_matrix)

    results: List[Dict[str, Any]] = []
    correct_predictions = 0
    effective_count = 0

    for _, row in tqdm(ingredient_df.iterrows(), total=len(ingredient_df)):
        ingredient_name = str(row["ingredient_name"])
        actual_foodtable_id = _to_json_scalar(row["foodtable_id"])
        actual_foodtable_id_str = _to_id_str(row["foodtable_id"])

        embedding = ingredient_embedding_map.get(ingredient_name)
        top10_candidates: List[Dict[str, Any]] = []
        predicted_id: Optional[str] = None

        if embedding is not None:
            normalized_ingredient = _normalize_vector(np.asarray(embedding, dtype=np.float32))
            if float(np.linalg.norm(normalized_ingredient)) > 0.0:
                top10_candidates = _get_top_candidates(
                    normalized_ingredient,
                    food_codes,
                    food_names,
                    normalized_food_matrix,
                )
                if top10_candidates:
                    predicted_id = top10_candidates[0]["id"]
                    effective_count += 1
                    if predicted_id == actual_foodtable_id_str:
                        correct_predictions += 1

        record = {
            "ingredient": ingredient_name,
            "foodtable_maping": actual_foodtable_id,
            "response": {"id": str(predicted_id)} if predicted_id is not None else None,
            "top10_candidates": top10_candidates,
        }
        results.append(record)

    accuracy = (correct_predictions / effective_count) * 100 if effective_count > 0 else 0.0
    return results, accuracy, effective_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding baseline matching runner with cache-aware embedding generation.",
    )
    parser.add_argument("--model", required=True, choices=list_model_keys())
    parser.add_argument(
        "--device",
        default=None,
        help="Device string for the embedding backend (e.g., cuda:1, cuda:0, cpu).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "res")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--ingredient-csv", type=Path, default=DEFAULT_INGREDIENT_CSV)
    parser.add_argument("--foodtable-csv", type=Path, default=DEFAULT_FOODTABLE_CSV)
    parser.add_argument("--force-ingredient-embedding", action="store_true")
    parser.add_argument("--force-foodtable-embedding", action="store_true")
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Only build/load embedding caches and skip nearest-neighbor matching.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    spec = resolve_model_spec(args.model)
    output_tag = args.output_tag if args.output_tag else spec.output_tag
    output_dir = args.output_root / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    ingredient_embedding_path = output_dir / f"ingredient_embedding_{output_tag}.json"
    foodtable_embedding_path = output_dir / f"foodtable_embedding_{output_tag}.json"
    matching_output_path = output_dir / f"ingredient_matching_{output_tag}.json"

    ingredient_df = _load_ingredient_df(args.ingredient_csv)
    foodtable_df = _load_foodtable_df(args.foodtable_csv)
    foodtable_name_to_id = _build_foodtable_name_to_id(foodtable_df)

    needs_ingredient_embedding = args.force_ingredient_embedding or not ingredient_embedding_path.exists()
    needs_foodtable_embedding = args.force_foodtable_embedding or not foodtable_embedding_path.exists()

    encode_texts: Optional[Callable[[List[str]], Any]] = None
    resolved_device: Optional[str] = args.device

    if needs_ingredient_embedding or needs_foodtable_embedding:
        encode_texts, _, resolved_device = build_embedder(
            model_key=args.model,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )

    if needs_ingredient_embedding:
        if encode_texts is None:
            raise RuntimeError("Embedder is not initialized for ingredient embedding generation.")
        print(f"[BUILD] ingredient embeddings: {ingredient_embedding_path}")
        ingredient_embedding_map = _create_ingredient_embeddings(
            ingredient_df=ingredient_df,
            encode_texts=encode_texts,
            output_path=ingredient_embedding_path,
            model_key=args.model,
            model_name=spec.hf_model_name,
        )
    else:
        print(f"[SKIP] ingredient embeddings already exist: {ingredient_embedding_path}")
        ingredient_embedding_map = _load_ingredient_embedding_map(ingredient_embedding_path)

    if needs_foodtable_embedding:
        if encode_texts is None:
            raise RuntimeError("Embedder is not initialized for foodtable embedding generation.")
        print(f"[BUILD] foodtable embeddings: {foodtable_embedding_path}")
        food_codes, food_names, food_matrix = _create_foodtable_embeddings(
            foodtable_df=foodtable_df,
            encode_texts=encode_texts,
            output_path=foodtable_embedding_path,
            model_key=args.model,
            model_name=spec.hf_model_name,
        )
    else:
        print(f"[SKIP] foodtable embeddings already exist: {foodtable_embedding_path}")
        food_codes, food_names, food_matrix = _load_foodtable_embeddings(
            foodtable_embedding_path,
            foodtable_name_to_id,
        )

    print(f"Model key: {args.model}")
    print(f"Model name: {spec.hf_model_name}")
    print(f"Device: {resolved_device}")
    print(f"Output directory: {output_dir}")

    if args.skip_matching:
        print("[DONE] embeddings are ready. Matching was skipped by --skip-matching.")
        return

    results, accuracy, effective_count = _run_matching(
        ingredient_df=ingredient_df,
        ingredient_embedding_map=ingredient_embedding_map,
        food_codes=food_codes,
        food_names=food_names,
        food_matrix=food_matrix,
    )

    _save_json(matching_output_path, results)

    print(f"[DONE] matching results: {matching_output_path}")
    print(f"Accuracy: {accuracy:.2f}% ({effective_count} effective rows)")


if __name__ == "__main__":
    main()
