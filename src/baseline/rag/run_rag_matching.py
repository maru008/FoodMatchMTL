from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from rag_model_clients import (
    build_rag_runtime,
    list_embedding_model_keys,
    list_llm_model_keys,
    list_rag_preset_keys,
    resolve_rag_preset,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INGREDIENT_CSV = BASE_DIR / "data" / "ingredient_label.csv"
DEFAULT_FOODTABLE_CSV = (
    BASE_DIR / "data" / "data_nutrient_FoodTableJP" / "data_preprocessed" / "food_nutrition_all.csv"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "res"
DEFAULT_EMBEDDING_ROOT = BASE_DIR / "src" / "baseline" / "embedding" / "res"


def _parse_parallel_gpu_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None

    gpu_ids: list[int] = []
    for chunk in value.split(","):
        part = chunk.strip()
        if not part:
            continue
        try:
            gpu_id = int(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid GPU id '{part}' in --parallel-gpu-ids."
            ) from error
        if gpu_id < 0:
            raise argparse.ArgumentTypeError("GPU ids must be non-negative.")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    return tuple(gpu_ids) if gpu_ids else None


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json(text: str) -> Dict[str, Any]:
    codeblock_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
    match = codeblock_pattern.search(text)
    candidate = match.group(1).strip() if match else text.strip()

    brace_pattern = re.compile(r"\{[\s\S]*\}")
    match = brace_pattern.search(candidate)
    if match:
        candidate = match.group(0)

    return json.loads(candidate)


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
    ingredient_df = pd.read_csv(path)
    required = ["ingredient_name", "foodtable_id"]
    missing = [column for column in required if column not in ingredient_df.columns]
    if missing:
        raise ValueError(f"ingredient csv missing columns: {missing}")
    return ingredient_df


def _load_foodtable_df(path: Path) -> pd.DataFrame:
    foodtable_df = pd.read_csv(path, low_memory=False)
    required = ["food_code", "food_name"]
    missing = [column for column in required if column not in foodtable_df.columns]
    if missing:
        raise ValueError(f"foodtable csv missing columns: {missing}")
    return foodtable_df


def _build_foodtable_name_to_id(foodtable_df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for _, row in foodtable_df[["food_name", "food_code"]].dropna().iterrows():
        name = str(row["food_name"])
        if name in mapping:
            continue
        mapping[name] = _to_id_str(row["food_code"])
    return mapping


def _resolve_embedding_paths(
    embedding_root: Path,
    embedding_tag: str,
) -> Tuple[Path, Path]:
    nested_dir = embedding_root / embedding_tag
    ingredient_path = nested_dir / f"ingredient_embedding_{embedding_tag}.json"
    foodtable_path = nested_dir / f"foodtable_embedding_{embedding_tag}.json"

    if ingredient_path.exists() and foodtable_path.exists():
        return ingredient_path, foodtable_path

    legacy_ingredient = embedding_root / f"ingredient_embedding_{embedding_tag}.json"
    legacy_foodtable = embedding_root / f"foodtable_embedding_{embedding_tag}.json"
    if legacy_ingredient.exists() and legacy_foodtable.exists():
        return legacy_ingredient, legacy_foodtable

    missing = []
    if not ingredient_path.exists() and not legacy_ingredient.exists():
        missing.append(str(ingredient_path))
    if not foodtable_path.exists() and not legacy_foodtable.exists():
        missing.append(str(foodtable_path))

    raise FileNotFoundError(
        "Embedding cache files are not found. Please run baseline_embedding first. "
        f"Missing: {missing}"
    )


def _load_ingredient_embeddings(path: Path) -> Dict[str, np.ndarray]:
    payload = _load_json(path)
    embedding_map: Dict[str, np.ndarray] = {}

    if isinstance(payload, dict) and "items" in payload:
        for item in payload.get("items", []):
            name = str(item.get("ingredient", ""))
            embedding = item.get("embedding")
            if not name or embedding is None:
                continue
            embedding_map[name] = np.asarray(embedding, dtype=np.float32)
        return embedding_map

    if isinstance(payload, dict):
        for name, embedding in payload.items():
            embedding_map[str(name)] = np.asarray(embedding, dtype=np.float32)
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
        for item in payload.get("items", []):
            food_code = _to_id_str(item.get("food_code", "")).strip()
            food_name = str(item.get("food_name", "")).strip()
            embedding = item.get("embedding")
            if not food_name or embedding is None:
                continue
            if not food_code:
                food_code = foodtable_name_to_id.get(food_name, "")
            if not food_code:
                continue
            food_codes.append(food_code)
            food_names.append(food_name)
            vectors.append(np.asarray(embedding, dtype=np.float32))
    elif isinstance(payload, dict):
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


def _create_prompt_topk(ingredient_name: str, candidates_topk: List[Dict[str, Any]]) -> str:
    lines = []
    for candidate in candidates_topk:
        code = str(candidate["food_code"])
        name = str(candidate["food_name"]).replace("\u3000", " ")
        lines.append(f"{code}\t{name}")
    candidate_text = "\n".join(lines)

    prompt = (
        "あなたは食材名の照合アシスタントです。\n"
        "下の候補一覧（food_code\\tfood_name）から、入力食材名に最も対応する1件を選び、"
        "その food_code のみを JSON で返してください。\n\n"
        "※ 解説や他の文字は一切出力しないこと。id は必ず候補のいずれかである。\n\n"
        f"入力食材名: {ingredient_name}\n\n"
        "【候補一覧】\n"
        "food_code\tfood_name\n"
        "----------------------------------------\n"
        f"{candidate_text}\n"
        "----------------------------------------\n"
        "【出力】次の1行のみ：\n"
        '{"id":"<food_code>"}\n'
    )
    return prompt


def _retrieve_topk_candidates(
    ingredient_embedding: np.ndarray,
    food_codes: List[str],
    food_names: List[str],
    normalized_food_matrix: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    scores = normalized_food_matrix.dot(ingredient_embedding)
    if scores.size == 0:
        return []

    k_eff = max(1, min(k, scores.shape[0]))
    if k_eff == scores.shape[0]:
        sorted_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -k_eff)[-k_eff:]
        sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    topk = []
    for index in sorted_indices[:k_eff]:
        topk.append(
            {
                "food_code": food_codes[index],
                "food_name": food_names[index],
                "sim": float(scores[index]),
            }
        )
    return topk


def _compute_metrics(
    records: List[Dict[str, Any]],
    ingredient_df: pd.DataFrame,
) -> Dict[str, Any]:
    result_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    result_by_ingredient: Dict[str, Dict[str, Any]] = {}
    for record in records:
        ingredient = record.get("ingredient")
        if ingredient is None:
            continue
        ingredient_str = str(ingredient)
        result_by_ingredient[ingredient_str] = record
        gold_id = _normalize_food_id(record.get("foodtable_maping"))
        if gold_id is not None:
            result_by_pair[(ingredient_str, gold_id)] = record

    exact_match_count = 0
    category_match_count = 0
    effective_count = 0
    evaluable_count = 0

    for _, row in ingredient_df.iterrows():
        ingredient_name = str(row["ingredient_name"])
        gold_id = _normalize_food_id(row["foodtable_id"])
        if gold_id is None:
            continue

        evaluable_count += 1
        record = result_by_pair.get((ingredient_name, gold_id))
        if record is None:
            record = result_by_ingredient.get(ingredient_name)
        predicted_id: Optional[Any] = None
        if isinstance(record, dict):
            response = record.get("response")
            if isinstance(response, dict):
                predicted_id = response.get("id")

        predicted_norm = _normalize_food_id(predicted_id)
        if predicted_norm is not None:
            effective_count += 1
            if predicted_norm == gold_id:
                exact_match_count += 1
            if predicted_norm[:2] == gold_id[:2]:
                category_match_count += 1

    acc = (exact_match_count / evaluable_count) if evaluable_count > 0 else 0.0
    category_match = (category_match_count / evaluable_count) if evaluable_count > 0 else 0.0
    return {
        "acc": float(acc),
        "em": float(acc),
        "category_match": float(category_match),
        "total_rows": int(len(ingredient_df)),
        "evaluable_rows": int(evaluable_count),
        "effective_rows": int(effective_count),
        "exact_match_count": int(exact_match_count),
        "category_match_count": int(category_match_count),
    }


def run_rag_matching(
    ingredient_df: pd.DataFrame,
    ingredient_embedding_map: Dict[str, np.ndarray],
    food_codes: List[str],
    food_names: List[str],
    food_matrix: np.ndarray,
    infer_fn,
    output_path: Path,
    k: int,
    max_retries: int = 5,
    save_every: int = 50,
    limit: Optional[int] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        results = _load_json(output_path)
        if not isinstance(results, list):
            raise ValueError(f"Output json is not a list: {output_path}")
        print(f"[INFO] Resume from existing output: {output_path}")
    else:
        results = []

    if limit is not None and limit > 0:
        ingredient_df = ingredient_df.head(limit)

    done_keys: set[Tuple[str, str]] = set()
    for item in results:
        ingredient = item.get("ingredient")
        gold_id = _normalize_food_id(item.get("foodtable_maping"))
        if ingredient is None or gold_id is None:
            continue
        done_keys.add((str(ingredient), gold_id))

    normalized_food_matrix = _normalize_rows(food_matrix)
    retry_limit = max(5, int(max_retries))

    for _, row in tqdm(ingredient_df.iterrows(), total=len(ingredient_df)):
        ingredient_name = str(row["ingredient_name"])
        gold_id = _normalize_food_id(row["foodtable_id"])
        if gold_id is not None and (ingredient_name, gold_id) in done_keys:
            continue

        record: Dict[str, Any] = {
            "ingredient": ingredient_name,
            "foodtable_maping": _to_json_scalar(row["foodtable_id"]),
            "candidates_topk": [],
            "prompt": None,
            "response": None,
            "retray_count": 0,
        }
        try:
            ingredient_embedding = ingredient_embedding_map.get(ingredient_name)
            if ingredient_embedding is None:
                results.append(record)
                if gold_id is not None:
                    done_keys.add((ingredient_name, gold_id))
                if len(results) % save_every == 0:
                    _save_json(output_path, results)
                continue

            normalized_ingredient = _normalize_vector(np.asarray(ingredient_embedding, dtype=np.float32))
            if float(np.linalg.norm(normalized_ingredient)) == 0.0:
                results.append(record)
                if gold_id is not None:
                    done_keys.add((ingredient_name, gold_id))
                if len(results) % save_every == 0:
                    _save_json(output_path, results)
                continue

            candidates_topk = _retrieve_topk_candidates(
                ingredient_embedding=normalized_ingredient,
                food_codes=food_codes,
                food_names=food_names,
                normalized_food_matrix=normalized_food_matrix,
                k=k,
            )
            record["candidates_topk"] = candidates_topk
            if not candidates_topk:
                results.append(record)
                if gold_id is not None:
                    done_keys.add((ingredient_name, gold_id))
                if len(results) % save_every == 0:
                    _save_json(output_path, results)
                continue

            prompt = _create_prompt_topk(ingredient_name, candidates_topk)
            record["prompt"] = prompt

            parsed: Optional[Dict[str, Any]] = None
            retry_count = 0
            last_error: Optional[str] = None

            for attempt in range(retry_limit):
                try:
                    raw_content = infer_fn(prompt)
                except Exception as error:
                    retry_count += 1
                    last_error = f"inference_failed: {error}"
                    print(f"[WARN] inference failed (attempt {attempt + 1}): {error}")
                    continue

                try:
                    parsed = _extract_json(raw_content)
                    break
                except Exception as error:
                    retry_count += 1
                    last_error = f"json_parse_failed: {error}"
                    print(f"[WARN] json parse failed (attempt {attempt + 1}): {error}")
                    continue

            record["retray_count"] = retry_count
            record["response"] = parsed
            if parsed is None and last_error is not None:
                record["error"] = last_error

        except Exception as error:
            record["error"] = f"record_failed: {error}"
            record["response"] = None
            print(f"[ERROR] failed to process ingredient '{ingredient_name}': {error}")

        results.append(record)
        if gold_id is not None:
            done_keys.add((ingredient_name, gold_id))

        if len(results) % save_every == 0:
            _save_json(output_path, results)

    _save_json(output_path, results)

    metrics = _compute_metrics(results, ingredient_df)
    print("\n=== RAG result ===")
    print(
        "Rows: total={total} evaluable={evaluable} effective={effective}".format(
            total=metrics["total_rows"],
            evaluable=metrics["evaluable_rows"],
            effective=metrics["effective_rows"],
        )
    )
    print(
        "Correct: {correct} / {evaluable}".format(
            correct=metrics["exact_match_count"],
            evaluable=metrics["evaluable_rows"],
        )
    )
    print(f"Acc: {metrics['acc']:.4f}")
    print(f"EM: {metrics['em']:.4f}")
    print(f"CategoryMatch: {metrics['category_match']:.4f}")
    print(f"Saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG runner using selected LLM model and embedding model caches.",
    )

    parser.add_argument("--rag-preset", choices=list_rag_preset_keys(), default=None)
    parser.add_argument("--llm-model", choices=list_llm_model_keys(), default=None)
    parser.add_argument("--embedding-model", choices=list_embedding_model_keys(), default=None)

    parser.add_argument(
        "--provider",
        default="auto",
        help="Inference provider. auto/ollama/transformers. Default: auto",
    )
    parser.add_argument("--llm-model-name", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU index for Ollama option main_gpu (ex: 0, 1).",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="GPU count for Ollama option num_gpu.",
    )
    parser.add_argument(
        "--parallel-gpu-ids",
        type=_parse_parallel_gpu_ids,
        default=None,
        help="Comma-separated GPU ids for transformers model sharding (ex: 0,1).",
    )

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--ingredient-csv", type=Path, default=DEFAULT_INGREDIENT_CSV)
    parser.add_argument("--foodtable-csv", type=Path, default=DEFAULT_FOODTABLE_CSV)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default="ingredient_matching_rag")

    return parser


def _resolve_models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Tuple[str, str]:
    llm_model_key = args.llm_model
    embedding_model_key = args.embedding_model

    if args.rag_preset:
        preset = resolve_rag_preset(args.rag_preset)
        if llm_model_key is None:
            llm_model_key = preset.llm_model_key
        if embedding_model_key is None:
            embedding_model_key = preset.embedding_model_key

    if llm_model_key is None or embedding_model_key is None:
        parser.error("Specify --llm-model and --embedding-model, or use --rag-preset.")

    return llm_model_key, embedding_model_key


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    llm_model_key, embedding_model_key = _resolve_models(args, parser)

    runtime = build_rag_runtime(
        llm_model_key=llm_model_key,
        embedding_model_key=embedding_model_key,
        provider=args.provider,
        llm_model_name_override=args.llm_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        seed=args.seed,
        num_predict=args.num_predict,
        gpu_id=args.gpu_id,
        num_gpu=args.num_gpu,
        parallel_gpu_ids=args.parallel_gpu_ids,
    )

    ingredient_path, foodtable_path = _resolve_embedding_paths(
        embedding_root=args.embedding_root,
        embedding_tag=runtime.embedding_output_tag,
    )

    ingredient_df = _load_ingredient_df(args.ingredient_csv)
    foodtable_df = _load_foodtable_df(args.foodtable_csv)
    foodtable_name_to_id = _build_foodtable_name_to_id(foodtable_df)

    ingredient_embedding_map = _load_ingredient_embeddings(ingredient_path)
    food_codes, food_names, food_matrix = _load_foodtable_embeddings(
        foodtable_path,
        foodtable_name_to_id,
    )

    output_dir = args.output_root / runtime.llm_output_tag / runtime.embedding_output_tag
    output_path = output_dir / f"{args.output_prefix}_k{args.k}.json"

    print(f"LLM model key: {runtime.llm_model_key}")
    print(f"LLM model name: {runtime.llm_model_name}")
    print(f"Parallel GPU ids: {args.parallel_gpu_ids}")
    print(f"Embedding model key: {runtime.embedding_model_key}")
    print(f"Embedding model name: {runtime.embedding_model_name}")
    print(f"Embedding ingredient cache: {ingredient_path}")
    print(f"Embedding foodtable cache: {foodtable_path}")
    print(f"Output path: {output_path}")

    run_rag_matching(
        ingredient_df=ingredient_df,
        ingredient_embedding_map=ingredient_embedding_map,
        food_codes=food_codes,
        food_names=food_names,
        food_matrix=food_matrix,
        infer_fn=runtime.infer_fn,
        output_path=output_path,
        k=args.k,
        max_retries=args.max_retries,
        save_every=args.save_every,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
