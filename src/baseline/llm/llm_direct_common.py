import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[3]
FOODTABLE_JP = BASE_DIR / "data" / "data_nutrient_FoodTableJP"
INGREDIENT_LABEL_CSV = BASE_DIR / "data" / "ingredient_label.csv"

DTYPE_DICT = {
    "column_216": "object",
    "column_217": "object",
    "column_218": "object",
    "column_220": "object",
    "column_231": "object",
    "column_235": "object",
    "column_244": "object",
    "column_253": "object",
    "column_291": "object",
}

ModelInferFn = Callable[[str], str]


def create_prompt_all(trg_ingredient_name: str, foodtable: pd.DataFrame) -> str:
    req = ["food_code", "food_name"]
    missing = [c for c in req if c not in foodtable.columns]
    if missing:
        raise ValueError(f"foodtable に必須列がありません: {missing}")

    df = foodtable[req].copy()
    df["food_code"] = df["food_code"].astype(str)
    df["food_name"] = df["food_name"].astype(str).str.replace("\u3000", " ", regex=False)

    lines = [f"{r.food_code}\t{r.food_name}" for r in df.itertuples(index=False)]
    candidates = "\n".join(lines)

    prompt = (
        "あなたは食材名の照合アシスタントです。\n"
        "下の候補一覧（food_code\\tfood_name）から、入力食材名に最も対応する1件を選び、"
        "その food_code のみを JSON で返してください。\n\n"
        "※ 解説や他の文字は一切出力しないこと。id は必ず候補のいずれかである。\n\n"
        f"入力食材名: {trg_ingredient_name}\n\n"
        "【候補一覧】\n"
        "food_code\tfood_name\n"
        "----------------------------------------\n"
        f"{candidates}\n"
        "----------------------------------------\n"
        "【出力】次の1行のみ：\n"
        '{"id":"<food_code>"}\n'
    )
    return prompt


def extract_json(text: str) -> dict:
    codeblock_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
    match = codeblock_pattern.search(text)
    candidate = match.group(1).strip() if match else text.strip()

    brace_pattern = re.compile(r"\{[\s\S]*\}")
    match = brace_pattern.search(candidate)
    if match:
        candidate = match.group(0)

    return json.loads(candidate)


def _load_matching_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ingredient_label = pd.read_csv(INGREDIENT_LABEL_CSV)
    foodtable = pd.read_csv(
        FOODTABLE_JP / "data_preprocessed/food_nutrition_all.csv",
        dtype=DTYPE_DICT,
        low_memory=False,
    )
    required = {"ingredient_name", "foodtable_id"}
    missing = [column for column in required if column not in ingredient_label.columns]
    if missing:
        raise ValueError(f"ingredient_label csv missing columns: {missing}")

    foodmaping_trg = ingredient_label[["ingredient_name", "foodtable_id"]].copy()
    foodmaping_trg = foodmaping_trg.rename(
        columns={"ingredient_name": "ingredient", "foodtable_id": "foodtable_maping"}
    )
    return foodmaping_trg, foodtable


def _save_json(output_path: Path, response_json: List[Dict]) -> None:
    output_path.write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_food_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
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


def _compute_metrics(records: List[Dict], targets: pd.DataFrame) -> Dict[str, float | int]:
    result_by_pair: Dict[tuple[str, str], Dict] = {}
    result_by_ingredient: Dict[str, Dict] = {}
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

    for _, row in targets.iterrows():
        ingredient = str(row["ingredient"])
        gold_id = _normalize_food_id(row["foodtable_maping"])
        if gold_id is None:
            continue

        evaluable_count += 1
        record = result_by_pair.get((ingredient, gold_id))
        if record is None:
            record = result_by_ingredient.get(ingredient)
        predicted_id: object = None
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
        "total_rows": int(len(targets)),
        "evaluable_rows": int(evaluable_count),
        "effective_rows": int(effective_count),
        "exact_match_count": int(exact_match_count),
        "category_match_count": int(category_match_count),
    }


def run_direct_matching(
    output_dir: Path,
    model_infer: ModelInferFn,
    save_every: int = 50,
    max_retries: int = 5,
    limit: Optional[int] = None,
    output_filename: str = "ingredient_matching_direct.json",
    completed_filename: str = "ingredient_matching_direct_comp.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    output_path_comp = output_dir / completed_filename

    foodmaping_trg, foodtable = _load_matching_data()
    if limit is not None and limit > 0:
        foodmaping_trg = foodmaping_trg.head(limit)

    if output_path.exists():
        response_json = json.loads(output_path.read_text(encoding="utf-8"))
        print("すでに存在するファイルを読み込みました。")
    else:
        response_json = []

    done_keys: set[tuple[str, str]] = set()
    for item in response_json:
        ingredient = item.get("ingredient")
        gold_id = _normalize_food_id(item.get("foodtable_maping"))
        if ingredient is None or gold_id is None:
            continue
        done_keys.add((str(ingredient), gold_id))

    total_targets = len(foodmaping_trg)

    retry_limit = max(5, int(max_retries))

    for _, row in tqdm(foodmaping_trg.iterrows(), total=total_targets):
        ing = str(row["ingredient"])
        gold_id = _normalize_food_id(row["foodtable_maping"])
        if gold_id is not None and (ing, gold_id) in done_keys:
            continue

        add_json = {
            "ingredient": ing,
            "foodtable_maping": row["foodtable_maping"],
            "prompt": None,
            "response": None,
            "retray_count": 0,
        }

        try:
            prompt = create_prompt_all(ing, foodtable)
            add_json["prompt"] = prompt
            retray_count = 0
            parsed = None
            last_error = None

            for attempt in range(retry_limit):
                try:
                    content = model_infer(prompt)
                except Exception as error:
                    print(f"[WARN] モデル推論失敗 (attempt {attempt + 1}): {error}")
                    retray_count += 1
                    last_error = f"inference_failed: {error}"
                    continue

                try:
                    parsed = extract_json(content)
                    break
                except Exception as error:
                    print(f"[WARN] JSON parse失敗 (attempt {attempt + 1}): {error}")
                    retray_count += 1
                    last_error = f"json_parse_failed: {error}"
                    continue

            add_json["response"] = parsed
            add_json["retray_count"] = retray_count
            if parsed is None and last_error is not None:
                add_json["error"] = last_error

        except Exception as error:
            print(f"[ERROR] 処理失敗: {error}")
            add_json["error"] = f"record_failed: {error}"
            add_json["response"] = None

        response_json.append(add_json)
        if gold_id is not None:
            done_keys.add((ing, gold_id))

        if len(response_json) % save_every == 0:
            _save_json(output_path, response_json)

    target_keys: set[tuple[str, str]] = set()
    for _, row in foodmaping_trg.iterrows():
        gold_id = _normalize_food_id(row["foodtable_maping"])
        if gold_id is None:
            continue
        target_keys.add((str(row["ingredient"]), gold_id))
    completed_count = sum(
        1 for key in target_keys if key in done_keys
    )

    if completed_count == total_targets:
        print("全ての食材に対する処理が完了しました。")
        _save_json(output_path_comp, response_json)
        saved_path = output_path_comp
    else:
        _save_json(output_path, response_json)
        saved_path = output_path

    metrics = _compute_metrics(response_json, foodmaping_trg)
    print("\n=== LLM direct result ===")
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
    print(f"Saved to: {saved_path}")
