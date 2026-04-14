#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("src/FoodMatchMTL/train_data/ingredient_alias_dict.json")
DEFAULT_OUTPUT = Path("src/FoodMatchMTL/train_data/ingredient_alias_dict.json")
DEFAULT_PAIR_CSV_OUTPUT = Path("src/FoodMatchMTL/train_data/ingredient_alias_pairs.csv")
DEFAULT_FOODTABLE_NAME2ID_PATH = Path("data/foodtable_name2id.json")
REMOVE_KEY = "成分識別子"
JAPANESE_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺ一-龯々〆ヵヶ]")


def load_alias_dict(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Input JSON must be an object: {path}")

    normalized: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[key] = [str(v) for v in value]
        else:
            raise ValueError(f"Value for key '{key}' must be a list")
    return normalized


def load_foodtable_name2id(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"foodtable_name2id JSON must be an object: {path}")

    normalized: dict[str, int] = {}
    for raw_key, raw_value in data.items():
        key = str(raw_key).strip()
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"ID for '{raw_key}' is not an integer: {raw_value}") from e

        if key in normalized and normalized[key] != value:
            raise ValueError(
                f"Conflicting IDs for normalized key '{key}': {normalized[key]} vs {value}"
            )
        normalized[key] = value

    return normalized


def extract_category_id(food_id: int) -> str:
    return str(food_id).zfill(5)[:2]


def build_category_id_by_foodtable_name(
    food_table_names: list[str], foodtable_name2id: dict[str, int]
) -> dict[str, str]:
    category_by_name: dict[str, str] = {}
    missing = []

    for food_table_name in food_table_names:
        normalized_name = food_table_name.strip()
        food_id = foodtable_name2id.get(normalized_name)
        if food_id is None:
            missing.append(food_table_name)
            continue
        category_by_name[food_table_name] = extract_category_id(food_id)

    if missing:
        missing_preview = ", ".join(missing[:10])
        raise ValueError(
            f"Missing food_table_name in foodtable_name2id.json: {len(missing)} entries "
            f"(e.g. {missing_preview})"
        )

    return category_by_name


def has_japanese_text(text: str) -> bool:
    return bool(JAPANESE_CHAR_RE.search(text))


def clean_alias_values(alias_dict: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
    cleaned: dict[str, list[str]] = {}
    removed_non_japanese = 0

    for key, aliases in alias_dict.items():
        if key == REMOVE_KEY:
            continue
        kept_aliases = []
        for alias in aliases:
            if has_japanese_text(alias):
                kept_aliases.append(alias)
            else:
                removed_non_japanese += 1
        cleaned[key] = kept_aliases

    return cleaned, removed_non_japanese


def remove_cross_list_duplicates(alias_dict: dict[str, list[str]]) -> tuple[dict[str, list[str]], int, int]:
    filtered, removed_non_japanese = clean_alias_values(alias_dict)

    alias_count = Counter()
    for aliases in filtered.values():
        alias_count.update(set(aliases))

    result: dict[str, list[str]] = {}
    removed_cross_duplicates = 0
    for key, aliases in filtered.items():
        unique_in_list = []
        seen = set()
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            if alias_count[alias] == 1:
                unique_in_list.append(alias)
            else:
                removed_cross_duplicates += 1
        result[key] = unique_in_list
    return result, removed_non_japanese, removed_cross_duplicates


def save_alias_dict(path: Path, data: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_pair_csv(path: Path, data: dict[str, list[str]], category_by_name: dict[str, str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["food_name", "food_table_name", "food_table_category_id"])
        for food_table_name, aliases in data.items():
            category_id = category_by_name[food_table_name]
            for food_name in aliases:
                writer.writerow([food_name, food_table_name, category_id])
                row_count += 1
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean ingredient alias dictionary by removing '成分識別子' "
            "and aliases shared across multiple lists. "
            "Also remove aliases that have no Japanese text."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=DEFAULT_PAIR_CSV_OUTPUT,
        help=f"Output pair CSV path (default: {DEFAULT_PAIR_CSV_OUTPUT})",
    )
    parser.add_argument(
        "--foodtable-name2id",
        type=Path,
        default=DEFAULT_FOODTABLE_NAME2ID_PATH,
        help=f"foodtable_name2id JSON path (default: {DEFAULT_FOODTABLE_NAME2ID_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alias_dict = load_alias_dict(args.input)
    cleaned, removed_non_japanese, removed_cross_duplicates = remove_cross_list_duplicates(alias_dict)
    foodtable_name2id = load_foodtable_name2id(args.foodtable_name2id)
    category_by_name = build_category_id_by_foodtable_name(list(cleaned.keys()), foodtable_name2id)
    save_alias_dict(args.output, cleaned)
    pair_rows = save_pair_csv(args.pair_csv, cleaned, category_by_name)

    input_keys = len(alias_dict)
    output_keys = len(cleaned)
    removed_key = REMOVE_KEY in alias_dict
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Pair CSV: {args.pair_csv}")
    print(f"foodtable_name2id: {args.foodtable_name2id}")
    print(f"Keys: {input_keys} -> {output_keys} (removed '{REMOVE_KEY}': {removed_key})")
    print(f"Removed non-Japanese aliases: {removed_non_japanese}")
    print(f"Removed cross-list aliases: {removed_cross_duplicates}")
    print(f"Pair rows: {pair_rows}")


if __name__ == "__main__":
    main()
