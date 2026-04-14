#!/usr/bin/env python3
"""
Build a lightweight NCBI taxonomy graph for contrastive learning.

Outputs (JSON) under `src/FoodMatchMTL/train_data/`:
  - taxonomy_parent.json
  - taxonomy_names.json
  - food2tax.json
  - taxonomy_depth.json

Example:
  python3 src/FoodMatchMTL/build_taxonomy_graph.py

Debug examples:
  python3 src/FoodMatchMTL/build_taxonomy_graph.py --debug-tax 9606 --debug-max-depth 8
  python3 src/FoodMatchMTL/build_taxonomy_graph.py --debug-food 1001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_BIO_LINK_CSV = Path(
    "data/data_nutrient_FoodTableJP/data_preprocessed/biological_species_link.csv"
)
DEFAULT_NCBI_DIR = Path("data/data_bio_NCBI")
DEFAULT_NAME2ID_JSON = Path("data/foodtable_name2id.json")
DEFAULT_OUTPUT_DIR = Path("src/FoodMatchMTL/train_data")

NODES_DMP = "nodes.dmp"
NAMES_DMP = "names.dmp"
MERGED_DMP = "merged.dmp"
DELNODES_DMP = "delnodes.dmp"

TAXONOMY_PARENT_JSON = "taxonomy_parent.json"
TAXONOMY_NAMES_JSON = "taxonomy_names.json"
FOOD2TAX_JSON = "food2tax.json"
TAXONOMY_DEPTH_JSON = "taxonomy_depth.json"

JP_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺ一-龯々〆ヵヶ]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
BRACKET_NOTE_RE = re.compile(r"\[[^\]]*\]")
SP_SPP_TOKEN_RE = re.compile(r"\b(spp|sp)\.?\b", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{1,}")

IGNORE_TOKENS = {
    "hybrid",
    "of",
    "x",
    "sp",
    "spp",
    "cf",
    "aff",
    "nr",
    "group",
    "complex",
    "subsp",
    "ssp",
    "var",
    "forma",
    "f",
}

RANK_SPECIES = "species"
RANK_GENUS = "genus"
RANK_FAMILY = "family"

RESOLUTION_SCORE = {
    "exact": 3,
    "genus_species": 2,
    "genus": 1,
}


@dataclass
class FoodRecord:
    row_index: int
    food_id: str | None
    food_name: str
    scientific_raw: str
    sci_candidates: list[str]
    genus_species_candidates: list[str]
    genus_candidates: list[str]


class TaxonomyGraphBuilder:
    def __init__(
        self,
        bio_link_csv: Path,
        ncbi_dir: Path,
        foodtable_name2id_json: Path,
        output_dir: Path,
    ) -> None:
        self.bio_link_csv = bio_link_csv
        self.ncbi_dir = ncbi_dir
        self.foodtable_name2id_json = foodtable_name2id_json
        self.output_dir = output_dir

        self.parent_by_taxid: dict[int, int] = {}
        self.rank_by_taxid: dict[int, str] = {}
        self.depth_by_taxid: dict[int, int] = {}
        self.merged_map: dict[int, int] = {}
        self.delnodes: set[int] = set()
        self._resolved_taxid_cache: dict[int, int] = {}

        self.names_taxids: set[int] = set()
        self.scientific_cache: dict[int, str] = {}

        self.food2tax: dict[str, int] = {}
        self.food_id2name: dict[str, str] = {}

        self.exact_index: dict[str, set[int]] = defaultdict(set)
        self.genus_species_index: dict[str, set[int]] = defaultdict(set)
        self.genus_index: dict[str, set[int]] = defaultdict(set)
        self.species_ancestor_cache: dict[int, int | None] = {}
        self.genus_ancestor_cache: dict[int, int | None] = {}
        self.family_ancestor_cache: dict[int, int | None] = {}

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        name2id = self._load_foodtable_name2id(self.foodtable_name2id_json)
        food_records, exact_q, gs_q, genus_q = self._load_food_records(self.bio_link_csv, name2id)

        self._load_merged(self.ncbi_dir / MERGED_DMP)
        self._load_delnodes(self.ncbi_dir / DELNODES_DMP)
        self._load_nodes(self.ncbi_dir / NODES_DMP)

        self._write_int_map_json(
            self.output_dir / TAXONOMY_PARENT_JSON,
            self.parent_by_taxid,
        )

        missing_scientific_count = self._build_taxonomy_names_and_indices(
            names_dmp_path=self.ncbi_dir / NAMES_DMP,
            output_path=self.output_dir / TAXONOMY_NAMES_JSON,
            exact_queries=exact_q,
            genus_species_queries=gs_q,
            genus_queries=genus_q,
        )

        (
            self.food2tax,
            unresolved_by_food_id,
            unresolved_no_food_id,
            linked_tax_count,
        ) = self._resolve_food_to_tax(food_records)
        self._write_food2tax_json(self.output_dir / FOOD2TAX_JSON, self.food2tax)

        self.depth_by_taxid = self._compute_depths(self.parent_by_taxid)
        self._write_int_map_json(
            self.output_dir / TAXONOMY_DEPTH_JSON,
            self.depth_by_taxid,
        )

        self._run_quality_checks(self.food2tax)

        unresolved_food_ids = set(unresolved_by_food_id.keys())
        unresolved_food_count = len(unresolved_food_ids) + len(unresolved_no_food_id)
        unresolved_name_counter = Counter()
        for fid in unresolved_food_ids:
            unresolved_name_counter[self.food_id2name.get(fid, fid)] += 1
        for rec, _reason in unresolved_no_food_id:
            unresolved_name_counter[rec.food_name] += 1

        for food_id, reasons in sorted(unresolved_by_food_id.items()):
            reason_summary = ", ".join(
                f"{reason}:{cnt}" for reason, cnt in Counter(reasons).most_common(3)
            )
            print(
                f"WARN unresolved food_id={food_id} food_name={self.food_id2name.get(food_id, '')} "
                f"reasons={reason_summary}"
            )
        for rec, reason in unresolved_no_food_id:
            print(
                f"WARN unresolved row={rec.row_index} food_name={rec.food_name} "
                f"reason={reason}"
            )

        print(f"総 taxonomy ノード数: {len(self.parent_by_taxid)}")
        print(f"food とリンクされた tax_id 数: {linked_tax_count}")
        print(f"food2tax に成功した食品数: {len(self.food2tax)}")
        print(f"未解決食品数: {unresolved_food_count}")
        print("未解決食品（上位10件）:")
        for food_name, count in unresolved_name_counter.most_common(10):
            print(f"  - {food_name}: {count}")
        if missing_scientific_count:
            print(
                f"WARN scientific name 欠損 tax_id 数 (taxonomy_names 補完): "
                f"{missing_scientific_count}"
            )

    @staticmethod
    def _parse_dmp_line(line: str) -> list[str]:
        line = line.rstrip("\n")
        if line.endswith("\t|"):
            line = line[:-2]
        return [p.strip() for p in line.split("\t|\t")]

    def _resolve_tax_id(self, tax_id: int) -> int:
        cached = self._resolved_taxid_cache.get(tax_id)
        if cached is not None:
            return cached

        path: list[int] = []
        seen: set[int] = set()
        cur = tax_id
        while cur in self.merged_map and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = self.merged_map[cur]

        for x in path:
            self._resolved_taxid_cache[x] = cur
        self._resolved_taxid_cache[tax_id] = cur
        return cur

    def _load_merged(self, merged_path: Path) -> None:
        if not merged_path.exists():
            return
        with merged_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self._parse_dmp_line(line)
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    continue
                old_id = int(parts[0])
                new_id = int(parts[1])
                self.merged_map[old_id] = new_id

    def _load_delnodes(self, delnodes_path: Path) -> None:
        if not delnodes_path.exists():
            return
        with delnodes_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self._parse_dmp_line(line)
                if not parts or not parts[0]:
                    continue
                self.delnodes.add(int(parts[0]))

    def _load_nodes(self, nodes_path: Path) -> None:
        with nodes_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self._parse_dmp_line(line)
                if len(parts) < 3:
                    continue
                raw_tax_id = int(parts[0])
                raw_parent_id = int(parts[1])
                rank = sys.intern(parts[2])

                tax_id = self._resolve_tax_id(raw_tax_id)
                parent_id = self._resolve_tax_id(raw_parent_id)

                if tax_id in self.delnodes:
                    continue
                if parent_id in self.delnodes and parent_id != tax_id:
                    parent_id = tax_id

                self.parent_by_taxid[tax_id] = parent_id
                self.rank_by_taxid[tax_id] = rank

    @staticmethod
    def _normalize_scientific_name(text: str) -> str:
        s = str(text).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        s = s.replace("\u3000", " ")
        s = BRACKET_NOTE_RE.sub(" ", s)
        s = s.replace("*", " ")

        def _replace_sp_spp(m: re.Match[str]) -> str:
            token = m.group(1).lower()
            return "spp" if token == "spp" else "sp"

        s = SP_SPP_TOKEN_RE.sub(_replace_sp_spp, s)
        s = SPACE_RE.sub(" ", s).strip()
        s = s.rstrip(". ").strip()
        s = SPACE_RE.sub(" ", s).strip()
        return s

    @staticmethod
    def _split_scientific_candidates(normalized_name: str) -> list[str]:
        if not normalized_name:
            return []
        parts = re.split(r"[;,/]", normalized_name)
        candidates: list[str] = []
        seen: set[str] = set()
        for part in parts:
            p = SPACE_RE.sub(" ", part).strip().rstrip(".").strip()
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                candidates.append(p)
        if not candidates:
            candidates.append(normalized_name)
        return candidates

    @staticmethod
    def _extract_genus(name: str) -> str | None:
        for raw_token in re.split(r"\s+", name):
            token = raw_token.strip("()[]{}.*").rstrip(".")
            low = token.lower()
            if low in IGNORE_TOKENS:
                continue
            if LATIN_TOKEN_RE.fullmatch(token):
                return low
        return None

    @staticmethod
    def _extract_genus_species(name: str) -> str | None:
        found: list[str] = []
        for raw_token in re.split(r"\s+", name):
            token = raw_token.strip("()[]{}.*").rstrip(".")
            low = token.lower()
            if low in IGNORE_TOKENS:
                continue
            if LATIN_TOKEN_RE.fullmatch(token):
                found.append(low)
                if len(found) >= 2:
                    return f"{found[0]} {found[1]}"
        return None

    @staticmethod
    def _has_japanese(text: str) -> bool:
        return bool(JP_CHAR_RE.search(text))

    @staticmethod
    def _is_english_like(text: str) -> bool:
        return text.isascii() and bool(ASCII_LETTER_RE.search(text))

    @staticmethod
    def _find_column(columns: Iterable[str], exact: list[str], contains: list[str]) -> str:
        col_list = list(columns)
        for c in exact:
            if c in col_list:
                return c
        for c in col_list:
            for token in contains:
                if token in c:
                    return c
        raise ValueError(
            f"Required column not found. exact={exact}, contains={contains}, "
            f"available={col_list}"
        )

    @staticmethod
    def _load_foodtable_name2id(path: Path) -> dict[str, int]:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: dict[str, int] = {}
        for k, v in raw.items():
            out[str(k).strip()] = int(v)
        return out

    def _load_food_records(
        self,
        csv_path: Path,
        foodtable_name2id: dict[str, int],
    ) -> tuple[list[FoodRecord], set[str], set[str], set[str]]:
        df = pd.read_csv(csv_path)

        food_id_col = self._find_column(df.columns, exact=["食品番号"], contains=["食品番号", "food_id", "id"])
        food_name_col = self._find_column(
            df.columns,
            exact=["食品名（ ）内は生物種名"],
            contains=["食品名", "food_name"],
        )
        scientific_col = self._find_column(df.columns, exact=["学名"], contains=["学名", "scientific"])

        records: list[FoodRecord] = []
        exact_queries: set[str] = set()
        gs_queries: set[str] = set()
        genus_queries: set[str] = set()

        for idx, row in df.iterrows():
            food_name = str(row.get(food_name_col, "")).strip()

            food_id: str | None = None
            raw_id = row.get(food_id_col)
            if pd.notna(raw_id):
                try:
                    food_id = str(int(raw_id))
                except (ValueError, TypeError):
                    food_id = None
            if food_id is None and food_name:
                mapped_id = foodtable_name2id.get(food_name)
                if mapped_id is not None:
                    food_id = str(int(mapped_id))

            scientific_raw = str(row.get(scientific_col, ""))
            normalized = self._normalize_scientific_name(scientific_raw)
            sci_candidates = self._split_scientific_candidates(normalized)
            gs_candidates: list[str] = []
            genus_candidates: list[str] = []

            for cand in sci_candidates:
                exact_queries.add(cand)
                gs = self._extract_genus_species(cand)
                if gs and gs not in gs_candidates:
                    gs_candidates.append(gs)
                    gs_queries.add(gs)
                genus = self._extract_genus(cand)
                if genus and genus not in genus_candidates:
                    genus_candidates.append(genus)
                    genus_queries.add(genus)

            rec = FoodRecord(
                row_index=int(idx),
                food_id=food_id,
                food_name=food_name,
                scientific_raw=scientific_raw,
                sci_candidates=sci_candidates,
                genus_species_candidates=gs_candidates,
                genus_candidates=genus_candidates,
            )
            records.append(rec)
            if food_id is not None and food_name:
                self.food_id2name.setdefault(food_id, food_name)

        return records, exact_queries, gs_queries, genus_queries

    def _get_ancestor_by_rank(
        self,
        tax_id: int,
        target_rank: str,
        cache: dict[int, int | None],
    ) -> int | None:
        cached = cache.get(tax_id, None)
        if tax_id in cache:
            return cached

        cur = tax_id
        seen: set[int] = set()
        while True:
            if cur in seen:
                cache[tax_id] = None
                return None
            seen.add(cur)

            rank = self.rank_by_taxid.get(cur)
            if rank == target_rank:
                cache[tax_id] = cur
                return cur

            parent = self.parent_by_taxid.get(cur)
            if parent is None or parent == cur:
                cache[tax_id] = None
                return None
            cur = parent

    def _build_taxonomy_names_and_indices(
        self,
        names_dmp_path: Path,
        output_path: Path,
        exact_queries: set[str],
        genus_species_queries: set[str],
        genus_queries: set[str],
    ) -> int:
        missing_scientific_count = 0

        def new_bucket() -> dict[str, object]:
            return {
                "scientific": None,
                "english": [],
                "japanese": [],
                "synonyms": [],
                "_seen_en": set(),
                "_seen_ja": set(),
                "_seen_syn": set(),
            }

        def add_non_scientific(bucket: dict[str, object], name_txt: str) -> None:
            if self._has_japanese(name_txt):
                if name_txt not in bucket["_seen_ja"]:
                    bucket["japanese"].append(name_txt)
                    bucket["_seen_ja"].add(name_txt)
                return
            if self._is_english_like(name_txt):
                if name_txt not in bucket["_seen_en"]:
                    bucket["english"].append(name_txt)
                    bucket["_seen_en"].add(name_txt)
                return
            if name_txt not in bucket["_seen_syn"]:
                bucket["synonyms"].append(name_txt)
                bucket["_seen_syn"].add(name_txt)

        def finalize_bucket(tax_id: int, bucket: dict[str, object]) -> dict[str, object]:
            scientific = bucket["scientific"]
            if scientific is None:
                if bucket["english"]:
                    scientific = bucket["english"][0]
                elif bucket["japanese"]:
                    scientific = bucket["japanese"][0]
                elif bucket["synonyms"]:
                    scientific = bucket["synonyms"][0]
                else:
                    scientific = f"taxid_{tax_id}"
                nonlocal missing_scientific_count
                missing_scientific_count += 1

            obj = {
                "scientific": scientific,
                "english": bucket["english"],
                "japanese": bucket["japanese"],
                "synonyms": bucket["synonyms"],
            }
            return obj

        current_tax_id: int | None = None
        bucket: dict[str, object] | None = None
        first_entry = True

        with names_dmp_path.open("r", encoding="utf-8", errors="ignore") as in_f, output_path.open(
            "w", encoding="utf-8"
        ) as out_f:
            out_f.write("{\n")
            for line in in_f:
                parts = self._parse_dmp_line(line)
                if len(parts) < 4:
                    continue
                raw_tax_id = int(parts[0])
                name_txt = parts[1]
                name_class = parts[3]
                if not name_txt:
                    continue

                canonical_tax_id = self._resolve_tax_id(raw_tax_id)
                if canonical_tax_id in self.delnodes:
                    continue

                # Build food matching indices from scientific names (merged IDs included).
                if name_class == "scientific name":
                    normalized = self._normalize_scientific_name(name_txt)
                    if normalized:
                        if normalized in exact_queries:
                            self.exact_index[normalized].add(canonical_tax_id)

                        gs = self._extract_genus_species(normalized)
                        if gs and gs in genus_species_queries:
                            species_tid = self._get_ancestor_by_rank(
                                canonical_tax_id, RANK_SPECIES, self.species_ancestor_cache
                            )
                            if species_tid is None and self.rank_by_taxid.get(canonical_tax_id) == RANK_SPECIES:
                                species_tid = canonical_tax_id
                            if species_tid is not None:
                                self.genus_species_index[gs].add(species_tid)

                        genus = self._extract_genus(normalized)
                        if genus and genus in genus_queries:
                            genus_tid = self._get_ancestor_by_rank(
                                canonical_tax_id, RANK_GENUS, self.genus_ancestor_cache
                            )
                            if genus_tid is None and self.rank_by_taxid.get(canonical_tax_id) == RANK_GENUS:
                                genus_tid = canonical_tax_id
                            if genus_tid is not None:
                                self.genus_index[genus].add(genus_tid)

                # taxonomy_names.json: skip merged old IDs to keep one canonical entry.
                if raw_tax_id in self.merged_map or raw_tax_id in self.delnodes:
                    continue

                tax_id = raw_tax_id
                if current_tax_id is None:
                    current_tax_id = tax_id
                    bucket = new_bucket()
                    self.names_taxids.add(tax_id)

                if tax_id != current_tax_id:
                    assert bucket is not None
                    obj = finalize_bucket(current_tax_id, bucket)
                    if not first_entry:
                        out_f.write(",\n")
                    out_f.write(json.dumps(str(current_tax_id), ensure_ascii=False))
                    out_f.write(": ")
                    out_f.write(json.dumps(obj, ensure_ascii=False))
                    first_entry = False

                    current_tax_id = tax_id
                    bucket = new_bucket()
                    self.names_taxids.add(tax_id)

                assert bucket is not None
                if name_class == "scientific name":
                    if bucket["scientific"] is None:
                        bucket["scientific"] = name_txt
                else:
                    add_non_scientific(bucket, name_txt)

            if current_tax_id is not None and bucket is not None:
                obj = finalize_bucket(current_tax_id, bucket)
                if not first_entry:
                    out_f.write(",\n")
                out_f.write(json.dumps(str(current_tax_id), ensure_ascii=False))
                out_f.write(": ")
                out_f.write(json.dumps(obj, ensure_ascii=False))

            out_f.write("\n}\n")

        return missing_scientific_count

    def _resolve_food_to_tax(
        self, food_records: list[FoodRecord]
    ) -> tuple[dict[str, int], dict[str, list[str]], list[tuple[FoodRecord, str]], int]:
        best_by_food_id: dict[str, tuple[int, int, str]] = {}
        unresolved_by_food_id: dict[str, list[str]] = defaultdict(list)
        unresolved_no_food_id: list[tuple[FoodRecord, str]] = []

        def unique_or_none(candidates: set[int]) -> int | None:
            if len(candidates) == 1:
                return next(iter(candidates))
            return None

        for rec in food_records:
            if not rec.food_id:
                unresolved_no_food_id.append((rec, "food_id_missing"))
                continue

            resolved_tax_id: int | None = None
            strategy = ""

            exact_candidates: set[int] = set()
            for sci in rec.sci_candidates:
                exact_candidates.update(self.exact_index.get(sci, set()))
            resolved_tax_id = unique_or_none(exact_candidates)
            if resolved_tax_id is not None:
                strategy = "exact"
            else:
                gs_candidates: set[int] = set()
                for gs in rec.genus_species_candidates:
                    gs_candidates.update(self.genus_species_index.get(gs, set()))
                resolved_tax_id = unique_or_none(gs_candidates)
                if resolved_tax_id is not None:
                    strategy = "genus_species"
                else:
                    genus_candidates: set[int] = set()
                    for genus in rec.genus_candidates:
                        genus_candidates.update(self.genus_index.get(genus, set()))
                    resolved_tax_id = unique_or_none(genus_candidates)
                    if resolved_tax_id is not None:
                        strategy = "genus"

            if resolved_tax_id is None:
                if exact_candidates:
                    unresolved_by_food_id[rec.food_id].append(
                        f"exact_ambiguous:{len(exact_candidates)}"
                    )
                elif rec.genus_species_candidates:
                    gs_candidates = set()
                    for gs in rec.genus_species_candidates:
                        gs_candidates.update(self.genus_species_index.get(gs, set()))
                    if gs_candidates:
                        unresolved_by_food_id[rec.food_id].append(
                            f"genus_species_ambiguous:{len(gs_candidates)}"
                        )
                    else:
                        unresolved_by_food_id[rec.food_id].append("genus_species_not_found")
                elif rec.genus_candidates:
                    genus_candidates = set()
                    for genus in rec.genus_candidates:
                        genus_candidates.update(self.genus_index.get(genus, set()))
                    if genus_candidates:
                        unresolved_by_food_id[rec.food_id].append(
                            f"genus_ambiguous:{len(genus_candidates)}"
                        )
                    else:
                        unresolved_by_food_id[rec.food_id].append("genus_not_found")
                else:
                    unresolved_by_food_id[rec.food_id].append("scientific_name_missing")
                continue

            score = RESOLUTION_SCORE[strategy]
            prev = best_by_food_id.get(rec.food_id)
            if prev is None:
                best_by_food_id[rec.food_id] = (resolved_tax_id, score, strategy)
                continue

            prev_tax, prev_score, prev_strategy = prev
            if prev_tax == resolved_tax_id:
                continue
            if score > prev_score:
                print(
                    f"WARN food_id={rec.food_id} resolved tax conflict: "
                    f"{prev_tax}({prev_strategy}) -> {resolved_tax_id}({strategy})"
                )
                best_by_food_id[rec.food_id] = (resolved_tax_id, score, strategy)
            else:
                unresolved_by_food_id[rec.food_id].append(
                    f"conflict_keep:{prev_tax}({prev_strategy})_drop:{resolved_tax_id}({strategy})"
                )

        food2tax = {fid: val[0] for fid, val in best_by_food_id.items()}
        unresolved_ids = set(unresolved_by_food_id.keys()) - set(food2tax.keys())
        for fid in list(unresolved_by_food_id.keys()):
            if fid not in unresolved_ids:
                unresolved_by_food_id.pop(fid, None)

        linked_tax_count = len({tax for tax in food2tax.values()})
        return food2tax, unresolved_by_food_id, unresolved_no_food_id, linked_tax_count

    def _compute_depths(self, parent_map: dict[int, int]) -> dict[int, int]:
        depth: dict[int, int] = {}
        for tax_id in parent_map.keys():
            if tax_id in depth:
                continue

            stack: list[int] = []
            seen_pos: dict[int, int] = {}
            cur = tax_id

            while cur not in depth:
                if cur in seen_pos:
                    cycle_start = seen_pos[cur]
                    for cyc in stack[cycle_start:]:
                        depth[cyc] = 0
                    break

                seen_pos[cur] = len(stack)
                stack.append(cur)
                parent = parent_map.get(cur)
                if parent is None or parent == cur:
                    depth[cur] = 0
                    break
                cur = parent

            for node in reversed(stack):
                if node in depth:
                    continue
                parent = parent_map.get(node)
                if parent is None or parent == node:
                    depth[node] = 0
                else:
                    depth[node] = depth.get(parent, 0) + 1
        return depth

    def _is_ancestor(self, ancestor: int, tax_id: int, max_steps: int = 512) -> bool:
        cur = tax_id
        for _ in range(max_steps):
            if cur == ancestor:
                return True
            parent = self.parent_by_taxid.get(cur)
            if parent is None or parent == cur:
                return False
            cur = parent
        return False

    def _run_quality_checks(self, food2tax: dict[str, int]) -> None:
        missing_in_parent = sorted({tid for tid in food2tax.values() if tid not in self.parent_by_taxid})
        missing_in_names = sorted({tid for tid in food2tax.values() if tid not in self.names_taxids})
        if missing_in_parent:
            raise ValueError(
                f"Quality check failed: food2tax tax_id missing in taxonomy_parent: "
                f"{missing_in_parent[:10]}"
            )
        if missing_in_names:
            raise ValueError(
                f"Quality check failed: food2tax tax_id missing in taxonomy_names: "
                f"{missing_in_names[:10]}"
            )

        mapped_tax_ids = sorted(set(food2tax.values()))
        genus_groups: dict[int, list[int]] = defaultdict(list)
        family_groups: dict[int, list[int]] = defaultdict(list)
        for tid in mapped_tax_ids:
            genus_tid = self._get_ancestor_by_rank(tid, RANK_GENUS, self.genus_ancestor_cache)
            family_tid = self._get_ancestor_by_rank(tid, RANK_FAMILY, self.family_ancestor_cache)
            if genus_tid is not None:
                genus_groups[genus_tid].append(tid)
            if family_tid is not None:
                family_groups[family_tid].append(tid)

        check_failures: list[str] = []
        for group_name, groups in [("genus", genus_groups), ("family", family_groups)]:
            tested = 0
            for ancestor_tid, members in groups.items():
                uniq = sorted(set(members))
                if len(uniq) < 2:
                    continue
                tested += 1
                a = uniq[0]
                b = uniq[1]
                if not self._is_ancestor(ancestor_tid, a) or not self._is_ancestor(ancestor_tid, b):
                    check_failures.append(
                        f"{group_name} check failed: ancestor={ancestor_tid}, members={a},{b}"
                    )
                    continue
                if self.depth_by_taxid.get(ancestor_tid, 0) > self.depth_by_taxid.get(a, 0):
                    check_failures.append(
                        f"{group_name} depth inconsistent: ancestor={ancestor_tid}, member={a}"
                    )
                if self.depth_by_taxid.get(ancestor_tid, 0) > self.depth_by_taxid.get(b, 0):
                    check_failures.append(
                        f"{group_name} depth inconsistent: ancestor={ancestor_tid}, member={b}"
                    )
                if tested >= 200:
                    break

        if check_failures:
            raise ValueError(f"Quality checks failed: {check_failures[:5]}")

    @staticmethod
    def _write_int_map_json(path: Path, data: dict[int, int]) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("{")
            first = True
            for key, value in data.items():
                if not first:
                    f.write(",")
                f.write(json.dumps(str(key), ensure_ascii=False))
                f.write(":")
                f.write(str(int(value)))
                first = False
            f.write("}\n")

    @staticmethod
    def _write_food2tax_json(path: Path, food2tax: dict[str, int]) -> None:
        serialized = {str(k): int(v) for k, v in sorted(food2tax.items(), key=lambda x: int(x[0]))}
        with path.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    def _load_scientific_names_for_taxids(self, tax_ids: set[int]) -> None:
        missing = {tid for tid in tax_ids if tid not in self.scientific_cache}
        if not missing:
            return
        names_path = self.ncbi_dir / NAMES_DMP
        with names_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self._parse_dmp_line(line)
                if len(parts) < 4:
                    continue
                if parts[3] != "scientific name":
                    continue
                raw_tid = int(parts[0])
                tid = self._resolve_tax_id(raw_tid)
                if tid in missing:
                    self.scientific_cache[tid] = parts[1]
                    missing.remove(tid)
                    if not missing:
                        break
        for tid in missing:
            self.scientific_cache[tid] = "<scientific name not found>"

    def debug_print_tax(self, tax_id: int, max_depth: int = 5) -> None:
        tid = self._resolve_tax_id(int(tax_id))
        chain: list[int] = []
        cur = tid
        for _ in range(max_depth + 1):
            chain.append(cur)
            parent = self.parent_by_taxid.get(cur)
            if parent is None or parent == cur:
                break
            cur = parent

        self._load_scientific_names_for_taxids(set(chain))
        print(f"[debug_print_tax] tax_id={tax_id} normalized_tax_id={tid}")
        for depth, node in enumerate(chain):
            parent = self.parent_by_taxid.get(node)
            rank = self.rank_by_taxid.get(node, "<unknown>")
            sci = self.scientific_cache.get(node, "<scientific name not found>")
            print(
                f"  depth+{depth}: tax_id={node}, rank={rank}, "
                f"scientific={sci}, parent={parent}"
            )

    def debug_print_food(self, food_id: str, max_depth: int = 5) -> None:
        fid = str(food_id)
        tax_id = self.food2tax.get(fid)
        if tax_id is None:
            print(f"[debug_print_food] food_id={fid} is not resolved in food2tax")
            return
        food_name = self.food_id2name.get(fid, "<unknown>")
        print(f"[debug_print_food] food_id={fid}, food_name={food_name}, tax_id={tax_id}")
        self.debug_print_tax(tax_id, max_depth=max_depth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight NCBI taxonomy graph JSONs.")
    parser.add_argument("--bio-link-csv", type=Path, default=DEFAULT_BIO_LINK_CSV)
    parser.add_argument("--ncbi-dir", type=Path, default=DEFAULT_NCBI_DIR)
    parser.add_argument("--foodtable-name2id", type=Path, default=DEFAULT_NAME2ID_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--debug-tax", action="append", type=int, default=[])
    parser.add_argument("--debug-food", action="append", type=str, default=[])
    parser.add_argument("--debug-max-depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = TaxonomyGraphBuilder(
        bio_link_csv=args.bio_link_csv,
        ncbi_dir=args.ncbi_dir,
        foodtable_name2id_json=args.foodtable_name2id,
        output_dir=args.output_dir,
    )
    builder.run()

    for tax_id in args.debug_tax:
        builder.debug_print_tax(tax_id, max_depth=args.debug_max_depth)
    for food_id in args.debug_food:
        builder.debug_print_food(food_id, max_depth=args.debug_max_depth)


if __name__ == "__main__":
    main()
