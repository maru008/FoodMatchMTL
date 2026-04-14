#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "transformers が必要です。指定の singularity 環境で実行してください: "
        "`singularity exec --nv --bind /workspace env python3 -m src.ours_v2.train ...`"
    ) from exc


ENCODER_ALIASES = {
    "sentence_bert": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "qwen3_8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3:8b": "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "multilingual_e5_large": "intfloat/multilingual-e5-large",
    "e5_large": "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large": "intfloat/multilingual-e5-large",
    "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "mdeberta_v3_base": "microsoft/mdeberta-v3-base",
    "mdeberta-v3-base": "microsoft/mdeberta-v3-base",
    "microsoft/mdeberta-v3-base": "microsoft/mdeberta-v3-base",
}

ENCODER_TOKENIZER_KWARGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-Embedding-8B": {"padding_side": "left"},
}

ENCODER_TRUST_REMOTE_CODE: dict[str, bool] = {
    "Qwen/Qwen3-Embedding-8B": True,
}

E5_ENCODER_NAMES = {"intfloat/multilingual-e5-large"}

DEFAULT_QWEN_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MODE_TO_TASKS = {
    "only_id": {"id"},
    "id_cat": {"id", "cat"},
    "id_tax": {"id", "tax"},
    "multitask": {"id", "cat", "tax"},
}

TASK1_MODE_SWEEP = ["only_id", "id_cat", "id_tax", "multitask"]

MODE_TO_TASK_PATTERN = {
    "only_id": "task1",
    "id_cat": "task1+2",
    "id_tax": "task1+3",
    "multitask": "task1+2+3",
}

DEFAULT_TAXONOMY_RANK_FILTER = {
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def str2bool_or_auto(value: str | bool) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    return str2bool(normalized)


def normalize_food_id(value: Any) -> str | None:
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


def normalize_category_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "00"
    try:
        number = int(float(text))
        return f"{number:02d}"
    except ValueError:
        digits = re.sub(r"\D", "", text)
        if digits:
            return digits.zfill(2)[-2:]
        return text[:2].zfill(2)


def resolve_encoder_name(alias_or_name: str) -> str:
    return ENCODER_ALIASES.get(alias_or_name, alias_or_name)


def resolve_encoder_tokenizer_kwargs(alias_or_name: str) -> dict[str, Any]:
    encoder_name = resolve_encoder_name(alias_or_name)
    return dict(ENCODER_TOKENIZER_KWARGS.get(encoder_name, {}))


def resolve_encoder_trust_remote_code(alias_or_name: str) -> bool:
    encoder_name = resolve_encoder_name(alias_or_name)
    return bool(ENCODER_TRUST_REMOTE_CODE.get(encoder_name, False))


def resolve_use_e5_prefix(use_e5_prefix_arg: bool | str, encoder_name: str) -> bool:
    if isinstance(use_e5_prefix_arg, bool):
        return use_e5_prefix_arg
    if str(use_e5_prefix_arg).strip().lower() == "auto":
        return encoder_name in E5_ENCODER_NAMES
    return str2bool(str(use_e5_prefix_arg))


def resolve_encoder_torch_dtype(dtype_arg: str, encoder_name: str, device: torch.device) -> torch.dtype | None:
    normalized = dtype_arg.strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized != "auto":
        raise ValueError(f"Unsupported encoder_torch_dtype: {dtype_arg}")

    if "qwen3-embedding-8b" in encoder_name.lower() and device.type == "cuda":
        bf16_checker = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_supported = bool(bf16_checker()) if callable(bf16_checker) else False
        return torch.bfloat16 if bf16_supported else torch.float16
    return None


def parse_csv_items(value: str) -> list[str]:
    out: list[str] = []
    for token in value.split(","):
        stripped = token.strip()
        if stripped:
            out.append(stripped)
    return out


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        size = int(param.numel())
        total += size
        if param.requires_grad:
            trainable += size
    return trainable, total


def maybe_apply_lora_to_encoder(
    *,
    encoder: nn.Module,
    args: argparse.Namespace,
    encoder_name: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if not args.use_lora:
        return encoder, {"enabled": False}

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as exc:
        raise ImportError(
            "LoRA を有効にするには `peft` が必要です。環境に `pip install peft` を追加してください。"
        ) from exc

    target_modules = parse_csv_items(args.lora_target_modules)
    if not target_modules:
        if "qwen3-embedding-8b" in encoder_name.lower():
            target_modules = list(DEFAULT_QWEN_LORA_TARGET_MODULES)
        else:
            target_modules = ["query", "value"]

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias=str(args.lora_bias),
    )
    encoder = get_peft_model(encoder, lora_config)

    if args.lora_gradient_checkpointing:
        if hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()
        base_model = encoder.get_base_model() if hasattr(encoder, "get_base_model") else encoder
        if hasattr(base_model, "config"):
            try:
                base_model.config.use_cache = False
            except Exception:
                pass

    trainable_params, total_params = count_trainable_parameters(encoder)
    if args.lora_print_trainable_params and hasattr(encoder, "print_trainable_parameters"):
        try:
            encoder.print_trainable_parameters()
        except Exception:
            pass

    return encoder, {
        "enabled": True,
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": target_modules,
        "bias": str(args.lora_bias),
        "gradient_checkpointing": bool(args.lora_gradient_checkpointing),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }


def resolve_hidden_size(model: nn.Module) -> int:
    target = model.module if isinstance(model, nn.DataParallel) else model
    config = getattr(target, "config", None)
    if config is None and hasattr(target, "base_model"):
        config = getattr(target.base_model, "config", None)
    hidden_size = getattr(config, "hidden_size", None) if config is not None else None
    if hidden_size is None:
        raise ValueError("encoder hidden_size could not be resolved from model config.")
    return int(hidden_size)


def sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return None


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def maybe_load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        print(f"[WARN] python-dotenv が未導入のため .env を読み込めません: {env_file}")
        return
    load_dotenv(dotenv_path=env_file, override=False)


def build_default_wandb_run_name(
    *,
    mode: str,
    encoder_alias: str,
    lr: float,
    batch_size: int,
    alpha: float,
    beta: float,
    seed: int,
) -> str:
    return (
        f"{sanitize_token(mode)}_{sanitize_token(encoder_alias)}_"
        f"lr{lr:g}_bs{batch_size}_a{alpha:g}_b{beta:g}_seed{seed}"
    )


def get_mode_task_pattern(mode: str) -> str:
    pattern = MODE_TO_TASK_PATTERN.get(mode)
    if pattern is None:
        return "unknown"
    return pattern


def init_wandb_run(
    *,
    args: argparse.Namespace,
    encoder_name: str,
    run_dir: Path,
) -> tuple[Any | None, Any | None]:
    if not args.use_wandb:
        return None, None

    maybe_load_env_file(Path(args.env_file))
    api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")
    if api_key and not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key
    if not api_key:
        print("[WARN] WANDB_API_KEY/WANDB_KEY が未設定のため wandb を無効化して実行します。")
        return None, None

    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[WARN] wandb の import に失敗したため無効化します: {exc}")
        return None, None

    run_name = args.wandb_run_name or build_default_wandb_run_name(
        mode=args.mode,
        encoder_alias=args.encoder,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
    )
    active_tasks = sorted(MODE_TO_TASKS[args.mode])
    task_pattern = get_mode_task_pattern(args.mode)
    wandb_tags = [
        f"mode:{args.mode}",
        f"task_pattern:{task_pattern}",
        *[f"task:{task_name}" for task_name in active_tasks],
    ]
    if args.wandb_group:
        wandb_tags.append("task1_mode_sweep")
    wandb_config = {
        "mode": args.mode,
        "active_tasks": active_tasks,
        "task_pattern": task_pattern,
        "encoder_name": encoder_name,
        "encoder_alias": args.encoder,
        "encoder_torch_dtype": args.encoder_torch_dtype,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "beta": args.beta,
        "max_len": args.max_len,
        "pooling": args.pooling,
        "normalize": args.normalize,
        "score_fn": args.score_fn,
        "pseudo_link_threshold": args.pseudo_link_threshold,
        "pseudo_link_weight": args.pseudo_link_weight,
        "taxonomy_restrict_mode": args.taxonomy_restrict_mode,
        "loss_type": args.loss_type,
        "ngram_n": args.ngram_n,
        "ngram_temperature": args.ngram_temperature,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "random_seed": args.seed,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "lora_bias": args.lora_bias,
        "lora_gradient_checkpointing": args.lora_gradient_checkpointing,
        "use_e5_prefix": args.use_e5_prefix,
        "wandb_group": args.wandb_group,
    }

    try:
        run = wandb.init(
            entity="yoshimaru",
            project="FoodMatchMTL",
            name=run_name,
            config=wandb_config,
            dir=str(run_dir),
            group=(args.wandb_group or None),
            tags=wandb_tags,
        )
        return wandb, run
    except Exception as exc:
        print(f"[WARN] wandb 初期化に失敗したため無効化します: {exc}")
        return None, None


@dataclass
class AliasExample:
    food_name: str
    food_table_name: str
    category_id: str
    category_idx: int


@dataclass
class TaxLinkInfo:
    tax_id: int
    score: float
    source: str  # "gold" or "pseudo"


class AliasPairDataset(Dataset[AliasExample]):
    def __init__(self, frame: pd.DataFrame, category_to_idx: dict[str, int]) -> None:
        self.examples: list[AliasExample] = []
        for _, row in frame.iterrows():
            cat_id = normalize_category_id(row["food_table_category_id"])
            if cat_id not in category_to_idx:
                continue
            self.examples.append(
                AliasExample(
                    food_name=str(row["food_name"]).strip(),
                    food_table_name=str(row["food_table_name"]).strip(),
                    category_id=cat_id,
                    category_idx=category_to_idx[cat_id],
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> AliasExample:
        return self.examples[index]


def alias_collate_fn(batch: list[AliasExample]) -> dict[str, Any]:
    return {
        "food_names": [item.food_name for item in batch],
        "food_table_names": [item.food_table_name for item in batch],
        "category_idx": torch.tensor([item.category_idx for item in batch], dtype=torch.long),
        "category_id": [item.category_id for item in batch],
    }


class CategoryClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(embeddings))


class ProjectionHead(nn.Module):
    """Task-specific 2-layer MLP projection head.

    エンコーダの共有表現からタスク固有のビューを提供し、タスク間の勾配干渉を防ぐ。
    - encoder は Task2/Task3 で構造化された表現を学習する
    - proj_id は Task1 専用の細粒度 ID 識別空間に変換する
    - 評価時も proj_id を通すことで、構造化エンコーダ上の ID 識別に特化した検索を行う

    Architecture: Linear -> LayerNorm -> GELU -> Linear (no bias)
    """

    def __init__(self, in_dim: int, out_dim: int | None = None, hidden_dim: int | None = None) -> None:
        super().__init__()
        out_d = out_dim or in_dim
        hidden_d = hidden_dim or max(in_dim // 2, 256)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_d, bias=True),
            nn.LayerNorm(hidden_d),
            nn.GELU(),
            nn.Linear(hidden_d, out_d, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dynamic task-weighting helpers
# ---------------------------------------------------------------------------

class UncertaintyWeighter(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al., NeurIPS 2018).

    Learns one log(σ²) per task as a trainable scalar.
    Weighted total loss:
        L = Σ_i  [ exp(-s_i) * L_i  +  s_i / 2 ]
    where s_i = log(σ_i²).

    Initialising s_i = 0  ⟺  σ_i = 1  ⟺  weight = 1 at step 0.
    As training progresses, tasks with high uncertainty (large σ) receive
    lower effective weights, acting as automatic task balancing.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in task_names}
        )

    def forward(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | None = None
        effective_weights: dict[str, float] = {}
        for name, loss in losses.items():
            s = self.log_vars[name]
            weighted = torch.exp(-s) * loss + s * 0.5
            effective_weights[name] = float(torch.exp(-s).detach().item())
            total = weighted if total is None else total + weighted
        assert total is not None
        return total, effective_weights

    def log_var_dict(self) -> dict[str, float]:
        return {name: float(self.log_vars[name].detach().item()) for name in self.task_names}


class DWAWeighter:
    """Dynamic Weight Average (Liu et al., CVPR 2019).

    At epoch t:
        w_i(t) = K * softmax( r_i(t) / T )
        r_i(t) = mean_L_i(t-1) / mean_L_i(t-2)

    where K = number of active tasks, T = temperature.
    Falls back to fixed weights for the first two epochs (insufficient history).
    """

    def __init__(
        self,
        task_names: list[str],
        fixed_weights: dict[str, float],
        temperature: float = 2.0,
    ) -> None:
        self.task_names = list(task_names)
        self.fixed_weights = dict(fixed_weights)
        self.temperature = max(float(temperature), 1e-6)
        self.loss_history: list[dict[str, float]] = []

    def compute_weights(self) -> dict[str, float]:
        K = len(self.task_names)
        if len(self.loss_history) < 2:
            return dict(self.fixed_weights)
        prev = self.loss_history[-1]
        prev2 = self.loss_history[-2]
        r = [
            prev.get(name, 1.0) / max(prev2.get(name, 1.0), 1e-8)
            for name in self.task_names
        ]
        r_tensor = torch.tensor(r, dtype=torch.float32)
        w = K * torch.softmax(r_tensor / self.temperature, dim=0)
        return {name: float(w[i].item()) for i, name in enumerate(self.task_names)}

    def record_epoch(self, epoch_losses: dict[str, float]) -> None:
        """Call at end of each epoch with per-task average losses."""
        self.loss_history.append({k: epoch_losses[k] for k in self.task_names if k in epoch_losses})

    @staticmethod
    def apply(
        losses: dict[str, torch.Tensor],
        weights: dict[str, float],
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for name, loss in losses.items():
            weighted = weights.get(name, 1.0) * loss
            total = weighted if total is None else total + weighted
        assert total is not None
        return total


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def pool_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    if pooling == "cls":
        return last_hidden_state[:, 0]
    raise ValueError(f"Unsupported pooling: {pooling}")


def format_texts(texts: list[str], text_type: str, use_e5_prefix: bool) -> list[str]:
    if not use_e5_prefix:
        return texts
    prefix = "query: " if text_type == "query" else "passage: "
    return [prefix + text for text in texts]


def encode_text_batch(
    *,
    encoder: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    device: torch.device,
    pooling: str,
    normalize: bool,
    text_type: str,
    use_e5_prefix: bool,
) -> torch.Tensor:
    prepared_texts = format_texts(texts, text_type=text_type, use_e5_prefix=use_e5_prefix)
    tokens = tokenizer(
        prepared_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = encoder(**tokens, return_dict=False)
    last_hidden_state = outputs[0]
    attention_mask = tokens["attention_mask"]
    if attention_mask.device != last_hidden_state.device:
        attention_mask = attention_mask.to(last_hidden_state.device)
    embeddings = pool_embeddings(last_hidden_state, attention_mask, pooling)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    if embeddings.device != device:
        embeddings = embeddings.to(device)
    return embeddings


@torch.no_grad()
def encode_texts_in_batches(
    *,
    encoder: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    device: torch.device,
    pooling: str,
    normalize: bool,
    text_type: str,
    use_e5_prefix: bool,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_emb = encode_text_batch(
            encoder=encoder,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_len=max_len,
            device=device,
            pooling=pooling,
            normalize=normalize,
            text_type=text_type,
            use_e5_prefix=use_e5_prefix,
        )
        chunks.append(batch_emb.detach().cpu())
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def similarity_matrix(a: torch.Tensor, b: torch.Tensor, score_fn: str) -> torch.Tensor:
    if score_fn == "cosine":
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
    return a @ b.T


def pair_similarity(a: torch.Tensor, b: torch.Tensor, score_fn: str) -> torch.Tensor:
    if score_fn == "cosine":
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
    return (a * b).sum(dim=1)


class TaxonomyDistance:
    def __init__(
        self,
        taxonomy_parent: dict[int, int],
        taxonomy_depth: dict[int, int] | None = None,
        max_distance: int = 10000,
    ) -> None:
        self.taxonomy_parent = taxonomy_parent
        self.taxonomy_depth = taxonomy_depth or {}
        self.max_distance = max_distance
        self.ancestor_cache: dict[int, dict[int, int]] = {}
        self.distance_cache: dict[tuple[int, int], int] = {}

    def _ancestors_with_distance(self, tax_id: int) -> dict[int, int]:
        cached = self.ancestor_cache.get(tax_id)
        if cached is not None:
            return cached

        ancestors: dict[int, int] = {}
        current = tax_id
        distance = 0
        seen: set[int] = set()
        while current not in seen:
            seen.add(current)
            ancestors[current] = distance
            parent = self.taxonomy_parent.get(current)
            if parent is None or parent == current:
                break
            current = parent
            distance += 1

        self.ancestor_cache[tax_id] = ancestors
        return ancestors

    def distance(self, tax_a: int, tax_b: int) -> int:
        if tax_a == tax_b:
            return 0
        key = (tax_a, tax_b) if tax_a < tax_b else (tax_b, tax_a)
        cached = self.distance_cache.get(key)
        if cached is not None:
            return cached

        ancestors_a = self._ancestors_with_distance(tax_a)
        current = tax_b
        distance_b = 0
        seen: set[int] = set()
        result = self.max_distance
        while current not in seen:
            seen.add(current)
            dist_a = ancestors_a.get(current)
            if dist_a is not None:
                result = dist_a + distance_b
                break
            parent = self.taxonomy_parent.get(current)
            if parent is None or parent == current:
                break
            current = parent
            distance_b += 1

        self.distance_cache[key] = int(result)
        return int(result)


class TaxonomyTripletSampler:
    def __init__(
        self,
        canonical_names: list[str],
        canonical_to_tax: dict[str, int],
        canonical_link_weight: dict[str, float] | None,
        taxonomy_distance: TaxonomyDistance,
        soft_pos_hop: int,
        neg_hop: int,
        seed: int,
    ) -> None:
        self.canonical_names = canonical_names
        self.canonical_to_tax = canonical_to_tax
        self.taxonomy_distance = taxonomy_distance
        self.soft_pos_hop = soft_pos_hop
        self.neg_hop = neg_hop
        self.rng = random.Random(seed)
        self.canonical_link_weight = canonical_link_weight or {}

        self.anchor_indices: list[int] = []
        self.tax_by_index: dict[int, int] = {}
        self.link_weight_by_index: dict[int, float] = {}
        self.indices_by_tax: dict[int, list[int]] = defaultdict(list)

        for idx, canonical in enumerate(canonical_names):
            tax_id = canonical_to_tax.get(canonical)
            if tax_id is None:
                continue
            self.anchor_indices.append(idx)
            self.tax_by_index[idx] = tax_id
            self.link_weight_by_index[idx] = float(self.canonical_link_weight.get(canonical, 1.0))
            self.indices_by_tax[tax_id].append(idx)

        self.tax_ids = sorted(self.indices_by_tax.keys())
        self.soft_tax_cache: dict[int, list[tuple[int, int]]] = {}
        self.neg_tax_cache: dict[int, list[int]] = {}

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_indices)

    def _soft_tax_candidates(self, anchor_tax: int) -> list[tuple[int, int]]:
        cached = self.soft_tax_cache.get(anchor_tax)
        if cached is not None:
            return cached
        candidates: list[tuple[int, int]] = []
        for candidate_tax in self.tax_ids:
            if candidate_tax == anchor_tax:
                continue
            hop = self.taxonomy_distance.distance(anchor_tax, candidate_tax)
            if 1 <= hop <= self.soft_pos_hop:
                if self.indices_by_tax[candidate_tax]:
                    candidates.append((candidate_tax, hop))
        self.soft_tax_cache[anchor_tax] = candidates
        return candidates

    def _neg_tax_candidates(self, anchor_tax: int) -> list[int]:
        cached = self.neg_tax_cache.get(anchor_tax)
        if cached is not None:
            return cached
        candidates: list[int] = []
        farthest_tax = None
        farthest_hop = -1
        for candidate_tax in self.tax_ids:
            if candidate_tax == anchor_tax:
                continue
            hop = self.taxonomy_distance.distance(anchor_tax, candidate_tax)
            if hop >= self.neg_hop:
                candidates.append(candidate_tax)
            if hop > farthest_hop:
                farthest_hop = hop
                farthest_tax = candidate_tax
        if not candidates and farthest_tax is not None:
            candidates = [farthest_tax]
        self.neg_tax_cache[anchor_tax] = candidates
        return candidates

    def _sample_positive(self, anchor_index: int, anchor_tax: int) -> tuple[int | None, float]:
        same_tax = [idx for idx in self.indices_by_tax[anchor_tax] if idx != anchor_index]
        if same_tax:
            return self.rng.choice(same_tax), 1.0

        soft_candidates = self._soft_tax_candidates(anchor_tax)
        if not soft_candidates:
            return None, 0.0

        weights = [1.0 / (hop + 1.0) for _, hop in soft_candidates]
        selected_tax, hop = self.rng.choices(soft_candidates, weights=weights, k=1)[0]
        selected_index = self.rng.choice(self.indices_by_tax[selected_tax])
        soft_weight = max(0.1, (self.soft_pos_hop - hop + 1) / (self.soft_pos_hop + 1))
        return selected_index, float(soft_weight)

    def _sample_negative(self, anchor_tax: int) -> int | None:
        neg_taxes = self._neg_tax_candidates(anchor_tax)
        if not neg_taxes:
            return None
        selected_tax = self.rng.choice(neg_taxes)
        return self.rng.choice(self.indices_by_tax[selected_tax])

    def sample_triplets(
        self,
        batch_size: int,
    ) -> tuple[list[int], list[int], list[int], torch.Tensor]:
        anchor_indices: list[int] = []
        pos_indices: list[int] = []
        neg_indices: list[int] = []
        pos_weights: list[float] = []

        if not self.anchor_indices:
            return anchor_indices, pos_indices, neg_indices, torch.empty((0,), dtype=torch.float32)

        max_trials = max(batch_size * 5, 32)
        trials = 0
        while len(anchor_indices) < batch_size and trials < max_trials:
            trials += 1
            anchor_idx = self.rng.choice(self.anchor_indices)
            anchor_tax = self.tax_by_index[anchor_idx]
            pos_idx, pos_weight = self._sample_positive(anchor_idx, anchor_tax)
            if pos_idx is None:
                continue
            neg_idx = self._sample_negative(anchor_tax)
            if neg_idx is None:
                continue
            anchor_indices.append(anchor_idx)
            pos_indices.append(pos_idx)
            neg_indices.append(neg_idx)
            pair_weight = (
                pos_weight
                * self.link_weight_by_index.get(anchor_idx, 1.0)
                * self.link_weight_by_index.get(pos_idx, 1.0)
            )
            pos_weights.append(float(pair_weight))

        return (
            anchor_indices,
            pos_indices,
            neg_indices,
            torch.tensor(pos_weights, dtype=torch.float32),
        )

    def maybe_precompute(self) -> None:
        for tax_id in self.tax_ids:
            self._soft_tax_candidates(tax_id)
            self._neg_tax_candidates(tax_id)


def compute_triplet_loss(
    anchor_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    neg_embeddings: torch.Tensor,
    pos_weights: torch.Tensor,
    margin: float,
    score_fn: str,
) -> torch.Tensor:
    sim_pos = pair_similarity(anchor_embeddings, pos_embeddings, score_fn=score_fn)
    sim_neg = pair_similarity(anchor_embeddings, neg_embeddings, score_fn=score_fn)
    losses = F.relu(sim_neg - sim_pos + margin)
    weighted = losses * pos_weights.to(losses.device)
    return weighted.mean()


def _get_char_ngram_counts(text: str, n: int) -> dict[str, int]:
    """文字N-gramのカウント辞書を返す。"""
    normalized = text.lower().replace(" ", "").replace("\u3000", "")
    counts: dict[str, int] = {}
    for i in range(len(normalized) - n + 1):
        gram = normalized[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def compute_ngram_bm25_matrix(
    queries: list[str],
    canonicals: list[str],
    n: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> torch.Tensor:
    """文字N-gramをトークンとしたBM25スコア行列を計算する。

    行 = query、列 = canonical。スコアは非負の実数。
    同一バッチ内のcanonicalをコーパスとしてIDF・平均文書長を計算する。
    """
    # --- canonical側の統計量 ---
    doc_ngrams_list: list[dict[str, int]] = [_get_char_ngram_counts(c, n) for c in canonicals]
    doc_lengths = [sum(counts.values()) for counts in doc_ngrams_list]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

    N = len(canonicals)
    df: dict[str, int] = {}
    for counts in doc_ngrams_list:
        for gram in counts:
            df[gram] = df.get(gram, 0) + 1
    # Robertson IDF (smooth)
    idf: dict[str, float] = {
        gram: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1.0)
        for gram, cnt in df.items()
    }

    sim = torch.zeros(len(queries), len(canonicals), dtype=torch.float32)
    for i, query in enumerate(queries):
        query_ngrams = _get_char_ngram_counts(query, n)
        for j, doc_ngrams in enumerate(doc_ngrams_list):
            dl = doc_lengths[j]
            score = 0.0
            for gram, tf_q in query_ngrams.items():
                if gram not in doc_ngrams:
                    continue
                tf_d = doc_ngrams[gram]
                idf_val = idf.get(gram, 0.0)
                denom = tf_d + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
                score += idf_val * (tf_d * (k1 + 1.0)) / denom
            sim[i, j] = score
    return sim


def compute_ngram_infonce_loss(
    logits: torch.Tensor,
    query_texts: list[str],
    canonical_texts: list[str],
    ngram_n: int,
    ngram_temperature: float,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> torch.Tensor:
    """N-gram BM25スコアをソフトラベルとして使うInfoNCE Loss。

    通常のInfoNCEが対角要素のみを正例とするのに対し、
    バッチ内の全ペアのBM25スコアをtarget分布として使用する。
    これによりN-gram的に類似した食品名を"部分的な正例"として扱える。
    """
    ngram_sim = compute_ngram_bm25_matrix(
        query_texts, canonical_texts, n=ngram_n, k1=bm25_k1, b=bm25_b
    ).to(logits.device)

    # diagonal (真の正例) は必ず最大値以上になるようクリップして安定化
    diag_scores = ngram_sim.diagonal()
    # ソフトラベルを temperature でスケールして softmax
    soft_labels = F.softmax(ngram_sim / max(ngram_temperature, 1e-6), dim=1)

    # KL divergence 形式のクロスエントロピー: -sum(soft_labels * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_labels * log_probs).sum(dim=1).mean()
    return loss


def compute_taxonomy_infonce_loss(
    logits: torch.Tensor,
    batch_canonicals: list[str],
    canonical_to_tax: dict[str, int],
    taxonomy_distance: "TaxonomyDistance",
    tax_soft_temp: float = 2.0,
) -> torch.Tensor:
    """Taxonomy-aware soft-label InfoNCE。

    通常の InfoNCE が対角のみを正例とするのに対し、taxonomy 距離に基づくソフトラベルを使用する。
    - 対角（真の正例）: weight=1.0 (+ exp(0/T)=1.0)
    - 同一または近い taxonomy: 小さい正の重み（弱い正例として扱う）
    - 遠い taxonomy: ほぼ 0 の重み（強い負例）

    これにより Task3（taxonomy triplet）の構造信号を Task1 の retrieval 損失に直接統合する。
    taxonomy mapping がない canonical が多い場合は通常の InfoNCE に退化する。
    """
    B = logits.size(0)
    tax_ids = [canonical_to_tax.get(c) for c in batch_canonicals]
    temp = max(tax_soft_temp, 1e-6)

    # ソフトラベル行列を構築（対角=1.0、それ以外はtaxonomy距離から計算）
    scores = torch.zeros(B, B, dtype=torch.float32)
    for i in range(B):
        scores[i, i] = 1.0  # hard positive
        if tax_ids[i] is None:
            continue
        for j in range(B):
            if i == j or tax_ids[j] is None:
                continue
            dist = taxonomy_distance.distance(tax_ids[i], tax_ids[j])
            # 近い taxonomy ほど高い重み: exp(-dist / T)
            scores[i, j] = math.exp(-dist / temp)

    # 行方向に softmax でターゲット分布へ変換
    soft_labels = F.softmax(scores.to(logits.device), dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    # KL-divergence スタイルのクロスエントロピー
    loss = -(soft_labels * log_probs).sum(dim=1).mean()
    return loss


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_train_pairs(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"food_name", "food_table_name", "food_table_category_id"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"train pair CSV missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["food_name", "food_table_name", "food_table_category_id"]).copy()
    frame["food_name"] = frame["food_name"].astype(str).str.strip()
    frame["food_table_name"] = frame["food_table_name"].astype(str).str.strip()
    frame["food_table_category_id"] = frame["food_table_category_id"].map(normalize_category_id)
    frame = frame[(frame["food_name"] != "") & (frame["food_table_name"] != "")]
    frame = frame.reset_index(drop=True)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return frame


def load_eval_ingredients(path: Path, max_items: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Preferred format for ours_v2 experiments:
    # ingredient_name, foodtable_id, foodtable_name
    if {"ingredient_name", "foodtable_id"}.issubset(frame.columns):
        frame = frame.copy()
        frame["ingredient"] = frame["ingredient_name"].astype(str).str.replace("\u3000", "", regex=False)
        frame["foodtable_maping"] = frame["foodtable_id"]
        frame["gold_id"] = frame["foodtable_maping"].map(normalize_food_id)
        if max_items > 0:
            frame = frame.head(max_items).reset_index(drop=True)
        return frame[["ingredient", "foodtable_maping", "gold_id"]]

    required = {"ingredient", "foodtable_maping"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"eval CSV missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["foodtable_maping"]).copy()
    frame["ingredient"] = frame["ingredient"].astype(str).str.replace("\u3000", "", regex=False)
    frame["gold_id"] = frame["foodtable_maping"].map(normalize_food_id)
    if max_items > 0:
        frame = frame.head(max_items).reset_index(drop=True)
    return frame[["ingredient", "foodtable_maping", "gold_id"]]


def build_canonical_to_tax_mapping(
    canonical_names: Iterable[str],
    food2tax_raw: dict[Any, Any],
    foodtable_name2id: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    canonical_to_tax: dict[str, int] = {}
    stats = {
        "key_is_name": 0,
        "key_is_id": 0,
        "missing_food_id": 0,
        "missing_food2tax": 0,
    }

    food2tax_str: dict[str, int] = {}
    for key, value in food2tax_raw.items():
        try:
            food2tax_str[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    for canonical in canonical_names:
        if canonical in food2tax_raw:
            try:
                canonical_to_tax[canonical] = int(food2tax_raw[canonical])
                stats["key_is_name"] += 1
                continue
            except (TypeError, ValueError):
                pass

        food_id = foodtable_name2id.get(canonical)
        if food_id is None:
            stats["missing_food_id"] += 1
            continue

        tax_id = food2tax_str.get(str(food_id))
        if tax_id is None:
            stats["missing_food2tax"] += 1
            continue
        canonical_to_tax[canonical] = int(tax_id)
        stats["key_is_id"] += 1

    return canonical_to_tax, stats


def parse_dmp_line(line: str) -> list[str]:
    line = line.rstrip("\n")
    if line.endswith("\t|"):
        line = line[:-2]
    return [part.strip() for part in line.split("\t|\t")]


def collect_ancestors(tax_id: int, taxonomy_parent: dict[int, int]) -> list[int]:
    ancestors: list[int] = []
    current = tax_id
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        ancestors.append(current)
        parent = taxonomy_parent.get(current)
        if parent is None or parent == current:
            break
        current = parent
    return ancestors


def collect_children_for_targets(taxonomy_parent: dict[int, int], targets: set[int]) -> set[int]:
    children: set[int] = set()
    for child, parent in taxonomy_parent.items():
        if child == parent:
            continue
        if parent in targets:
            children.add(child)
    return children


def collect_rank_filtered_tax_ids(
    *,
    nodes_dmp_path: Path,
    allowed_ranks: set[str],
) -> set[int]:
    if not nodes_dmp_path.exists():
        raise FileNotFoundError(f"nodes.dmp not found: {nodes_dmp_path}")
    selected: set[int] = set()
    with nodes_dmp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = parse_dmp_line(line)
            if len(parts) < 3:
                continue
            try:
                tax_id = int(parts[0])
            except ValueError:
                continue
            rank = parts[2].strip().lower()
            if rank in allowed_ranks:
                selected.add(tax_id)
    return selected


def select_taxonomy_candidate_ids(
    *,
    restrict_mode: str,
    linked_tax_ids: set[int],
    taxonomy_parent: dict[int, int],
    nodes_dmp_path: Path,
    taxonomy_max_nodes: int,
) -> tuple[set[int], dict[str, Any]]:
    stats: dict[str, Any] = {
        "restrict_mode": restrict_mode,
        "linked_tax_ids": int(len(linked_tax_ids)),
        "fallback_mode": None,
        "truncated": False,
    }
    candidates: set[int] = set()

    if restrict_mode == "ancestors_of_linked":
        if linked_tax_ids:
            ancestors: set[int] = set()
            for tax_id in linked_tax_ids:
                ancestors.update(collect_ancestors(tax_id, taxonomy_parent))
            neighbor_targets = set(linked_tax_ids)
            for tax_id in linked_tax_ids:
                parent = taxonomy_parent.get(tax_id)
                if parent is not None:
                    neighbor_targets.add(parent)
            children = collect_children_for_targets(taxonomy_parent, neighbor_targets)
            candidates = ancestors | neighbor_targets | children
            stats["ancestors_count"] = int(len(ancestors))
            stats["neighbors_count"] = int(len(children))
        else:
            stats["fallback_mode"] = "rank_filter"
            candidates = collect_rank_filtered_tax_ids(
                nodes_dmp_path=nodes_dmp_path,
                allowed_ranks=DEFAULT_TAXONOMY_RANK_FILTER,
            )
    elif restrict_mode == "rank_filter":
        candidates = collect_rank_filtered_tax_ids(
            nodes_dmp_path=nodes_dmp_path,
            allowed_ranks=DEFAULT_TAXONOMY_RANK_FILTER,
        )
    elif restrict_mode == "all":
        candidates = set(taxonomy_parent.keys())
    else:
        raise ValueError(f"Unsupported taxonomy_restrict_mode: {restrict_mode}")

    stats["candidates_before_cap"] = int(len(candidates))
    if taxonomy_max_nodes > 0 and len(candidates) > taxonomy_max_nodes:
        capped: set[int] = set()
        if linked_tax_ids:
            for tax_id in linked_tax_ids:
                if tax_id in candidates:
                    capped.add(tax_id)
            for tax_id in linked_tax_ids:
                for ancestor in collect_ancestors(tax_id, taxonomy_parent):
                    if ancestor in candidates:
                        capped.add(ancestor)
                    if len(capped) >= taxonomy_max_nodes:
                        break
                if len(capped) >= taxonomy_max_nodes:
                    break
        if len(capped) < taxonomy_max_nodes:
            for tax_id in sorted(candidates):
                capped.add(tax_id)
                if len(capped) >= taxonomy_max_nodes:
                    break
        candidates = capped
        stats["truncated"] = True
    stats["candidates_after_cap"] = int(len(candidates))
    return candidates, stats


def load_taxonomy_names_subset(
    taxonomy_names_path: Path,
    target_tax_ids: set[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], str]:
    target_tax_keys = None if target_tax_ids is None else {str(int(tid)) for tid in target_tax_ids}
    records: dict[int, dict[str, Any]] = {}

    parse_failed = False
    with taxonomy_names_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped in {"{", "}"}:
                continue
            if stripped.endswith(","):
                stripped = stripped[:-1]
            if ": " not in stripped:
                continue
            key_part, value_part = stripped.split(": ", 1)
            if not key_part.startswith('"'):
                continue
            try:
                key = json.loads(key_part)
                if target_tax_keys is not None and key not in target_tax_keys:
                    continue
                value = json.loads(value_part)
                records[int(key)] = value
            except Exception:
                parse_failed = True
                break

    if not parse_failed:
        return records, "line_parser"

    # Fallback when line parser cannot decode (format changed, unexpected multiline etc.).
    full = load_json(taxonomy_names_path)
    if not isinstance(full, dict):
        raise ValueError(f"taxonomy_names JSON must be object: {taxonomy_names_path}")
    if target_tax_keys is None:
        return {int(k): v for k, v in full.items()}, "full_json"
    for key, value in full.items():
        if key in target_tax_keys:
            records[int(key)] = value
    return records, "full_json"


def normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def build_taxonomy_node_text(name_obj: dict[str, Any], max_parts: int = 3) -> str | None:
    pieces: list[str] = []
    seen: set[str] = set()
    fields = [
        normalize_text_list(name_obj.get("scientific")),
        normalize_text_list(name_obj.get("english")),
        normalize_text_list(name_obj.get("japanese")),
        normalize_text_list(name_obj.get("synonyms")),
    ]
    for candidates in fields:
        if not candidates:
            continue
        candidate = candidates[0].strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        pieces.append(candidate)
        seen.add(key)
        if len(pieces) >= max_parts:
            break
    if not pieces:
        return None
    return " | ".join(pieces[:max_parts])


def serialize_tax_link_info(tax_link_info: dict[str, TaxLinkInfo]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for canonical, info in sorted(tax_link_info.items()):
        payload[canonical] = {
            "tax_id": int(info.tax_id),
            "score": float(info.score),
            "source": info.source,
        }
    return payload


def load_cached_tax_link_info(
    *,
    cache_path: Path,
    canonical_names: list[str],
    gold_canonical_to_tax: dict[str, int],
) -> tuple[dict[str, TaxLinkInfo], dict[str, Any]]:
    raw_payload = load_json(cache_path)
    canonical_set = set(canonical_names)
    tax_link_info: dict[str, TaxLinkInfo] = {}

    for canonical, raw in raw_payload.items():
        if canonical not in canonical_set or not isinstance(raw, dict):
            continue
        try:
            tax_id = int(raw["tax_id"])
        except (KeyError, TypeError, ValueError):
            continue
        score = float(raw.get("score", 1.0))
        source = str(raw.get("source", "pseudo"))
        tax_link_info[canonical] = TaxLinkInfo(tax_id=tax_id, score=score, source=source)

    for canonical, tax_id in gold_canonical_to_tax.items():
        if canonical in canonical_set:
            tax_link_info[canonical] = TaxLinkInfo(tax_id=int(tax_id), score=1.0, source="gold")

    pseudo_linked = sum(1 for info in tax_link_info.values() if info.source != "gold")
    stats = {
        "gold_linked": int(len(gold_canonical_to_tax)),
        "pseudo_linked": int(pseudo_linked),
        "excluded_unlinked": int(len(canonical_names) - len(tax_link_info)),
        "candidate_tax_nodes": None,
        "taxonomy_names_parser": None,
        "node_text_count": None,
        "threshold": None,
        "loaded_from_cache": True,
        "cache_path": str(cache_path),
    }
    return tax_link_info, stats


@torch.no_grad()
def build_pseudo_tax_links(
    *,
    canonical_names: list[str],
    gold_canonical_to_tax: dict[str, int],
    encoder: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_len: int,
    pooling: str,
    normalize: bool,
    use_e5_prefix: bool,
    taxonomy_parent: dict[int, int],
    taxonomy_names_path: Path,
    nodes_dmp_path: Path,
    restrict_mode: str,
    taxonomy_max_nodes: int,
    pseudo_link_threshold: float,
    pseudo_link_batch_size: int,
) -> tuple[dict[str, TaxLinkInfo], dict[str, Any]]:
    linked_tax_ids = {int(tax_id) for tax_id in gold_canonical_to_tax.values()}
    candidate_tax_ids, restrict_stats = select_taxonomy_candidate_ids(
        restrict_mode=restrict_mode,
        linked_tax_ids=linked_tax_ids,
        taxonomy_parent=taxonomy_parent,
        nodes_dmp_path=nodes_dmp_path,
        taxonomy_max_nodes=taxonomy_max_nodes,
    )
    if not candidate_tax_ids:
        tax_link_info = {
            canonical: TaxLinkInfo(tax_id=int(tax_id), score=1.0, source="gold")
            for canonical, tax_id in gold_canonical_to_tax.items()
        }
        stats = {
            "gold_linked": int(len(gold_canonical_to_tax)),
            "pseudo_linked": 0,
            "excluded_unlinked": int(len(canonical_names) - len(gold_canonical_to_tax)),
            "candidate_tax_nodes": 0,
            "taxonomy_names_parser": None,
            "node_text_count": 0,
            "threshold": float(pseudo_link_threshold),
            "restrict": restrict_stats,
        }
        return tax_link_info, stats

    taxonomy_names, parser_name = load_taxonomy_names_subset(
        taxonomy_names_path=taxonomy_names_path,
        target_tax_ids=candidate_tax_ids,
    )
    node_tax_ids: list[int] = []
    node_texts: list[str] = []
    for tax_id in sorted(candidate_tax_ids):
        name_obj = taxonomy_names.get(tax_id)
        if not isinstance(name_obj, dict):
            continue
        node_text = build_taxonomy_node_text(name_obj, max_parts=3)
        if node_text is None:
            continue
        node_tax_ids.append(tax_id)
        node_texts.append(node_text)

    tax_link_info: dict[str, TaxLinkInfo] = {
        canonical: TaxLinkInfo(tax_id=int(tax_id), score=1.0, source="gold")
        for canonical, tax_id in gold_canonical_to_tax.items()
    }

    unresolved_canonical = [name for name in canonical_names if name not in gold_canonical_to_tax]
    pseudo_linked = 0
    if unresolved_canonical and node_texts:
        node_embeddings = encode_texts_in_batches(
            encoder=encoder,
            tokenizer=tokenizer,
            texts=node_texts,
            max_len=max_len,
            device=device,
            pooling=pooling,
            normalize=normalize,
            text_type="passage",
            use_e5_prefix=use_e5_prefix,
            batch_size=pseudo_link_batch_size,
        )
        unresolved_embeddings = encode_texts_in_batches(
            encoder=encoder,
            tokenizer=tokenizer,
            texts=unresolved_canonical,
            max_len=max_len,
            device=device,
            pooling=pooling,
            normalize=normalize,
            text_type="query",
            use_e5_prefix=use_e5_prefix,
            batch_size=pseudo_link_batch_size,
        )
        top_indices, top_scores = retrieve_top1_with_scores(
            query_embeddings_cpu=unresolved_embeddings,
            candidate_embeddings_cpu=node_embeddings,
            device=device,
            chunk_size=pseudo_link_batch_size,
        )
        for i, canonical in enumerate(unresolved_canonical):
            score = float(top_scores[i].item())
            if score < pseudo_link_threshold:
                continue
            tax_id = int(node_tax_ids[int(top_indices[i].item())])
            tax_link_info[canonical] = TaxLinkInfo(
                tax_id=tax_id,
                score=score,
                source="pseudo",
            )
            pseudo_linked += 1

    excluded_unlinked = len(canonical_names) - len(tax_link_info)
    stats = {
        "gold_linked": int(len(gold_canonical_to_tax)),
        "pseudo_linked": int(pseudo_linked),
        "excluded_unlinked": int(excluded_unlinked),
        "candidate_tax_nodes": int(len(candidate_tax_ids)),
        "taxonomy_names_parser": parser_name,
        "node_text_count": int(len(node_texts)),
        "threshold": float(pseudo_link_threshold),
        "restrict": restrict_stats,
    }
    return tax_link_info, stats


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        return []
    parsed: list[int] = []
    for token in gpu_ids.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    return parsed


def choose_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA not available but device requested: {device_arg}")
        return torch.device(normalized)
    return torch.device(normalized)


def get_hf_device_map(model: nn.Module) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return device_map
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else None
    device_map = getattr(base_model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return device_map
    return {}


def resolve_model_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    device_map = get_hf_device_map(model)
    if not device_map:
        return fallback

    cuda_indices: list[int] = []
    for target in device_map.values():
        if isinstance(target, int):
            cuda_indices.append(int(target))
        elif isinstance(target, str) and target.startswith("cuda"):
            cuda_indices.append(int(target.split(":", 1)[1]) if ":" in target else 0)

    if cuda_indices:
        return torch.device(f"cuda:{min(cuda_indices)}")
    if any(target == "cpu" for target in device_map.values()):
        return torch.device("cpu")
    return fallback


def retrieve_top1_indices(
    *,
    query_embeddings_cpu: torch.Tensor,
    candidate_embeddings_cpu: torch.Tensor,
    score_fn: str,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    if len(query_embeddings_cpu) == 0:
        return torch.empty((0,), dtype=torch.long)
    candidate_embeddings = candidate_embeddings_cpu.to(device)
    top_indices: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings_cpu), chunk_size):
        chunk = query_embeddings_cpu[start : start + chunk_size].to(device)
        scores = similarity_matrix(chunk, candidate_embeddings, score_fn=score_fn)
        pred = torch.argmax(scores, dim=1)
        top_indices.append(pred.detach().cpu())
    return torch.cat(top_indices, dim=0)


def retrieve_topk_indices_with_scores(
    *,
    query_embeddings_cpu: torch.Tensor,
    candidate_embeddings_cpu: torch.Tensor,
    score_fn: str,
    device: torch.device,
    chunk_size: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(query_embeddings_cpu) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.float32),
        )
    effective_topk = max(1, min(int(topk), int(candidate_embeddings_cpu.size(0))))
    candidate_embeddings = candidate_embeddings_cpu.to(device)
    top_indices: list[torch.Tensor] = []
    top_scores: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings_cpu), chunk_size):
        chunk = query_embeddings_cpu[start : start + chunk_size].to(device)
        scores = similarity_matrix(chunk, candidate_embeddings, score_fn=score_fn)
        best_scores, best_indices = torch.topk(scores, k=effective_topk, dim=1)
        top_indices.append(best_indices.detach().cpu())
        top_scores.append(best_scores.detach().cpu())
    return torch.cat(top_indices, dim=0), torch.cat(top_scores, dim=0)


def retrieve_top1_with_scores(
    *,
    query_embeddings_cpu: torch.Tensor,
    candidate_embeddings_cpu: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(query_embeddings_cpu) == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
    candidate_embeddings = F.normalize(candidate_embeddings_cpu.to(device), p=2, dim=1)
    top_indices: list[torch.Tensor] = []
    top_scores: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings_cpu), chunk_size):
        chunk = query_embeddings_cpu[start : start + chunk_size].to(device)
        chunk = F.normalize(chunk, p=2, dim=1)
        scores = chunk @ candidate_embeddings.T
        best_scores, best_indices = torch.max(scores, dim=1)
        top_indices.append(best_indices.detach().cpu())
        top_scores.append(best_scores.detach().cpu())
    return torch.cat(top_indices, dim=0), torch.cat(top_scores, dim=0)


@torch.no_grad()
def _apply_proj_head_batched(
    proj_head: nn.Module,
    embeddings_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    """Projection head を CPU テンソルにバッチ適用する（評価用）。"""
    proj_head.eval()
    chunks: list[torch.Tensor] = []
    for i in range(0, embeddings_cpu.size(0), batch_size):
        chunk = embeddings_cpu[i : i + batch_size].to(device)
        out = proj_head(chunk)
        if normalize:
            out = F.normalize(out, p=2, dim=1)
        chunks.append(out.detach().cpu())
    if not chunks:
        return embeddings_cpu
    return torch.cat(chunks, dim=0)


def evaluate_retrieval(
    *,
    encoder: nn.Module,
    tokenizer: Any,
    canonical_names: list[str],
    eval_frame: pd.DataFrame,
    foodtable_name2id: dict[str, int],
    device: torch.device,
    max_len: int,
    pooling: str,
    normalize: bool,
    score_fn: str,
    use_e5_prefix: bool,
    eval_batch_size: int,
    eval_topk: int,
    per_item: bool,
    save_embeddings: bool,
    proj_head: nn.Module | None = None,
) -> dict[str, Any]:
    encoder.eval()
    ingredients = eval_frame["ingredient"].astype(str).tolist()
    gold_ids = eval_frame["gold_id"].astype(str).tolist()

    candidate_encoder_embeddings = encode_texts_in_batches(
        encoder=encoder,
        tokenizer=tokenizer,
        texts=canonical_names,
        max_len=max_len,
        device=device,
        pooling=pooling,
        normalize=normalize,
        text_type="passage",
        use_e5_prefix=use_e5_prefix,
        batch_size=eval_batch_size,
    )
    query_encoder_embeddings = encode_texts_in_batches(
        encoder=encoder,
        tokenizer=tokenizer,
        texts=ingredients,
        max_len=max_len,
        device=device,
        pooling=pooling,
        normalize=normalize,
        text_type="query",
        use_e5_prefix=use_e5_prefix,
        batch_size=eval_batch_size,
    )
    candidate_embeddings = candidate_encoder_embeddings
    query_embeddings = query_encoder_embeddings
    # Projection head を評価に適用（use_proj_head_id=True の場合）
    if proj_head is not None:
        candidate_embeddings = _apply_proj_head_batched(
            proj_head, candidate_embeddings, eval_batch_size, device, normalize
        )
        query_embeddings = _apply_proj_head_batched(
            proj_head, query_embeddings, eval_batch_size, device, normalize
        )
    effective_topk = max(1, min(int(eval_topk), max(1, len(canonical_names))))
    topk_indices, topk_scores = retrieve_topk_indices_with_scores(
        query_embeddings_cpu=query_embeddings,
        candidate_embeddings_cpu=candidate_embeddings,
        score_fn=score_fn,
        device=device,
        chunk_size=eval_batch_size,
        topk=effective_topk,
    )

    acc1_id_values: list[int] = []
    acc10_id_values: list[int] = []
    acc1_cat_values: list[int] = []
    acc10_cat_values: list[int] = []
    num_unmapped = 0
    details: list[dict[str, Any]] = []

    for i, ingredient in enumerate(ingredients):
        candidate_rankings: list[dict[str, Any]] = []
        predicted_ids: list[str | None] = []
        predicted_cat_prefixes: list[str | None] = []
        for rank, (canonical_idx, score_value) in enumerate(
            zip(topk_indices[i].tolist(), topk_scores[i].tolist()),
            start=1,
        ):
            pred_canonical_k = canonical_names[int(canonical_idx)]
            pred_id_k = normalize_food_id(foodtable_name2id.get(pred_canonical_k))
            predicted_ids.append(pred_id_k)
            predicted_cat_prefixes.append(None if pred_id_k is None else pred_id_k[:2])
            candidate_rankings.append(
                {
                    "rank": rank,
                    "canonical": pred_canonical_k,
                    "id": pred_id_k,
                    "score": float(score_value),
                }
            )

        pred_canonical = candidate_rankings[0]["canonical"]
        pred_id = candidate_rankings[0]["id"]
        gold_id = gold_ids[i]
        gold_cat = gold_id[:2] if gold_id else None
        if pred_id is None:
            num_unmapped += 1
        acc1_id = int(pred_id is not None and pred_id == gold_id)
        acc10_id = int(gold_id in predicted_ids)
        acc1_cat = int(pred_id is not None and gold_cat is not None and pred_id[:2] == gold_cat)
        acc10_cat = int(gold_cat is not None and gold_cat in predicted_cat_prefixes)
        gold_rank_id = next((rank for rank, pred_id_k in enumerate(predicted_ids, start=1) if pred_id_k == gold_id), None)
        gold_rank_category = next(
            (
                rank
                for rank, pred_cat_prefix in enumerate(predicted_cat_prefixes, start=1)
                if gold_cat is not None and pred_cat_prefix == gold_cat
            ),
            None,
        )
        acc1_id_values.append(acc1_id)
        acc10_id_values.append(acc10_id)
        acc1_cat_values.append(acc1_cat)
        acc10_cat_values.append(acc10_cat)
        if per_item:
            details.append(
                {
                    "ingredient": ingredient,
                    "gold_id": gold_id,
                    "gold_category": gold_cat,
                    "pred_canonical": pred_canonical,
                    "pred_id": pred_id,
                    "em": acc1_id,
                    "cat_match": acc1_cat,
                    "acc1_id": acc1_id,
                    "acc10_id": acc10_id,
                    "acc1_category": acc1_cat,
                    "acc10_category": acc10_cat,
                    "gold_rank_id": gold_rank_id,
                    "gold_rank_category": gold_rank_category,
                    "topk_predictions": candidate_rankings,
                }
            )

    result = {
        "metrics": {
            "EM": safe_mean(acc1_id_values),
            "CategoryMatch": safe_mean(acc1_cat_values),
            "Acc@1_ID": safe_mean(acc1_id_values),
            "Acc@10_ID": safe_mean(acc10_id_values),
            "Acc@1_Category": safe_mean(acc1_cat_values),
            "Acc@10_Category": safe_mean(acc10_cat_values),
            "num_unmapped_pred": int(num_unmapped),
        },
    }
    if per_item:
        result["per_item"] = details
    if save_embeddings:
        embedding_payload: dict[str, Any] = {
            "meta": {
                "datetime": datetime.now().isoformat(),
                "num_queries": int(len(ingredients)),
                "embedding_dim": int(query_encoder_embeddings.size(1)) if query_encoder_embeddings.ndim == 2 else 0,
                "normalize": bool(normalize),
                "score_fn": score_fn,
                "proj_head_applied": bool(proj_head is not None),
            },
            "ingredients": ingredients,
            "gold_ids": gold_ids,
            "gold_categories": [gold_id[:2] if gold_id else None for gold_id in gold_ids],
            "query_encoder_embeddings": query_encoder_embeddings.detach().cpu().contiguous(),
        }
        if proj_head is not None:
            embedding_payload["query_retrieval_embeddings"] = query_embeddings.detach().cpu().contiguous()
        result["artifacts"] = {
            "eval_query_embeddings": embedding_payload,
        }
    return result


def create_run_dir(
    output_root: Path,
    mode: str,
    encoder_tag: str,
    lr: float,
    batch_size: int,
    alpha: float,
    beta: float,
    seed: int,
    loss_type: str = "infonce",
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_suffix = f"_loss-{sanitize_token(loss_type)}" if loss_type != "infonce" else ""
    run_name = (
        f"{timestamp}_{sanitize_token(mode)}_{sanitize_token(encoder_tag)}_"
        f"lr{lr:g}_bs{batch_size}_a{alpha:g}_b{beta:g}_seed{seed}{loss_suffix}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multitask training for ingredient matching.")
    parser.add_argument("--mode", type=str, choices=TASK1_MODE_SWEEP, required=True)
    parser.add_argument(
        "--encoder",
        type=str,
        default="multilingual_e5_large",
        help=(
            "Encoder key or HF model name. "
            "keys: multilingual_e5_large, sentence_bert, qwen3_8b"
        ),
    )
    parser.add_argument("--train_data_dir", type=Path, default=Path("./src/FoodMatchMTL/train_data"))
    parser.add_argument(
        "--eval_path",
        type=Path,
        default=Path("data/ingredient_label.csv"),
    )
    parser.add_argument(
        "--foodtable_name2id_path",
        type=Path,
        default=Path("./data/foodtable_name2id.json"),
    )
    parser.add_argument("--taxonomy_names_path", type=Path, default=None)
    parser.add_argument("--nodes_dmp_path", type=Path, default=Path("data/data_bio_NCBI/nodes.dmp"))
    parser.add_argument("--pseudo_food2tax_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=Path("./src/FoodMatchMTL/results"))
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--pooling", type=str, choices=["mean", "cls"], default="mean")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--score_fn", type=str, choices=["dot", "cosine"], default="cosine")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--soft_pos_hop", type=int, default=2)
    parser.add_argument("--neg_hop", type=int, default=6)
    parser.add_argument("--tax_batch_size", type=int, default=64)
    parser.add_argument("--enable_pseudo_link", type=str2bool, default=True)
    parser.add_argument("--pseudo_link_threshold", type=float, default=0.75)
    parser.add_argument("--pseudo_link_weight", type=float, default=0.5)
    parser.add_argument(
        "--taxonomy_restrict_mode",
        type=str,
        choices=["all", "rank_filter", "ancestors_of_linked"],
        default="ancestors_of_linked",
    )
    parser.add_argument("--taxonomy_max_nodes", type=int, default=0)
    parser.add_argument("--pseudo_link_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi_gpu", type=str2bool, default=False)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument(
        "--model_device_map",
        type=str,
        choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
        default="none",
        help="Shard a single model across multiple visible GPUs instead of replicating it.",
    )
    parser.add_argument(
        "--max_memory_per_gpu_gib",
        type=int,
        default=0,
        help="Optional per-GPU max memory budget used with --model_device_map.",
    )
    parser.add_argument(
        "--encoder_torch_dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_bias", type=str, choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--lora_gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--lora_print_trainable_params", type=str2bool, default=True)
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_log_interval", type=int, default=50)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--env_file", type=Path, default=Path("./.wandb_access.env"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--eval_max_items", type=int, default=0)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--use_e5_prefix", type=str2bool_or_auto, default="auto")
    parser.add_argument("--precompute_tax_cache", type=str2bool, default=False)
    parser.add_argument("--use_hard_negatives", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--save_per_item", type=str2bool, default=True)
    parser.add_argument("--save_eval_embeddings", type=str2bool, default=False)
    parser.add_argument("--run_task1_mode_sweep", type=str2bool, default=False)
    # --- Ablation: loss type ---
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["infonce", "ngram_infonce", "tax_infonce"],
        default="infonce",
        help=(
            "task1 (ID matching) の Loss 種別。"
            " infonce: 通常のInfoNCE (対角one-hot)。"
            " ngram_infonce: 文字N-gram BM25スコアをソフトラベルとして使うInfoNCE。"
            " tax_infonce: taxonomy距離をソフトラベルとして使うInfoNCE（Task3の信号をTask1に統合）。"
            "   ※ taxonomy データが必要。mode=multitask または --load_taxonomy_for_loss を使うこと。"
        ),
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=3,
        help="ngram_infonce で使う文字N-gramのN (default: 3)。",
    )
    parser.add_argument(
        "--ngram_temperature",
        type=float,
        default=1.0,
        help="ngram_infonce でBM25スコアをソフトラベルに変換するときのtemperature (default: 1.0)。",
    )
    parser.add_argument(
        "--bm25_k1",
        type=float,
        default=1.5,
        help="BM25のk1パラメータ (default: 1.5)。",
    )
    parser.add_argument(
        "--bm25_b",
        type=float,
        default=0.75,
        help="BM25のbパラメータ (default: 0.75)。",
    )
    # --- Dynamic task weighting ---
    # --- Projection heads ---
    parser.add_argument(
        "--use_proj_head_id",
        type=str2bool,
        default=False,
        help=(
            "Task1/評価にTask固有のProjection Head (2-layer MLP)を追加する。"
            " エンコーダはTask2/Task3で構造化され、proj_idがID識別空間に変換する。"
            " これによりTask2カテゴリ勾配とTask1の勾配干渉を防ぐ。"
        ),
    )
    parser.add_argument(
        "--proj_head_hidden_dim",
        type=int,
        default=0,
        help="ProjectionHeadの中間層次元。0の場合 hidden_size//2 を自動設定。",
    )
    parser.add_argument(
        "--cat_on_canonical",
        type=str2bool,
        default=False,
        help=(
            "Task2カテゴリ分類をquery埋め込みではなくcanonical埋め込みに適用する。"
            " queryのエンコーダ空間がカテゴリ勾配で汚染されるのを防ぎ、"
            " canonical側にカテゴリ構造を付与することでTask1の検索を補助する。"
        ),
    )
    parser.add_argument(
        "--tax_soft_temp",
        type=float,
        default=2.0,
        help=(
            "taxonomy_infonce でのtaxonomy距離→soft label変換のtemperature。"
            " 値が大きいほど遠い距離のitemに高い重みを付与（より緩やかな構造）。"
        ),
    )
    # --- Dynamic task weighting ---
    parser.add_argument(
        "--task_weighting",
        type=str,
        choices=["fixed", "uncertainty", "dwa"],
        default="fixed",
        help=(
            "マルチタスク損失の重み付け戦略。"
            " fixed: alpha/beta による固定重み（デフォルト・既存挙動を完全保持）。"
            " uncertainty: Kendall et al. (2018) 不確かさ重み付け（log_σ² を学習可能パラメータとして最適化）。"
            " dwa: Liu et al. (2019) Dynamic Weight Average（前エポック損失比に基づいて毎エポック重みを更新）。"
            " ※ active task が1つのみの場合は fixed にフォールバック。"
        ),
    )
    parser.add_argument(
        "--dwa_temperature",
        type=float,
        default=2.0,
        help="DWA の温度パラメータ T (softmax スケール。高いほど均等に近づく)。default: 2.0",
    )
    parser.add_argument(
        "--aux_warmup_epochs",
        type=int,
        default=0,
        help=(
            "補助タスク（Task2/3）の重みを最初の N エポックでウォームアップする（0=無効）。"
            "epoch=1 で weight=alpha/N, epoch=2 で weight=2*alpha/N, ..., epoch>=N で weight=alpha。"
            "マルチタスクにおいて Task1 が先に収束してから Task2/3 を徐々に導入するカリキュラム学習。"
        ),
    )
    return parser


def run_single_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)

    tasks = MODE_TO_TASKS[args.mode]
    mode_task_pattern = get_mode_task_pattern(args.mode)
    use_id = "id" in tasks
    use_cat = "cat" in tasks
    use_tax = "tax" in tasks
    # tax_infonce は taxonomy データが必要（use_tax でなくてもロードする）
    load_taxonomy = use_tax or args.loss_type == "tax_infonce"

    train_data_dir = args.train_data_dir
    pair_csv_path = train_data_dir / "ingredient_alias_pairs.csv"
    food2tax_path = train_data_dir / "food2tax.json"
    taxonomy_depth_path = train_data_dir / "taxonomy_depth.json"
    taxonomy_parent_path = train_data_dir / "taxonomy_parent.json"

    train_pairs = load_train_pairs(pair_csv_path, max_rows=args.max_train_rows, seed=args.seed)
    category_ids = sorted(train_pairs["food_table_category_id"].map(normalize_category_id).unique().tolist())
    category_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    idx_to_category = {idx: cat_id for cat_id, idx in category_to_idx.items()}
    dataset = AliasPairDataset(train_pairs, category_to_idx)
    if len(dataset) == 0:
        raise ValueError("Train dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=alias_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    canonical_names = sorted(train_pairs["food_table_name"].astype(str).str.strip().unique().tolist())
    foodtable_name2id_raw = load_json(args.foodtable_name2id_path)
    foodtable_name2id: dict[str, int] = {}
    for key, value in foodtable_name2id_raw.items():
        try:
            foodtable_name2id[str(key).strip()] = int(value)
        except (TypeError, ValueError):
            continue

    taxonomy_names_path = args.taxonomy_names_path or (train_data_dir / "taxonomy_names.json")
    pseudo_food2tax_path = args.pseudo_food2tax_path or (train_data_dir / "pseudo_food2tax.json")

    canonical_to_tax: dict[str, int] = {}
    canonical_link_weight: dict[str, float] = {}
    taxonomy_triplet_sampler: TaxonomyTripletSampler | None = None
    taxonomy_distance: TaxonomyDistance | None = None
    taxonomy_stats: dict[str, int] = {}
    pseudo_link_stats: dict[str, Any] = {}
    taxonomy_parent: dict[int, int] = {}
    taxonomy_depth: dict[int, int] = {}
    taxonomy_link_info: dict[str, TaxLinkInfo] = {}

    if load_taxonomy:
        food2tax_raw = load_json(food2tax_path)
        gold_canonical_to_tax, taxonomy_stats = build_canonical_to_tax_mapping(
            canonical_names=canonical_names,
            food2tax_raw=food2tax_raw,
            foodtable_name2id=foodtable_name2id,
        )
        taxonomy_parent_raw = load_json(taxonomy_parent_path)
        taxonomy_depth_raw = load_json(taxonomy_depth_path)
        taxonomy_parent = {int(k): int(v) for k, v in taxonomy_parent_raw.items()}
        taxonomy_depth = {int(k): int(v) for k, v in taxonomy_depth_raw.items()}
    else:
        gold_canonical_to_tax = {}

    encoder_name = resolve_encoder_name(args.encoder)
    device = choose_device(args.device)
    use_e5_prefix = resolve_use_e5_prefix(args.use_e5_prefix, encoder_name)
    encoder_trust_remote_code = resolve_encoder_trust_remote_code(args.encoder)
    tokenizer_kwargs = resolve_encoder_tokenizer_kwargs(args.encoder)
    encoder_torch_dtype = resolve_encoder_torch_dtype(args.encoder_torch_dtype, encoder_name, device)
    encoder_dtype_name = str(encoder_torch_dtype).replace("torch.", "") if encoder_torch_dtype is not None else "float32"

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        trust_remote_code=encoder_trust_remote_code,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model_init_kwargs: dict[str, Any] = {"trust_remote_code": encoder_trust_remote_code}
    if encoder_torch_dtype is not None:
        model_init_kwargs["torch_dtype"] = encoder_torch_dtype
    if args.use_lora:
        model_init_kwargs["low_cpu_mem_usage"] = True
    requested_device_map = args.model_device_map.strip().lower()
    if requested_device_map != "none":
        if not torch.cuda.is_available():
            print(f"[ModelShard] Ignoring model_device_map={requested_device_map} because CUDA is unavailable.")
        else:
            model_init_kwargs["device_map"] = requested_device_map
            if args.max_memory_per_gpu_gib > 0:
                model_init_kwargs["max_memory"] = {
                    gpu_idx: f"{int(args.max_memory_per_gpu_gib)}GiB"
                    for gpu_idx in range(torch.cuda.device_count())
                }
    encoder = AutoModel.from_pretrained(encoder_name, **model_init_kwargs)
    encoder, lora_runtime = maybe_apply_lora_to_encoder(
        encoder=encoder,
        args=args,
        encoder_name=encoder_name,
    )
    encoder_device_map = get_hf_device_map(encoder)
    if encoder_device_map:
        device = resolve_model_input_device(encoder, device)
    else:
        encoder.to(device)
    print(
        f"[Encoder] alias={args.encoder} name={encoder_name} dtype={encoder_dtype_name} "
        f"trust_remote_code={encoder_trust_remote_code} use_e5_prefix={use_e5_prefix} "
        f"lora={bool(lora_runtime.get('enabled', False))}"
    )
    if encoder_device_map:
        used_devices = sorted({str(v) for v in encoder_device_map.values()})
        print(f"[ModelShard] device_map={requested_device_map} targets={used_devices} input_device={device}")

    if load_taxonomy:
        if args.enable_pseudo_link:
            if pseudo_food2tax_path.exists():
                taxonomy_link_info, pseudo_link_stats = load_cached_tax_link_info(
                    cache_path=pseudo_food2tax_path,
                    canonical_names=canonical_names,
                    gold_canonical_to_tax=gold_canonical_to_tax,
                )
                print(
                    "[PseudoLink] loaded cache={path} gold={gold} pseudo={pseudo} excluded={excluded}".format(
                        path=pseudo_food2tax_path,
                        gold=pseudo_link_stats.get("gold_linked", 0),
                        pseudo=pseudo_link_stats.get("pseudo_linked", 0),
                        excluded=pseudo_link_stats.get("excluded_unlinked", 0),
                    )
                )
            else:
                encoder.eval()
                taxonomy_link_info, pseudo_link_stats = build_pseudo_tax_links(
                    canonical_names=canonical_names,
                    gold_canonical_to_tax=gold_canonical_to_tax,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    device=device,
                    max_len=args.max_len,
                    pooling=args.pooling,
                    normalize=args.normalize,
                    use_e5_prefix=use_e5_prefix,
                    taxonomy_parent=taxonomy_parent,
                    taxonomy_names_path=taxonomy_names_path,
                    nodes_dmp_path=args.nodes_dmp_path,
                    restrict_mode=args.taxonomy_restrict_mode,
                    taxonomy_max_nodes=args.taxonomy_max_nodes,
                    pseudo_link_threshold=args.pseudo_link_threshold,
                    pseudo_link_batch_size=max(1, args.pseudo_link_batch_size),
                )
                json_dump(pseudo_food2tax_path, serialize_tax_link_info(taxonomy_link_info))
                print(
                    "[PseudoLink] gold={gold} pseudo={pseudo} excluded={excluded} nodes={nodes} "
                    "restrict={mode} threshold={thr:.2f}".format(
                        gold=pseudo_link_stats.get("gold_linked", 0),
                        pseudo=pseudo_link_stats.get("pseudo_linked", 0),
                        excluded=pseudo_link_stats.get("excluded_unlinked", 0),
                        nodes=pseudo_link_stats.get("node_text_count", 0),
                        mode=args.taxonomy_restrict_mode,
                        thr=args.pseudo_link_threshold,
                    )
                )
        else:
            taxonomy_link_info = {
                canonical: TaxLinkInfo(tax_id=int(tax_id), score=1.0, source="gold")
                for canonical, tax_id in gold_canonical_to_tax.items()
            }
            pseudo_link_stats = {
                "gold_linked": int(len(gold_canonical_to_tax)),
                "pseudo_linked": 0,
                "excluded_unlinked": int(len(canonical_names) - len(gold_canonical_to_tax)),
                "disabled": True,
            }

        canonical_to_tax = {name: info.tax_id for name, info in taxonomy_link_info.items()}
        canonical_link_weight = {
            name: (1.0 if info.source == "gold" else float(args.pseudo_link_weight))
            for name, info in taxonomy_link_info.items()
        }

        taxonomy_distance = TaxonomyDistance(
            taxonomy_parent=taxonomy_parent,
            taxonomy_depth=taxonomy_depth,
        )
        # taxonomy triplet sampler は use_tax の場合のみ（tax_infonce のみの場合は不要）
        if use_tax:
            taxonomy_triplet_sampler = TaxonomyTripletSampler(
                canonical_names=canonical_names,
                canonical_to_tax=canonical_to_tax,
                canonical_link_weight=canonical_link_weight,
                taxonomy_distance=taxonomy_distance,
                soft_pos_hop=args.soft_pos_hop,
                neg_hop=args.neg_hop,
                seed=args.seed,
            )
            if taxonomy_triplet_sampler.num_anchors == 0:
                raise ValueError("Taxonomy task is enabled but no canonical has tax_id mapping.")
            if args.precompute_tax_cache:
                taxonomy_triplet_sampler.maybe_precompute()
        else:
            # tax_infonce のみの場合: taxonomy_distance はあるが triplet sampler は不要
            taxonomy_triplet_sampler = None

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    use_data_parallel = False
    if (
        args.multi_gpu
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and not encoder_device_map
        and (args.device.startswith("cuda") or args.device in {"auto", "cuda"})
    ):
        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) > 1:
            encoder = nn.DataParallel(encoder, device_ids=gpu_ids, output_device=gpu_ids[0])
            device = torch.device(f"cuda:{gpu_ids[0]}")
            use_data_parallel = True
            print(f"[MultiGPU] DataParallel enabled on logical GPU ids: {gpu_ids}")

    if not encoder_device_map:
        encoder.to(device)
    hidden_size = resolve_hidden_size(encoder)
    classifier = CategoryClassifier(hidden_size=hidden_size, num_classes=len(category_to_idx), dropout=args.dropout)
    classifier.to(device)

    # ---- Projection Head (use_proj_head_id) -----------------------------------
    # タスク固有の投影ヘッドにより、エンコーダ共有表現をTask1専用の識別空間に変換する。
    # これにより Task2(カテゴリ) / Task3(taxonomy) の勾配が Task1 の識別能力を損なわなくなる。
    proj_id: ProjectionHead | None = None
    if args.use_proj_head_id and use_id:
        proj_hidden = args.proj_head_hidden_dim if args.proj_head_hidden_dim > 0 else max(hidden_size // 2, 256)
        proj_id = ProjectionHead(hidden_size, hidden_size, proj_hidden).to(device)
        print(
            f"[ProjHead] proj_id enabled: in_dim={hidden_size} hidden_dim={proj_hidden} out_dim={hidden_size}"
        )
    # --------------------------------------------------------------------------

    # ---- Dynamic task-weighting setup ----------------------------------------
    # Tasks ordered consistently for the weighter (id first, then cat, tax).
    active_task_names: list[str] = [t for t in ["id", "cat", "tax"] if t in tasks]
    # Fixed weights reproduce the original alpha/beta behaviour exactly.
    fixed_task_weights: dict[str, float] = {"id": 1.0, "cat": args.alpha, "tax": args.beta}

    uncertainty_weighter: UncertaintyWeighter | None = None
    dwa_weighter: DWAWeighter | None = None

    if args.task_weighting == "uncertainty" and len(active_task_names) > 1:
        uncertainty_weighter = UncertaintyWeighter(active_task_names)
        uncertainty_weighter.to(device)
        print(f"[TaskWeighting] uncertainty  tasks={active_task_names}  (log_σ² learnable)")
    elif args.task_weighting == "dwa" and len(active_task_names) > 1:
        dwa_weighter = DWAWeighter(
            task_names=active_task_names,
            fixed_weights=fixed_task_weights,
            temperature=args.dwa_temperature,
        )
        print(f"[TaskWeighting] DWA  tasks={active_task_names}  T={args.dwa_temperature}  (fixed for first 2 epochs)")
    else:
        if args.task_weighting not in {"fixed", "uncertainty", "dwa"}:
            print(f"[TaskWeighting] Unknown method {args.task_weighting!r}; using fixed.")
        elif args.task_weighting != "fixed" and len(active_task_names) <= 1:
            print(f"[TaskWeighting] {args.task_weighting!r} requires ≥2 active tasks; using fixed.")
        print(f"[TaskWeighting] fixed  tasks={active_task_names}  weights={fixed_task_weights}")
    # --------------------------------------------------------------------------

    params = list(encoder.parameters()) + list(classifier.parameters())
    if uncertainty_weighter is not None:
        params += list(uncertainty_weighter.parameters())
    if proj_id is not None:
        params += list(proj_id.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    fp16_enabled = bool(args.fp16 and device.type == "cuda")
    if fp16_enabled and encoder_torch_dtype == torch.bfloat16:
        print("[INFO] encoder_torch_dtype=bfloat16 のため fp16 autocast を無効化します。")
        fp16_enabled = False
    scaler = GradScaler(enabled=fp16_enabled)

    run_dir = create_run_dir(
        output_root=args.output_root,
        mode=args.mode,
        encoder_tag=args.encoder,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        loss_type=args.loss_type,
    )
    train_log_path = run_dir / "train_log.jsonl"
    wandb_module, wandb_run = init_wandb_run(
        args=args,
        encoder_name=encoder_name,
        run_dir=run_dir,
    )

    config_payload = {
        "datetime": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "mode": args.mode,
        "task_pattern": mode_task_pattern,
        "encoder_name": encoder_name,
        "encoder_alias": args.encoder,
        "encoder_torch_dtype": encoder_dtype_name,
        "use_e5_prefix_resolved": use_e5_prefix,
        "lora_runtime": lora_runtime,
        "tasks": sorted(list(tasks)),
        "device": str(device),
        "multi_gpu": use_data_parallel,
        "gpu_ids": gpu_ids,
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "beta": args.beta,
            "max_len": args.max_len,
            "pooling": args.pooling,
            "normalize": args.normalize,
            "score_fn": args.score_fn,
            "temperature": args.temperature,
            "triplet_margin": args.triplet_margin,
            "soft_pos_hop": args.soft_pos_hop,
            "neg_hop": args.neg_hop,
            "tax_batch_size": args.tax_batch_size,
            "enable_pseudo_link": args.enable_pseudo_link,
            "pseudo_link_threshold": args.pseudo_link_threshold,
            "pseudo_link_weight": args.pseudo_link_weight,
            "taxonomy_restrict_mode": args.taxonomy_restrict_mode,
            "taxonomy_max_nodes": args.taxonomy_max_nodes,
            "pseudo_link_batch_size": args.pseudo_link_batch_size,
            "fp16_requested": args.fp16,
            "fp16_enabled": fp16_enabled,
            "use_e5_prefix_requested": args.use_e5_prefix,
            "use_e5_prefix_resolved": use_e5_prefix,
            "encoder_torch_dtype": args.encoder_torch_dtype,
            "encoder_torch_dtype_resolved": encoder_dtype_name,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "lora_bias": args.lora_bias,
            "lora_gradient_checkpointing": args.lora_gradient_checkpointing,
            "use_hard_negatives": args.use_hard_negatives,
            "eval_topk": args.eval_topk,
            "loss_type": args.loss_type,
            "ngram_n": args.ngram_n,
            "ngram_temperature": args.ngram_temperature,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "task_weighting": args.task_weighting,
            "dwa_temperature": args.dwa_temperature,
            "use_proj_head_id": args.use_proj_head_id,
            "proj_head_hidden_dim": args.proj_head_hidden_dim,
            "cat_on_canonical": args.cat_on_canonical,
            "tax_soft_temp": args.tax_soft_temp,
            "save_eval_embeddings": args.save_eval_embeddings,
            "use_wandb": args.use_wandb,
            "wandb_log_interval": args.wandb_log_interval,
            "wandb_run_name": args.wandb_run_name,
            "wandb_group": args.wandb_group,
            "run_task1_mode_sweep": args.run_task1_mode_sweep,
            "env_file": str(args.env_file),
        },
        "dataset": {
            "train_pair_rows": int(len(train_pairs)),
            "train_unique_canonical": int(len(canonical_names)),
            "train_num_categories": int(len(category_to_idx)),
            "taxonomy_mapped_canonical": int(len(canonical_to_tax)),
            "taxonomy_gold_linked_canonical": int(
                sum(1 for info in taxonomy_link_info.values() if info.source == "gold")
            ),
            "taxonomy_pseudo_linked_canonical": int(
                sum(1 for info in taxonomy_link_info.values() if info.source == "pseudo")
            ),
        },
        "taxonomy_mapping_stats": taxonomy_stats,
        "pseudo_link_stats": pseudo_link_stats,
        "paths": {
            "train_data_dir": str(train_data_dir),
            "pair_csv": str(pair_csv_path),
            "food2tax": str(food2tax_path),
            "pseudo_food2tax": str(pseudo_food2tax_path),
            "taxonomy_depth": str(taxonomy_depth_path),
            "taxonomy_parent": str(taxonomy_parent_path),
            "taxonomy_names": str(taxonomy_names_path),
            "nodes_dmp": str(args.nodes_dmp_path),
            "eval_csv": str(args.eval_path),
            "foodtable_name2id": str(args.foodtable_name2id_path),
            "output_root": str(args.output_root),
            "run_dir": str(run_dir),
        },
        "args": to_jsonable(vars(args)),
    }
    json_dump(run_dir / "config.json", config_payload)
    run_summary: dict[str, Any] = {
        "mode": args.mode,
        "task_pattern": mode_task_pattern,
        "encoder": args.encoder,
        "encoder_name": encoder_name,
        "encoder_torch_dtype": encoder_dtype_name,
        "use_e5_prefix": use_e5_prefix,
        "lora_enabled": bool(args.use_lora),
        "tasks": sorted(list(tasks)),
        "run_dir": str(run_dir),
        "eval_metrics": None,
    }

    try:
        if args.use_hard_negatives:
            print("[WARN] --use_hard_negatives は拡張フックのみ実装済みで、現状は in-batch negatives を使用します。")

        train_steps_default = len(loader) if (use_id or use_cat) else max(
            1, (taxonomy_triplet_sampler.num_anchors // max(1, args.tax_batch_size)) if taxonomy_triplet_sampler else 1
        )
        train_steps = args.train_steps_per_epoch if args.train_steps_per_epoch > 0 else train_steps_default
        global_step = 0
        # epoch_weights: effective weight applied to each task this epoch.
        # Initialised to fixed weights; DWA will update it each epoch.
        epoch_weights: dict[str, float] = dict(fixed_task_weights)

        with train_log_path.open("w", encoding="utf-8") as train_log_f:
            for epoch in range(1, args.epochs + 1):
                # ---- Compute DWA weights for this epoch (before any gradient steps) ----
                if dwa_weighter is not None:
                    epoch_weights = dwa_weighter.compute_weights()
                    print(
                        f"[DWA] Epoch {epoch} weights: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in epoch_weights.items())
                    )
                # ---- Auxiliary task warmup: linearly scale cat/tax weights ----
                if args.aux_warmup_epochs > 0 and len(active_task_names) > 1:
                    warmup_scale = min(1.0, epoch / args.aux_warmup_epochs)
                    epoch_weights = {
                        k: (v * warmup_scale if k != "id" else v)
                        for k, v in epoch_weights.items()
                    }
                    if epoch <= args.aux_warmup_epochs:
                        print(
                            f"[AuxWarmup] Epoch {epoch}/{args.aux_warmup_epochs} "
                            f"scale={warmup_scale:.2f} "
                            + ", ".join(f"{k}={v:.4f}" for k, v in epoch_weights.items())
                        )
                # --------------------------------------------------------------------------
                encoder.train()
                classifier.train()
                if proj_id is not None:
                    proj_id.train()
                if uncertainty_weighter is not None:
                    uncertainty_weighter.train()
                id_losses: list[float] = []
                cat_losses: list[float] = []
                tax_losses: list[float] = []
                total_losses: list[float] = []
                id_acc_values: list[float] = []
                cat_acc_values: list[float] = []
                tax_pos_rate_values: list[float] = []

                loader_iter = iter(loader)
                for step in range(1, train_steps + 1):
                    optimizer.zero_grad(set_to_none=True)

                    batch = None
                    if use_id or use_cat:
                        try:
                            batch = next(loader_iter)
                        except StopIteration:
                            loader_iter = iter(loader)
                            batch = next(loader_iter)

                    with autocast(enabled=fp16_enabled):
                        loss_id = None
                        loss_cat = None
                        loss_tax = None
                        id_top1_acc_step = None
                        category_acc_step = None
                        taxonomy_pos_rate_step = None

                        query_embeddings = None
                        if batch is not None and (use_id or use_cat):
                            query_embeddings = encode_text_batch(
                                encoder=encoder,
                                tokenizer=tokenizer,
                                texts=batch["food_names"],
                                max_len=args.max_len,
                                device=device,
                                pooling=args.pooling,
                                normalize=args.normalize,
                                text_type="query",
                                use_e5_prefix=use_e5_prefix,
                            )

                        if use_id and batch is not None:
                            canonical_embeddings = encode_text_batch(
                                encoder=encoder,
                                tokenizer=tokenizer,
                                texts=batch["food_table_names"],
                                max_len=args.max_len,
                                device=device,
                                pooling=args.pooling,
                                normalize=args.normalize,
                                text_type="passage",
                                use_e5_prefix=use_e5_prefix,
                            )
                            # --- proj_head_id: Task1 専用投影ヘッドを適用 ---
                            if proj_id is not None:
                                q_emb_t1 = F.normalize(proj_id(query_embeddings), p=2, dim=1)
                                c_emb_t1 = F.normalize(proj_id(canonical_embeddings), p=2, dim=1)
                            else:
                                q_emb_t1 = query_embeddings
                                c_emb_t1 = canonical_embeddings
                            logits = similarity_matrix(q_emb_t1, c_emb_t1, score_fn=args.score_fn)
                            logits = logits / max(args.temperature, 1e-6)
                            labels = torch.arange(logits.size(0), device=logits.device)
                            if args.loss_type == "ngram_infonce":
                                loss_id = compute_ngram_infonce_loss(
                                    logits=logits,
                                    query_texts=batch["food_names"],
                                    canonical_texts=batch["food_table_names"],
                                    ngram_n=args.ngram_n,
                                    ngram_temperature=args.ngram_temperature,
                                    bm25_k1=args.bm25_k1,
                                    bm25_b=args.bm25_b,
                                )
                            elif args.loss_type == "tax_infonce" and canonical_to_tax and taxonomy_distance is not None:
                                # Taxonomy-aware soft-label InfoNCE:
                                # バッチ内の canonical 間の taxonomy 距離をソフトラベルに変換する
                                loss_id = compute_taxonomy_infonce_loss(
                                    logits=logits,
                                    batch_canonicals=batch["food_table_names"],
                                    canonical_to_tax=canonical_to_tax,
                                    taxonomy_distance=taxonomy_distance,
                                    tax_soft_temp=args.tax_soft_temp,
                                )
                            else:
                                loss_id = F.cross_entropy(logits, labels)
                            id_top1_acc_step = float((logits.argmax(dim=1) == labels).float().mean().detach().item())

                        if use_cat and batch is not None:
                            category_labels = batch["category_idx"].to(device)
                            # --- cat_on_canonical: カテゴリ分類をcanonical埋め込みに適用 ---
                            # query空間へのカテゴリ勾配干渉を防ぐ。canonical側にカテゴリ構造を付与。
                            if args.cat_on_canonical and canonical_embeddings is not None:
                                cat_input = canonical_embeddings
                            else:
                                cat_input = query_embeddings
                            class_logits = classifier(cat_input)
                            loss_cat = F.cross_entropy(class_logits, category_labels)
                            category_acc_step = float(
                                (class_logits.argmax(dim=1) == category_labels).float().mean().detach().item()
                            )

                        if use_tax and taxonomy_triplet_sampler is not None:
                            anchor_idx, pos_idx, neg_idx, pos_weights = taxonomy_triplet_sampler.sample_triplets(
                                batch_size=max(1, args.tax_batch_size)
                            )
                            if anchor_idx:
                                anchor_names = [canonical_names[idx] for idx in anchor_idx]
                                pos_names = [canonical_names[idx] for idx in pos_idx]
                                neg_names = [canonical_names[idx] for idx in neg_idx]

                                anchor_embeddings = encode_text_batch(
                                    encoder=encoder,
                                    tokenizer=tokenizer,
                                    texts=anchor_names,
                                    max_len=args.max_len,
                                    device=device,
                                    pooling=args.pooling,
                                    normalize=args.normalize,
                                    text_type="passage",
                                    use_e5_prefix=use_e5_prefix,
                                )
                                pos_embeddings = encode_text_batch(
                                    encoder=encoder,
                                    tokenizer=tokenizer,
                                    texts=pos_names,
                                    max_len=args.max_len,
                                    device=device,
                                    pooling=args.pooling,
                                    normalize=args.normalize,
                                    text_type="passage",
                                    use_e5_prefix=use_e5_prefix,
                                )
                                neg_embeddings = encode_text_batch(
                                    encoder=encoder,
                                    tokenizer=tokenizer,
                                    texts=neg_names,
                                    max_len=args.max_len,
                                    device=device,
                                    pooling=args.pooling,
                                    normalize=args.normalize,
                                    text_type="passage",
                                    use_e5_prefix=use_e5_prefix,
                                )
                                sim_pos = pair_similarity(anchor_embeddings, pos_embeddings, score_fn=args.score_fn)
                                sim_neg = pair_similarity(anchor_embeddings, neg_embeddings, score_fn=args.score_fn)
                                taxonomy_pos_rate_step = float(
                                    (sim_pos > sim_neg).float().mean().detach().item()
                                )
                                loss_tax = compute_triplet_loss(
                                    anchor_embeddings=anchor_embeddings,
                                    pos_embeddings=pos_embeddings,
                                    neg_embeddings=neg_embeddings,
                                    pos_weights=pos_weights.to(device),
                                    margin=args.triplet_margin,
                                    score_fn=args.score_fn,
                                )

                        # ---- Collect per-task losses and combine with weighter ----
                        step_losses: dict[str, torch.Tensor] = {}
                        if loss_id is not None:
                            step_losses["id"] = loss_id
                        if loss_cat is not None:
                            step_losses["cat"] = loss_cat
                        if loss_tax is not None:
                            step_losses["tax"] = loss_tax

                        total_loss: torch.Tensor | None = None
                        if step_losses:
                            if uncertainty_weighter is not None:
                                total_loss, _ = uncertainty_weighter(step_losses)
                            elif dwa_weighter is not None:
                                total_loss = DWAWeighter.apply(step_losses, epoch_weights)
                            else:
                                # fixed weighting — uses epoch_weights (supports aux_warmup_epochs)
                                for name, loss in step_losses.items():
                                    w = epoch_weights.get(name, 1.0)
                                    weighted_loss = w * loss
                                    total_loss = (
                                        weighted_loss if total_loss is None
                                        else total_loss + weighted_loss
                                    )
                        # -----------------------------------------------------------

                    if total_loss is None:
                        continue

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1
                    total_loss_value = float(total_loss.detach().item())
                    total_losses.append(total_loss_value)
                    if loss_id is not None:
                        id_losses.append(float(loss_id.detach().item()))
                    if loss_cat is not None:
                        cat_losses.append(float(loss_cat.detach().item()))
                    if loss_tax is not None:
                        tax_losses.append(float(loss_tax.detach().item()))
                    if id_top1_acc_step is not None:
                        id_acc_values.append(id_top1_acc_step)
                    if category_acc_step is not None:
                        cat_acc_values.append(category_acc_step)
                    if taxonomy_pos_rate_step is not None:
                        tax_pos_rate_values.append(taxonomy_pos_rate_step)

                    if wandb_module is not None and max(1, args.wandb_log_interval) > 0:
                        if global_step % max(1, args.wandb_log_interval) == 0:
                            step_payload: dict[str, Any] = {
                                "loss/total": total_loss_value,
                                "train/epoch": epoch,
                                "train/step_in_epoch": step,
                            }
                            if loss_id is not None:
                                step_payload["loss/id_match"] = float(loss_id.detach().item())
                            if loss_cat is not None:
                                step_payload["loss/category"] = float(loss_cat.detach().item())
                            if loss_tax is not None:
                                step_payload["loss/taxonomy"] = float(loss_tax.detach().item())
                            if id_top1_acc_step is not None:
                                step_payload["train/id_top1_acc"] = id_top1_acc_step
                            if category_acc_step is not None:
                                step_payload["train/category_acc"] = category_acc_step
                            if taxonomy_pos_rate_step is not None:
                                step_payload["train/taxonomy_pos_rate"] = taxonomy_pos_rate_step
                            wandb_module.log(step_payload, step=global_step)

                epoch_log = {
                    "epoch": epoch,
                    "datetime": datetime.now().isoformat(),
                    "steps": train_steps,
                    "loss_total": safe_mean(total_losses),
                    "loss_task1_id": safe_mean(id_losses),
                    "loss_task2_cat": safe_mean(cat_losses),
                    "loss_task3_tax": safe_mean(tax_losses),
                    "train_id_top1_acc": safe_mean(id_acc_values),
                    "train_category_acc": safe_mean(cat_acc_values),
                    "train_taxonomy_pos_rate": safe_mean(tax_pos_rate_values),
                }
                print(
                    f"[Epoch {epoch}/{args.epochs}] total={epoch_log['loss_total']:.4f} "
                    f"id={epoch_log['loss_task1_id']:.4f} cat={epoch_log['loss_task2_cat']:.4f} "
                    f"tax={epoch_log['loss_task3_tax']:.4f}"
                )
                # ---- Post-epoch: update DWA history / log uncertainty weights ----
                if dwa_weighter is not None:
                    dwa_weighter.record_epoch({
                        "id": safe_mean(id_losses),
                        "cat": safe_mean(cat_losses),
                        "tax": safe_mean(tax_losses),
                    })
                    epoch_log["dwa_weights"] = dwa_weighter.compute_weights()
                if uncertainty_weighter is not None:
                    epoch_log["uncertainty_log_vars"] = uncertainty_weighter.log_var_dict()
                    epoch_log["uncertainty_effective_weights"] = {
                        name: round(float(torch.exp(-uncertainty_weighter.log_vars[name]).detach().item()), 6)
                        for name in uncertainty_weighter.task_names
                    }
                # -------------------------------------------------------------------
                train_log_f.write(json.dumps(epoch_log, ensure_ascii=False) + "\n")
                train_log_f.flush()

                if wandb_module is not None:
                    epoch_payload: dict[str, Any] = {
                        "train/epoch": epoch,
                        "loss/total": epoch_log["loss_total"],
                    }
                    if use_id:
                        epoch_payload["loss/id_match"] = epoch_log["loss_task1_id"]
                    if use_cat:
                        epoch_payload["loss/category"] = epoch_log["loss_task2_cat"]
                    if use_tax:
                        epoch_payload["loss/taxonomy"] = epoch_log["loss_task3_tax"]
                    if id_acc_values:
                        epoch_payload["train/id_top1_acc"] = epoch_log["train_id_top1_acc"]
                    if cat_acc_values:
                        epoch_payload["train/category_acc"] = epoch_log["train_category_acc"]
                    if tax_pos_rate_values:
                        epoch_payload["train/taxonomy_pos_rate"] = epoch_log["train_taxonomy_pos_rate"]
                    wandb_module.log(epoch_payload, step=global_step)

        eval_frame = load_eval_ingredients(args.eval_path, max_items=args.eval_max_items)
        print(
            f"[EvalData] path={args.eval_path} rows={len(eval_frame)} "
            f"unique_ingredient={int(eval_frame['ingredient'].nunique())}"
        )
        eval_need_per_item = args.save_per_item or (wandb_module is not None)
        eval_result = evaluate_retrieval(
            encoder=encoder,
            tokenizer=tokenizer,
            canonical_names=canonical_names,
            eval_frame=eval_frame,
            foodtable_name2id=foodtable_name2id,
            device=device,
            max_len=args.max_len,
            pooling=args.pooling,
            normalize=args.normalize,
            score_fn=args.score_fn,
            use_e5_prefix=use_e5_prefix,
            eval_batch_size=args.eval_batch_size,
            eval_topk=args.eval_topk,
            per_item=eval_need_per_item,
            save_embeddings=args.save_eval_embeddings,
            proj_head=proj_id,
        )
        saved_artifacts: dict[str, str] = {}
        for artifact_name, artifact_payload in eval_result.get("artifacts", {}).items():
            artifact_filename = f"{artifact_name}.pt"
            torch.save(artifact_payload, run_dir / artifact_filename)
            saved_artifacts[artifact_name] = artifact_filename
        eval_payload = {
            "meta": {
                "datetime": datetime.now().isoformat(),
                "git_commit": get_git_commit(),
                "mode": args.mode,
                "task_pattern": mode_task_pattern,
                "encoder_name": encoder_name,
                "encoder_torch_dtype": encoder_dtype_name,
                "use_e5_prefix_resolved": use_e5_prefix,
                "lora_runtime": lora_runtime,
                "hyperparams": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "eval_topk": args.eval_topk,
                    "max_len": args.max_len,
                    "pooling": args.pooling,
                    "normalize": args.normalize,
                    "score_fn": args.score_fn,
                    "temperature": args.temperature,
                    "triplet_margin": args.triplet_margin,
                    "soft_pos_hop": args.soft_pos_hop,
                    "neg_hop": args.neg_hop,
                    "tax_batch_size": args.tax_batch_size,
                    "enable_pseudo_link": args.enable_pseudo_link,
                    "pseudo_link_threshold": args.pseudo_link_threshold,
                    "pseudo_link_weight": args.pseudo_link_weight,
                    "taxonomy_restrict_mode": args.taxonomy_restrict_mode,
                    "taxonomy_max_nodes": args.taxonomy_max_nodes,
                    "pseudo_link_batch_size": args.pseudo_link_batch_size,
                    "fp16_requested": args.fp16,
                    "fp16_enabled": fp16_enabled,
                    "use_e5_prefix_requested": args.use_e5_prefix,
                    "use_e5_prefix_resolved": use_e5_prefix,
                    "encoder_torch_dtype": args.encoder_torch_dtype,
                    "encoder_torch_dtype_resolved": encoder_dtype_name,
                    "use_lora": args.use_lora,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "lora_bias": args.lora_bias,
                    "lora_gradient_checkpointing": args.lora_gradient_checkpointing,
                    "task_weighting": args.task_weighting,
                    "dwa_temperature": args.dwa_temperature,
                    "save_eval_embeddings": args.save_eval_embeddings,
                    "use_wandb": args.use_wandb,
                    "wandb_log_interval": args.wandb_log_interval,
                    "wandb_run_name": args.wandb_run_name,
                    "wandb_group": args.wandb_group,
                    "run_task1_mode_sweep": args.run_task1_mode_sweep,
                },
                "dataset_sizes": {
                    "train_pairs": int(len(train_pairs)),
                    "eval_unique_ingredients": int(len(eval_frame)),
                },
                "pseudo_link_stats": pseudo_link_stats,
            },
            "metrics": eval_result["metrics"],
        }
        if saved_artifacts:
            eval_payload["artifacts"] = saved_artifacts
        if args.save_per_item and "per_item" in eval_result:
            eval_payload["per_item"] = eval_result["per_item"]

        # ---- Save final task-weighting state ----
        if uncertainty_weighter is not None:
            eval_payload["meta"]["uncertainty_final_log_vars"] = uncertainty_weighter.log_var_dict()
            eval_payload["meta"]["uncertainty_final_weights"] = {
                name: round(float(torch.exp(-uncertainty_weighter.log_vars[name]).detach().item()), 6)
                for name in uncertainty_weighter.task_names
            }
        if dwa_weighter is not None:
            eval_payload["meta"]["dwa_weight_history"] = dwa_weighter.loss_history
            eval_payload["meta"]["dwa_final_weights"] = dwa_weighter.compute_weights()
        # -----------------------------------------

        json_dump(run_dir / "eval_result.json", eval_payload)

        if wandb_module is not None:
            eval_metrics = eval_result["metrics"]
            wandb_module.log(
                {
                    "eval/EM": eval_metrics["EM"],
                    "eval/CategoryMatch": eval_metrics["CategoryMatch"],
                    "eval/Acc@1_ID": eval_metrics["Acc@1_ID"],
                    "eval/Acc@10_ID": eval_metrics["Acc@10_ID"],
                    "eval/Acc@1_Category": eval_metrics["Acc@1_Category"],
                    "eval/Acc@10_Category": eval_metrics["Acc@10_Category"],
                    "eval/num_unmapped_pred": eval_metrics["num_unmapped_pred"],
                    "eval/num_eval_samples": int(len(eval_frame)),
                    "eval_EM": eval_metrics["EM"],
                    "eval_CategoryMatch": eval_metrics["CategoryMatch"],
                    "eval_Acc@1_ID": eval_metrics["Acc@1_ID"],
                    "eval_Acc@10_ID": eval_metrics["Acc@10_ID"],
                    "eval_Acc@1_Category": eval_metrics["Acc@1_Category"],
                    "eval_Acc@10_Category": eval_metrics["Acc@10_Category"],
                },
                step=global_step,
            )

            per_item_rows = eval_result.get("per_item", [])
            if per_item_rows:
                table = wandb_module.Table(
                    columns=[
                        "ingredient",
                        "gold_id",
                        "pred_canonical",
                        "pred_id",
                        "Acc@1_ID",
                        "Acc@10_ID",
                        "Acc@1_Category",
                        "Acc@10_Category",
                    ]
                )
                for row in per_item_rows[:1000]:
                    table.add_data(
                        row.get("ingredient"),
                        row.get("gold_id"),
                        row.get("pred_canonical"),
                        row.get("pred_id"),
                        row.get("acc1_id"),
                        row.get("acc10_id"),
                        row.get("acc1_category"),
                        row.get("acc10_category"),
                    )
                wandb_module.log({"eval/predictions": table}, step=global_step)

            task_table = wandb_module.Table(
                columns=["mode", "task_pattern", "active_tasks", "EM", "CategoryMatch", "notes"]
            )
            for mode_name in TASK1_MODE_SWEEP:
                if mode_name == args.mode:
                    task_table.add_data(
                        mode_name,
                        get_mode_task_pattern(mode_name),
                        ",".join(sorted(MODE_TO_TASKS[mode_name])),
                        eval_metrics["EM"],
                        eval_metrics["CategoryMatch"],
                        "active",
                    )
                else:
                    task_table.add_data(
                        mode_name,
                        get_mode_task_pattern(mode_name),
                        ",".join(sorted(MODE_TO_TASKS[mode_name])),
                        None,
                        None,
                        "not evaluated in this run",
                    )
            wandb_module.log({"eval/task_summary": task_table}, step=global_step)

        if args.save_model:
            encoder_to_save = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            checkpoint = {
                "encoder_name": encoder_name,
                "encoder_state_dict": encoder_to_save.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "category_to_idx": category_to_idx,
                "idx_to_category": idx_to_category,
                "canonical_names": canonical_names,
                "canonical_to_tax": canonical_to_tax,
                "canonical_link_weight": canonical_link_weight,
                "taxonomy_link_info": serialize_tax_link_info(taxonomy_link_info),
                "config": to_jsonable(vars(args)),
            }
            torch.save(checkpoint, run_dir / "model.pt")

        run_summary["eval_metrics"] = eval_payload["metrics"]
        print("Run directory:", run_dir)
        print("Eval metrics:", eval_payload["metrics"])
    finally:
        if wandb_module is not None and wandb_run is not None:
            try:
                wandb_module.finish()
            except Exception as exc:
                print(f"[WARN] wandb.finish() で例外が発生しました: {exc}")
    return run_summary


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.run_task1_mode_sweep:
        run_single_experiment(args)
        return

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_group = args.wandb_group or f"task1_mode_sweep_{sweep_id}_seed{args.seed}"
    summary: list[dict[str, Any]] = []
    print(
        f"[TaskSweep] Running {len(TASK1_MODE_SWEEP)} modes with fixed hyper-parameters: "
        + ", ".join(TASK1_MODE_SWEEP)
    )
    for mode_name in TASK1_MODE_SWEEP:
        run_args = argparse.Namespace(**vars(args))
        run_args.mode = mode_name
        run_args.run_task1_mode_sweep = True
        run_args.wandb_group = sweep_group
        if args.wandb_run_name:
            run_args.wandb_run_name = f"{args.wandb_run_name}_{mode_name}"
        print(
            f"[TaskSweep] Start mode={mode_name} task_pattern={get_mode_task_pattern(mode_name)} "
            f"group={sweep_group}"
        )
        summary.append(run_single_experiment(run_args))

    summary_payload = {
        "datetime": datetime.now().isoformat(),
        "sweep_group": sweep_group,
        "seed": args.seed,
        "modes": TASK1_MODE_SWEEP,
        "results": summary,
    }
    summary_path = Path(args.output_root) / f"{sanitize_token(sweep_group)}_summary.json"
    json_dump(summary_path, summary_payload)
    print(f"[TaskSweep] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
