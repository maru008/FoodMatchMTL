from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingModelSpec:
    key: str
    hf_model_name: str
    output_tag: str
    backend: str
    default_device: Optional[str]
    tokenizer_kwargs: Dict[str, Any]
    max_length: int = 512
    pooling: str = "mean"


MODEL_SPECS: Dict[str, EmbeddingModelSpec] = {
    "qwen3_8b": EmbeddingModelSpec(
        key="qwen3_8b",
        hf_model_name="Qwen/Qwen3-Embedding-8B",
        output_tag="qwen3_8b",
        backend="sentence_transformers",
        default_device="cuda:1",
        tokenizer_kwargs={"padding_side": "left"},
    ),
    "sentence_bert": EmbeddingModelSpec(
        key="sentence_bert",
        hf_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        output_tag="sentence_bert",
        backend="sentence_transformers",
        default_device="cuda:1",
        tokenizer_kwargs={},
    ),
    "multilingual_e5_large": EmbeddingModelSpec(
        key="multilingual_e5_large",
        hf_model_name="intfloat/multilingual-e5-large",
        output_tag="multilingual_e5_large",
        backend="sentence_transformers",
        default_device="cuda:1",
        tokenizer_kwargs={},
    ),
    "xlm_roberta_large": EmbeddingModelSpec(
        key="xlm_roberta_large",
        hf_model_name="FacebookAI/xlm-roberta-large",
        output_tag="xlm_roberta_large",
        backend="transformers",
        default_device="cuda:1",
        tokenizer_kwargs={},
        max_length=64,
        pooling="mean",
    ),
    "mdeberta_v3_base": EmbeddingModelSpec(
        key="mdeberta_v3_base",
        hf_model_name="microsoft/mdeberta-v3-base",
        output_tag="mdeberta_v3_base",
        backend="transformers",
        default_device="cuda:1",
        tokenizer_kwargs={},
        max_length=64,
        pooling="mean",
    ),
}


def list_model_keys() -> List[str]:
    return sorted(MODEL_SPECS.keys())


def resolve_model_spec(model_key: str) -> EmbeddingModelSpec:
    if model_key not in MODEL_SPECS:
        available = ", ".join(list_model_keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available}")
    return MODEL_SPECS[model_key]


def build_embedder(
    model_key: str,
    device: Optional[str] = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> Tuple[Callable[[List[str]], Any], EmbeddingModelSpec, Optional[str]]:
    spec = resolve_model_spec(model_key)
    resolved_device = device if device is not None else spec.default_device

    if spec.backend == "sentence_transformers":
        return (
            _build_sentence_transformer_embedder(
                spec=spec,
                resolved_device=resolved_device,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            ),
            spec,
            resolved_device,
        )
    if spec.backend == "transformers":
        return (
            _build_transformers_embedder(
                spec=spec,
                resolved_device=resolved_device,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            ),
            spec,
            resolved_device,
        )
    raise ValueError(f"Unsupported backend: {spec.backend}")


def _build_sentence_transformer_embedder(
    *,
    spec: EmbeddingModelSpec,
    resolved_device: Optional[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> Callable[[List[str]], Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'sentence-transformers' package is required. "
            "Install it with: pip install sentence-transformers"
        ) from error

    model_init_kwargs: Dict[str, Any] = {}
    if resolved_device is not None:
        model_init_kwargs["device"] = resolved_device
    if spec.tokenizer_kwargs:
        model_init_kwargs["tokenizer_kwargs"] = spec.tokenizer_kwargs

    model = SentenceTransformer(spec.hf_model_name, **model_init_kwargs)

    def encode_texts(texts: List[str]) -> Any:
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        )

    return encode_texts


def _build_transformers_embedder(
    *,
    spec: EmbeddingModelSpec,
    resolved_device: Optional[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> Callable[[List[str]], Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'transformers' and 'torch' packages are required. "
            "Install them with: pip install transformers torch"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_name, **spec.tokenizer_kwargs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model = AutoModel.from_pretrained(spec.hf_model_name)

    target_device = resolved_device if resolved_device is not None else "cpu"
    torch_device = torch.device(target_device)
    model.to(torch_device)
    model.eval()

    def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> Any:
        if not texts:
            hidden_size = getattr(model.config, "hidden_size", 0)
            return np.empty((0, hidden_size), dtype=np.float32)

        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=spec.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(torch_device) for key, value in tokens.items()}
            last_hidden_state = model(**tokens).last_hidden_state
            if spec.pooling == "mean":
                embeddings = mean_pool(last_hidden_state, tokens["attention_mask"])
            elif spec.pooling == "cls":
                embeddings = last_hidden_state[:, 0]
            else:
                raise ValueError(f"Unsupported pooling: {spec.pooling}")

            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32))

        return np.vstack(outputs)

    return encode_texts
