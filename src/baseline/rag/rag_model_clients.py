from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_LLM_DIR = SCRIPT_DIR.parent / "llm"
BASELINE_EMBEDDING_DIR = SCRIPT_DIR.parent / "embedding"

for module_dir in (BASELINE_LLM_DIR, BASELINE_EMBEDDING_DIR):
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from model_clients import (  # type: ignore
    build_model_infer,
    list_model_keys as list_llm_model_keys,
    resolve_model_spec as resolve_llm_model_spec,
)
from embedding_model_defs import (  # type: ignore
    list_model_keys as list_embedding_model_keys,
    resolve_model_spec as resolve_embedding_model_spec,
)


@dataclass(frozen=True)
class RagPreset:
    key: str
    llm_model_key: str
    embedding_model_key: str


@dataclass(frozen=True)
class RagRuntimeSpec:
    llm_model_key: str
    llm_model_name: str
    llm_output_tag: str
    embedding_model_key: str
    embedding_model_name: str
    embedding_output_tag: str
    infer_fn: Callable[[str], str]


RAG_PRESETS: Dict[str, RagPreset] = {
    "gptoss_20b__qwen3_8b": RagPreset(
        key="gptoss_20b__qwen3_8b",
        llm_model_key="gptoss_20b",
        embedding_model_key="qwen3_8b",
    ),
    "llama3_1_8b__qwen3_8b": RagPreset(
        key="llama3_1_8b__qwen3_8b",
        llm_model_key="llama3_1_8b",
        embedding_model_key="qwen3_8b",
    ),
    "mistral_8b__qwen3_8b": RagPreset(
        key="mistral_8b__qwen3_8b",
        llm_model_key="mistral_8b",
        embedding_model_key="qwen3_8b",
    ),
    "gemma2_9b_hf__qwen3_8b": RagPreset(
        key="gemma2_9b_hf__qwen3_8b",
        llm_model_key="gemma2_9b_hf",
        embedding_model_key="qwen3_8b",
    ),
    "qwen2_5_32b__qwen3_8b": RagPreset(
        key="qwen2_5_32b__qwen3_8b",
        llm_model_key="qwen2_5_32b",
        embedding_model_key="qwen3_8b",
    ),
}


def list_rag_preset_keys() -> List[str]:
    return sorted(RAG_PRESETS.keys())


def resolve_rag_preset(preset_key: str) -> RagPreset:
    if preset_key not in RAG_PRESETS:
        available = ", ".join(list_rag_preset_keys())
        raise ValueError(f"Unknown RAG preset key: {preset_key}. Available: {available}")
    return RAG_PRESETS[preset_key]


def build_rag_runtime(
    llm_model_key: str,
    embedding_model_key: str,
    provider: str = "auto",
    llm_model_name_override: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_ctx: Optional[int] = None,
    seed: Optional[int] = None,
    num_predict: Optional[int] = None,
    gpu_id: Optional[int] = None,
    num_gpu: Optional[int] = None,
    parallel_gpu_ids: Optional[tuple[int, ...]] = None,
) -> RagRuntimeSpec:
    infer_fn, llm_spec, resolved_llm_model_name = build_model_infer(
        llm_model_key,
        provider=provider,
        model_name_override=llm_model_name_override,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
        gpu_id=gpu_id,
        num_gpu=num_gpu,
        parallel_gpu_ids=parallel_gpu_ids,
    )
    embedding_spec = resolve_embedding_model_spec(embedding_model_key)

    return RagRuntimeSpec(
        llm_model_key=llm_model_key,
        llm_model_name=resolved_llm_model_name,
        llm_output_tag=llm_spec.output_tag,
        embedding_model_key=embedding_model_key,
        embedding_model_name=embedding_spec.hf_model_name,
        embedding_output_tag=embedding_spec.output_tag,
        infer_fn=infer_fn,
    )


__all__ = [
    "RagPreset",
    "RagRuntimeSpec",
    "build_rag_runtime",
    "list_embedding_model_keys",
    "list_llm_model_keys",
    "list_rag_preset_keys",
    "resolve_embedding_model_spec",
    "resolve_llm_model_spec",
    "resolve_rag_preset",
]
