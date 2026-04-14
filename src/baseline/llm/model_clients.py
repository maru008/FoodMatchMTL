from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    default_model_name: str
    output_tag: str


MODEL_SPECS: dict[str, ModelSpec] = {
    # HF (transformers) models
    # Note: Llama 3.1 has an 8B official instruct checkpoint on HF. We map the
    # requested "7b" slot to the closest widely-used 3.1 instruct model.
    "llama3_1_7b_hf": ModelSpec(
        key="llama3_1_7b_hf",
        provider="transformers",
        default_model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_tag="llama3_1_7b_hf",
    ),
    "mistral_hf": ModelSpec(
        key="mistral_hf",
        provider="transformers",
        default_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        output_tag="mistral_hf",
    ),
    # Gemma 2 9B Instruct is a close replacement in the same size band as
    # Mistral 7B while staying on a standard text-only HF CausalLM path.
    "gemma2_9b_hf": ModelSpec(
        key="gemma2_9b_hf",
        provider="transformers",
        default_model_name="google/gemma-2-9b-it",
        output_tag="gemma2_9b_hf",
    ),
    "qwen3_8b_hf": ModelSpec(
        key="qwen3_8b_hf",
        provider="transformers",
        default_model_name="Qwen/Qwen3-8B",
        output_tag="qwen3_8b_hf",
    ),
    "gptoss_20b": ModelSpec(
        key="gptoss_20b",
        provider="transformers",
        default_model_name="openai/gpt-oss-20b",
        output_tag="gptoss_20b",
    ),

    # Ollama models
    "llama3_3_latest": ModelSpec(
        key="llama3_3_latest",
        provider="ollama",
        default_model_name="llama3.3:latest",
        output_tag="llama3_3_latest",
    ),
    "mistral_latest": ModelSpec(
        key="mistral_latest",
        provider="ollama",
        default_model_name="mistral:latest",
        output_tag="mistral_latest",
    ),
    "llama4": ModelSpec(
        key="llama4",
        provider="ollama",
        default_model_name="llama4:latest",
        output_tag="llama4_latest",
    ),
    "llama3_1_8b": ModelSpec(
        key="llama3_1_8b",
        provider="ollama",
        default_model_name="llama3.1:8b",
        output_tag="llama3_1_8b",
    ),
    "mistral_8b": ModelSpec(
        key="mistral_8b",
        provider="ollama",
        default_model_name="mistral:8b",
        output_tag="mistral_8b",
    ),
    "qwen2_5_32b": ModelSpec(
        key="qwen2_5_32b",
        provider="ollama",
        default_model_name="qwen2.5:32b",
        output_tag="qwen2_5_32b",
    ),
    "deepseek_r1": ModelSpec(
        key="deepseek_r1",
        provider="ollama",
        default_model_name="yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b",
        output_tag="deepseek_r1",
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        provider="ollama",
        default_model_name="qwen3:8b",
        output_tag="qwen3_8b",
    ),
}


ROOT_DIR = Path(__file__).resolve().parents[3]
HF_TOKEN_CONFIG_PATH = ROOT_DIR / ".hf_access.env"
HF_OFFLOAD_ROOT = ROOT_DIR / ".hf_offload"

_TRANSFORMERS_CLIENT_CACHE: dict[
    tuple[str, tuple[int, ...], bool],
    tuple[Any, Any, Any, str],
] = {}


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        return text[1:-1].strip()
    return text


def _load_hf_token_from_config(path: Path) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if key in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
            return value
    return None


def _resolve_hf_token() -> str | None:
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env_value = os.getenv(env_key)
        if env_value:
            return env_value.strip()
    return _load_hf_token_from_config(HF_TOKEN_CONFIG_PATH)


def _is_gated_repo_error(error: Exception) -> bool:
    message = str(error).lower()
    gated_markers = [
        "gated repo",
        "cannot access gated repo",
        "access to model",
        "403",
        "authorized list",
    ]
    return any(marker in message for marker in gated_markers)


def _is_sentencepiece_error(error: Exception) -> bool:
    message = str(error).lower()
    return "sentencepiece" in message


def _needs_eager_attention(model_name: str) -> bool:
    lowered = model_name.lower()
    return "gemma" in lowered


def _uses_auto_torch_dtype(model_name: str) -> bool:
    lowered = model_name.lower()
    return "gpt-oss" in lowered


def _preferred_attention_implementation(model_name: str) -> str | None:
    lowered = model_name.lower()
    if "gpt-oss" in lowered:
        return "sdpa"
    if "gemma" in lowered:
        return "eager"
    return None


def _uses_harmony_channels(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


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
            raise ValueError(
                f"Invalid GPU id '{part}' in parallel GPU list: {value}"
            ) from error
        if gpu_id < 0:
            raise ValueError(f"GPU ids must be non-negative: {value}")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        return None
    return tuple(gpu_ids)


def _configure_torch_dynamo(torch_module) -> None:
    try:
        import torch._dynamo  # type: ignore
    except Exception:
        return
    torch_module._dynamo.config.suppress_errors = True


def _normalize_context_limit(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    if value >= 1_000_000_000:
        return None
    return value


def _resolve_context_limit(tokenizer: Any, model: Any) -> int | None:
    limits = [
        _normalize_context_limit(getattr(tokenizer, "model_max_length", None)),
        _normalize_context_limit(getattr(model.config, "max_position_embeddings", None)),
    ]
    valid_limits = [value for value in limits if value is not None]
    if not valid_limits:
        return None
    return min(valid_limits)


def _normalize_requested_gpu_ids(
    gpu_id: int | None,
    parallel_gpu_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if parallel_gpu_ids:
        return parallel_gpu_ids
    if gpu_id is None:
        return ()
    return (gpu_id,)


def _build_transformers_max_memory(torch_module, gpu_ids: tuple[int, ...]) -> dict[Any, str]:
    max_memory: dict[Any, str] = {"cpu": "128GiB"}
    for target_gpu in gpu_ids:
        total_bytes = int(torch_module.cuda.get_device_properties(target_gpu).total_memory)
        total_gib = max(1, total_bytes // (1024**3))
        usable_gib = max(8, total_gib - 6)
        max_memory[target_gpu] = f"{usable_gib}GiB"
    return max_memory


def _resolve_model_input_device(model: Any, fallback_device: str) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return f"cuda:{mapped_device}"
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                return mapped_device

    try:
        first_parameter = next(model.parameters())
    except StopIteration:
        return fallback_device
    return str(first_parameter.device)


def _build_chat_template_kwargs(model_name: str) -> dict[str, Any]:
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return {"enable_thinking": False}
    if _uses_harmony_channels(model_name):
        return {"reasoning_effort": "low"}
    return {}


def _extract_harmony_final_content(text: str) -> str | None:
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>)",
        text,
        flags=re.DOTALL,
    )
    if final_match:
        return final_match.group(1).strip()
    return None


def _decode_generated_content(tokenizer: Any, generated_ids: Any, model_name: str) -> str:
    if _uses_harmony_channels(model_name):
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        final_content = _extract_harmony_final_content(raw_text)
        if final_content:
            return final_content
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _build_ollama_options(
    temperature: float | None,
    top_p: float | None,
    num_ctx: int | None,
    seed: int | None,
    num_predict: int | None,
    gpu_id: int | None,
    num_gpu: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if seed is not None:
        options["seed"] = seed
    if num_predict is not None:
        options["num_predict"] = num_predict
    if gpu_id is not None:
        # Ollama option (llama.cpp backend): target main GPU index.
        options["main_gpu"] = gpu_id
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return options


def infer_with_ollama(
    prompt: str,
    *,
    model_name: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    gpu_id: int | None = None,
    num_gpu: int | None = None,
) -> str:
    try:
        import ollama
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'ollama' Python package is required for provider='ollama'. "
            "Install it with: pip install ollama"
        ) from error

    options = _build_ollama_options(
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
        gpu_id=gpu_id,
        num_gpu=num_gpu,
    )

    params: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
    }
    if options:
        params["options"] = options

    response = ollama.chat(**params)
    message = response.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama response does not include a text message content.")
    return content


def _resolve_transformers_device(
    torch_module,
    gpu_id: int | None,
    parallel_gpu_ids: tuple[int, ...] | None,
) -> str:
    if torch_module.cuda.is_available():
        normalized_gpu_ids = _normalize_requested_gpu_ids(gpu_id, parallel_gpu_ids)
        target_gpu = normalized_gpu_ids[0] if normalized_gpu_ids else 0
        return f"cuda:{target_gpu}"
    return "cpu"


def _load_transformers_client(
    model_name: str,
    gpu_id: int | None,
    parallel_gpu_ids: tuple[int, ...] | None,
) -> tuple[Any, Any, Any, str]:
    hf_token = _resolve_hf_token()
    normalized_gpu_ids = _normalize_requested_gpu_ids(gpu_id, parallel_gpu_ids)
    cache_key = (model_name, normalized_gpu_ids, hf_token is not None)
    if cache_key in _TRANSFORMERS_CLIENT_CACHE:
        return _TRANSFORMERS_CLIENT_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'transformers' and 'torch' packages are required for provider='transformers'. "
            "Install with: pip install transformers torch"
        ) from error

    _configure_torch_dynamo(torch)
    device = _resolve_transformers_device(torch, gpu_id, normalized_gpu_ids)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    use_model_parallel = device.startswith("cuda") and len(normalized_gpu_ids) > 1
    if device.startswith("cuda"):
        if _uses_auto_torch_dtype(model_name):
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16
    if use_model_parallel:
        HF_OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = _build_transformers_max_memory(torch, normalized_gpu_ids)
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["offload_folder"] = str(HF_OFFLOAD_ROOT / model_name.replace("/", "__"))
    preferred_attention = _preferred_attention_implementation(model_name)
    if preferred_attention is not None:
        model_kwargs["attn_implementation"] = preferred_attention

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            **model_kwargs,
        )
    except Exception as error:
        if _is_gated_repo_error(error):
            raise RuntimeError(
                "Failed to access a gated Hugging Face model. "
                f"Set HF token via env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN) "
                f"or root config file: {HF_TOKEN_CONFIG_PATH} "
                "(example: HF_TOKEN=hf_xxx)."
            ) from error
        if _is_sentencepiece_error(error):
            raise RuntimeError(
                "This Hugging Face tokenizer requires sentencepiece. "
                "Install it with: pip install sentencepiece"
            ) from error
        raise

    if not use_model_parallel:
        model.to(device)
    model.eval()

    input_device = _resolve_model_input_device(model, device)
    client = (tokenizer, model, torch, input_device)
    _TRANSFORMERS_CLIENT_CACHE[cache_key] = client
    return client


def infer_with_transformers(
    prompt: str,
    *,
    model_name: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    gpu_id: int | None = None,
    num_gpu: int | None = None,
    parallel_gpu_ids: tuple[int, ...] | None = None,
) -> str:
    del num_ctx  # Not directly mapped in this simple transformers backend.
    del num_gpu  # Not directly mapped in this simple transformers backend.

    tokenizer, model, torch_module, device = _load_transformers_client(
        model_name=model_name,
        gpu_id=gpu_id,
        parallel_gpu_ids=parallel_gpu_ids,
    )

    if seed is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_build_chat_template_kwargs(model_name),
            )
        except Exception:
            rendered_prompt = prompt
    else:
        rendered_prompt = prompt

    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    max_new_tokens = num_predict if num_predict is not None and num_predict > 0 else 256
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    context_limit = _resolve_context_limit(tokenizer, model)
    if context_limit is not None:
        if prompt_tokens >= context_limit:
            raise RuntimeError(
                f"Prompt length ({prompt_tokens} tokens) exceeds model context window "
                f"({context_limit}) for {model_name}. Use RAG or shorten the prompt."
            )
        allowed_new_tokens = context_limit - prompt_tokens
        if allowed_new_tokens <= 0:
            raise RuntimeError(
                f"No room left for generation: prompt uses {prompt_tokens}/{context_limit} "
                f"tokens for {model_name}."
            )
        if max_new_tokens > allowed_new_tokens:
            print(
                f"[INFO] Reducing max_new_tokens from {max_new_tokens} to "
                f"{allowed_new_tokens} to fit the {context_limit}-token context window."
            )
            max_new_tokens = allowed_new_tokens

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }

    do_sample = temperature is not None and temperature > 0
    generation_kwargs["do_sample"] = do_sample
    if do_sample:
        generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p

    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    try:
        with torch_module.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
    except Exception as error:
        message = str(error)
        if _needs_eager_attention(model_name) and (
            "AttentionInterface" in message or "torchdynamo" in message.lower()
        ):
            raise RuntimeError(
                "Gemma attention execution failed under the current torch/transformers "
                "stack. Eager attention is already enabled; if this still happens, "
                "retry in RAG mode or update the container runtime."
            ) from error
        raise

    input_length = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][input_length:]
    content = _decode_generated_content(tokenizer, generated_ids, model_name).strip()
    if not content:
        content = _decode_generated_content(tokenizer, output_ids[0], model_name).strip()
    return content


def infer_llama4(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="llama4:latest",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_llama3_3_latest(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="llama3.3:latest",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_llama3_1_8b(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="llama3.1:8b",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_mistral_8b(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="mistral:8b",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_mistral_latest(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="mistral:latest",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_qwen2_5_32b(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="qwen2.5:32b",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_deepseek_r1(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def infer_qwen3_8b(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
) -> str:
    return infer_with_ollama(
        prompt,
        model_name="qwen3:8b",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        seed=seed,
        num_predict=num_predict,
    )


def list_model_keys() -> list[str]:
    return sorted(MODEL_SPECS.keys())


def resolve_model_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        available = ", ".join(list_model_keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available}")
    return MODEL_SPECS[model_key]


def build_model_infer(
    model_key: str,
    *,
    provider: str = "auto",
    model_name_override: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    gpu_id: int | None = None,
    num_gpu: int | None = None,
    parallel_gpu_ids: tuple[int, ...] | None = None,
) -> tuple[Callable[[str], str], ModelSpec, str]:
    spec = resolve_model_spec(model_key)

    if provider == "auto":
        resolved_provider = spec.provider
    else:
        resolved_provider = provider

    if resolved_provider not in {"ollama", "transformers"}:
        raise ValueError(
            "Unsupported provider: "
            f"{resolved_provider}. Supported: 'ollama', 'transformers', 'auto'."
        )

    resolved_model_name = model_name_override if model_name_override else spec.default_model_name

    if resolved_provider == "ollama":
        infer_fn = partial(
            infer_with_ollama,
            model_name=resolved_model_name,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx,
            seed=seed,
            num_predict=num_predict,
            gpu_id=gpu_id,
            num_gpu=num_gpu,
        )
    else:
        infer_fn = partial(
            infer_with_transformers,
            model_name=resolved_model_name,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx,
            seed=seed,
            num_predict=num_predict,
            gpu_id=gpu_id,
            num_gpu=num_gpu,
            parallel_gpu_ids=parallel_gpu_ids,
        )

    return infer_fn, spec, resolved_model_name
