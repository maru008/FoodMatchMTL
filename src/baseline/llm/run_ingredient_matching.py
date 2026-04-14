from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from llm_direct_common import run_direct_matching
from model_clients import build_model_infer, list_model_keys


def _sanitize_output_tag(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return cleaned or "model"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ingredient direct matching with one entrypoint and pluggable model clients.",
    )
    parser.add_argument(
        "--model",
        choices=list_model_keys(),
        required=True,
        help="Model key defined in model_clients.py",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help="Inference provider. auto/ollama/transformers. Default: auto",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override provider model name (ex: llama3.1:8b).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "res",
        help="Root output directory. Model folder will be created under this path.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Output folder name under output-root. Defaults to model key tag.",
    )
    parser.add_argument(
        "--output-filename",
        default="ingredient_matching_direct.json",
        help="Progress output filename.",
    )
    parser.add_argument(
        "--completed-filename",
        default="ingredient_matching_direct_comp.json",
        help="Completed output filename.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save progress every N records.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per ingredient when inference or JSON parse fails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N ingredients (debug use).",
    )

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    infer_fn, spec, resolved_model_name = build_model_infer(
        args.model,
        provider=args.provider,
        model_name_override=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        seed=args.seed,
        num_predict=args.num_predict,
        gpu_id=args.gpu_id,
        num_gpu=args.num_gpu,
        parallel_gpu_ids=args.parallel_gpu_ids,
    )

    if args.output_tag:
        output_tag = args.output_tag
    elif args.model_name:
        output_tag = _sanitize_output_tag(args.model_name)
    else:
        output_tag = spec.output_tag

    output_dir = args.output_root / output_tag

    print(f"Model key: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Model name: {resolved_model_name}")
    print(f"Parallel GPU ids: {args.parallel_gpu_ids}")
    print(f"Output directory: {output_dir}")

    run_direct_matching(
        output_dir=output_dir,
        model_infer=infer_fn,
        save_every=args.save_every,
        max_retries=args.max_retries,
        limit=args.limit,
        output_filename=args.output_filename,
        completed_filename=args.completed_filename,
    )


if __name__ == "__main__":
    main()
