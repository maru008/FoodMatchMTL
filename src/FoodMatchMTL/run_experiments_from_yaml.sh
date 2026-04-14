#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-src/FoodMatchMTL/experiments/default_grid.yaml}"
GPU_LIST="${GPU_LIST:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPANDER_MODULE="${EXPANDER_MODULE:-src.FoodMatchMTL.expand_experiments}"
TRAIN_MODULE_OVERRIDE="${TRAIN_MODULE_OVERRIDE:-}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-env}"
SINGULARITY_BIND="${SINGULARITY_BIND:-/workspace}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --module)
      TRAIN_MODULE_OVERRIDE="$2"
      shift 2
      ;;
    --use-singularity)
      USE_SINGULARITY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ "$USE_SINGULARITY" == "1" ]] && ! command -v singularity >/dev/null 2>&1; then
  echo "[WARN] singularity コマンドが見つからないため USE_SINGULARITY=0 にフォールバックします。" >&2
  USE_SINGULARITY=0
fi

EXPAND_CMD=("$PYTHON_BIN" "-m" "$EXPANDER_MODULE" "--config" "$CONFIG_PATH" "--python_bin" "$PYTHON_BIN")
if [[ -n "$TRAIN_MODULE_OVERRIDE" ]]; then
  EXPAND_CMD+=("--module" "$TRAIN_MODULE_OVERRIDE")
fi

if [[ "$USE_SINGULARITY" == "1" ]]; then
  mapfile -t JOB_LINES < <(
    singularity exec --nv --bind "$SINGULARITY_BIND" "$SINGULARITY_IMAGE" "${EXPAND_CMD[@]}"
  )
else
  mapfile -t JOB_LINES < <("${EXPAND_CMD[@]}")
fi

if [[ ${#JOB_LINES[@]} -eq 0 ]]; then
  echo "No experiments expanded from YAML: $CONFIG_PATH" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPU provided (--gpus)" >&2
  exit 1
fi

TOTAL=${#JOB_LINES[@]}
echo "Config: $CONFIG_PATH"
echo "Total experiments: $TOTAL"
echo "GPUs: ${GPU_LIST}"
echo "Dry run: ${DRY_RUN}"
echo "Use singularity: ${USE_SINGULARITY}"

run_one() {
  local gpu="$1"
  local idx="$2"
  local name="$3"
  local cmd="$4"

  local final_cmd="$cmd"
  if [[ "$final_cmd" != *"--device "* ]]; then
    final_cmd="$final_cmd --device cuda:0"
  fi
  if [[ "$final_cmd" != *"--multi_gpu "* ]]; then
    final_cmd="$final_cmd --multi_gpu false"
  fi

  echo "[GPU${gpu}] [${idx}/${TOTAL}] ${name}"
  echo "  $final_cmd"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  if [[ "$USE_SINGULARITY" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" singularity exec --nv --bind "$SINGULARITY_BIND" "$SINGULARITY_IMAGE" \
      bash -lc "cd '$ROOT_DIR' && $final_cmd"
  else
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "cd '$ROOT_DIR' && $final_cmd"
  fi
}

pids=()
for gpu_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$gpu_index]}"
  (
    for i in "${!JOB_LINES[@]}"; do
      if (( i % ${#GPUS[@]} != gpu_index )); then
        continue
      fi
      line="${JOB_LINES[$i]}"
      IFS=$'\t' read -r idx name cmd <<< "$line"
      run_one "$gpu" "$idx" "$name" "$cmd"
    done
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All experiments completed."
