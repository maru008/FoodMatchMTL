#!/usr/bin/env bash
# =============================================================================
# run_proj_head_ablation.sh
# ProjectionHead + cat_on_canonical アブレーション実験
# 3 GPU 並列
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER="${CONTAINER:-${ROOT_DIR}/env}"
WORKSPACE="${WORKSPACE:-${ROOT_DIR}}"
OUT="./src/FoodMatchMTL/results/proj_head_ablation"
LOGS="${OUT}/logs"
FN2ID="./data/foodtable_name2id.json"
mkdir -p "$LOGS"

# singularity exec ラッパー
srun() {
    local gpu=$1; shift
    local label=$1; shift
    local log="${LOGS}/${label}.log"
    echo "[$(date '+%H:%M:%S')] START: ${label} (GPU=${gpu})"
    CUDA_VISIBLE_DEVICES=${gpu} singularity exec --nv \
        --bind "${WORKSPACE}:/workspace" "${CONTAINER}" \
        python3 -m src.FoodMatchMTL.train "$@" \
        >"${log}" 2>&1 &
    echo $!
}

# 共通オプション関数
e5_opts() {
    echo --encoder multilingual_e5_large \
         --lr 1e-5 --batch_size 8 --epochs 10 \
         --alpha 0.05 --beta 0.1 --seed 42 \
         --max_len 64 --pooling mean --normalize true --score_fn cosine \
         --temperature 0.05 \
         --foodtable_name2id_path "${FN2ID}" \
         --use_wandb false --output_root "${OUT}"
}
sbert_opts() {
    echo --encoder sentence_bert \
         --lr 1e-5 --batch_size 8 --epochs 10 \
         --alpha 0.05 --beta 0.1 --seed 42 \
         --max_len 64 --pooling mean --normalize true --score_fn cosine \
         --temperature 0.05 \
         --foodtable_name2id_path "${FN2ID}" \
         --use_wandb false --output_root "${OUT}"
}
qwen_opts() {
    echo --encoder qwen3_8b \
         --lr 1e-5 --batch_size 8 --epochs 10 \
         --alpha 0.05 --beta 0.1 --seed 42 \
         --normalize true --score_fn cosine \
         --temperature 0.05 \
         --foodtable_name2id_path "${FN2ID}" \
         --use_wandb false --output_root "${OUT}"
}

show_results() {
    singularity exec --bind "${WORKSPACE}:/workspace" "${CONTAINER}" python3 -c "
import os, json, glob
results = []
for f in sorted(glob.glob('/workspace/src/FoodMatchMTL/results/proj_head_ablation/**/eval_result.json', recursive=True)):
    d = json.load(open(f))
    cfg_f = os.path.join(os.path.dirname(f), 'config.json')
    if not os.path.exists(cfg_f): continue
    cfg = json.load(open(cfg_f))
    hp = cfg.get('hyperparams', {})
    enc = cfg.get('encoder_alias','?')[:10]
    mode = cfg.get('mode','?')
    proj = 'T' if hp.get('use_proj_head_id') else 'F'
    cat_c = 'T' if hp.get('cat_on_canonical') else 'F'
    lt = hp.get('loss_type','infonce')[:10]
    seed = hp.get('seed','?')
    em = d['metrics']['EM']
    cm = d['metrics']['CategoryMatch']
    results.append((enc,mode,proj,cat_c,lt,seed,em,cm))
print(f'  {\"enc\":<12} {\"mode\":<12} P C {\"loss\":<12} {\"seed\"} {\"EM\":>7} {\"CM\":>7}')
print('  ' + '-'*70)
for r in sorted(results, key=lambda x:(x[0],x[2],x[3],x[4],x[1])):
    print(f'  {r[0]:<12} {r[1]:<12} {r[2]} {r[3]} {r[4]:<12} {r[5]} {r[6]:>7.4f} {r[7]:>7.4f}')
" 2>/dev/null || echo "(集計スキップ)"
}

# ============================================================
# Round 1: E5 アブレーション (3 GPU 並列)
# ============================================================
echo "===== Round 1: E5 ablation ====="
cd "${WORKSPACE}"

P0=$(srun 0 r1_e5_proj_cat \
    --mode only_id --run_task1_mode_sweep true \
    $(e5_opts) \
    --use_proj_head_id true --cat_on_canonical true)

P1=$(srun 1 r1_e5_proj_only \
    --mode only_id --run_task1_mode_sweep true \
    $(e5_opts) \
    --use_proj_head_id true --cat_on_canonical false)

P2=$(srun 2 r1_e5_cat_only \
    --mode only_id --run_task1_mode_sweep true \
    $(e5_opts) \
    --use_proj_head_id false --cat_on_canonical true)

wait $P0 $P1 $P2 || true
echo "===== Round 1 done ====="
show_results

# ============================================================
# Round 2: tax_infonce + SBERT
# ============================================================
echo "===== Round 2: tax_infonce + SBERT ====="

P0=$(srun 0 r2_e5_proj_cat_taxinf \
    --mode only_id --run_task1_mode_sweep true \
    $(e5_opts) \
    --use_proj_head_id true --cat_on_canonical true \
    --loss_type tax_infonce --tax_soft_temp 2.0)

P1=$(srun 1 r2_e5_proj_cat_taxinf_b0 \
    --mode only_id --run_task1_mode_sweep true \
    $(e5_opts) --beta 0.0 \
    --use_proj_head_id true --cat_on_canonical true \
    --loss_type tax_infonce --tax_soft_temp 2.0)

P2=$(srun 2 r2_sbert_proj_cat \
    --mode only_id --run_task1_mode_sweep true \
    $(sbert_opts) \
    --use_proj_head_id true --cat_on_canonical true)

wait $P0 $P1 $P2 || true
echo "===== Round 2 done ====="
show_results

# ============================================================
# Round 3: Qwen3 + seed 確認
# ============================================================
echo "===== Round 3: Qwen3 + seed confirmation ====="

P0=$(srun 0 r3_qwen_proj_cat \
    --mode only_id --run_task1_mode_sweep true \
    $(qwen_opts) \
    --use_proj_head_id true --cat_on_canonical true)

P1=$(srun 1 r3_e5_proj_cat_s43 \
    --mode only_id --run_task1_mode_sweep true \
    --encoder multilingual_e5_large --lr 1e-5 --batch_size 8 --epochs 10 \
    --alpha 0.05 --beta 0.1 --seed 43 \
    --max_len 64 --pooling mean --normalize true --score_fn cosine \
    --temperature 0.05 \
    --foodtable_name2id_path "${FN2ID}" \
    --use_wandb false --output_root "${OUT}" \
    --use_proj_head_id true --cat_on_canonical true)

P2=$(srun 2 r3_sbert_proj_cat_s43 \
    --mode only_id --run_task1_mode_sweep true \
    --encoder sentence_bert --lr 1e-5 --batch_size 8 --epochs 10 \
    --alpha 0.05 --beta 0.1 --seed 43 \
    --max_len 64 --pooling mean --normalize true --score_fn cosine \
    --temperature 0.05 \
    --foodtable_name2id_path "${FN2ID}" \
    --use_wandb false --output_root "${OUT}" \
    --use_proj_head_id true --cat_on_canonical true)

wait $P0 $P1 $P2 || true
echo "===== Round 3 done ====="

echo "====== All done ======"
show_results
echo "Logs: ${LOGS}/"
