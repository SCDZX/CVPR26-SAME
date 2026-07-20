#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
source "${SCRIPT_DIR}/mtil_split2_5shot_config.sh"

DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
FEW_SHOT=${FEW_SHOT:-5}
MODEL=${MODEL:-ViT-B/16}
BATCH_SIZE=${BATCH_SIZE:-64}
BATCH_SIZE_EVAL=${BATCH_SIZE_EVAL:-128}
MASK_RATIO=${MASK_RATIO:-0.3}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mtil_split2_5shot}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/finetuned}
SAVE_MODELS=${SAVE_MODELS:-0}
POST_TRAIN_EVAL=${POST_TRAIN_EVAL:-1}
TRAIN_EVAL_LOG_DIR=${TRAIN_EVAL_LOG_DIR:-${OUTPUT_DIR}/finetune_eval_logs}
TRAIN_EVAL_RESULTS=${TRAIN_EVAL_RESULTS:-${OUTPUT_DIR}/finetune_eval_results.tsv}
KV_OUT=${KV_OUT:-${OUTPUT_DIR}/kv/split2_5shot_merged_kv.pkl}
RESET_MERGED_KV=${RESET_MERGED_KV:-1}
TARGET_WEIGHTS=${TARGET_WEIGHTS:-${MTIL_SPLIT2_DEFAULT_TARGET_WEIGHTS}}
ANCHOR_WEIGHTS=${ANCHOR_WEIGHTS:-${MTIL_SPLIT2_DEFAULT_ANCHOR_WEIGHTS}}
SCALE_REFERENCE=${SCALE_REFERENCE:-${MTIL_SPLIT2_SCALE_REFERENCE}}
TARGET_SCALE_REFERENCE_KK=${TARGET_SCALE_REFERENCE_KK:-${MTIL_SPLIT2_TARGET_SCALE_REFERENCE_KK}}
ANCHOR_SCALE_REFERENCE_KK=${ANCHOR_SCALE_REFERENCE_KK:-${MTIL_SPLIT2_ANCHOR_SCALE_REFERENCE_KK}}
GLOBAL_LABEL_DATASETS=${GLOBAL_LABEL_DATASETS:-$(IFS=,; echo "${MTIL_SPLIT2_DATASETS[*]}")}

mkdir -p "${OUTPUT_DIR}" "$(dirname "${KV_OUT}")"
if [[ "${SAVE_MODELS}" == "1" ]]; then
  mkdir -p "${CKPT_DIR}"
fi
if [[ "${POST_TRAIN_EVAL}" == "1" ]]; then
  mkdir -p "${TRAIN_EVAL_LOG_DIR}" "$(dirname "${TRAIN_EVAL_RESULTS}")"
  printf '%s\t%s\n' dataset top1 > "${TRAIN_EVAL_RESULTS}"
fi
if [[ "${RESET_MERGED_KV}" == "1" && -f "${KV_OUT}" ]]; then
  rm -f "${KV_OUT}"
fi

cd "${PROJECT_ROOT}"

SAVE_ARGS=()
if [[ "${SAVE_MODELS}" == "1" ]]; then
  SAVE_ARGS=(--save "${CKPT_DIR}")
fi

N=${#MTIL_SPLIT2_DATASETS[@]}
for ((i=0; i<N; i++)); do
  ds=${MTIL_SPLIT2_DATASETS[$i]}
  echo "[finetune] ${ds}"

  TRAIN_ARGS=(
    --model "${MODEL}"
    --data-location "${DATA_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --batch-size-eval "${BATCH_SIZE_EVAL}"
    --train-mode ffn_out
    --train-dataset "${ds}"
    --lr "${MTIL_SPLIT2_LR[$i]}"
    --few_shot "${FEW_SHOT}"
    --random-mask-ratio "${MASK_RATIO}"
    --ls "${MTIL_SPLIT2_LS[$i]}"
    --wd "${MTIL_SPLIT2_WD[$i]}"
    --iterations "${MTIL_SPLIT2_ITERATIONS[$i]}"
    --method finetune
    --merged-kv-output "${KV_OUT}"
    --merged-kv-datasets "${GLOBAL_LABEL_DATASETS}"
    --scale-reference "${SCALE_REFERENCE}"
    --target-scale-reference-kk "${TARGET_SCALE_REFERENCE_KK}"
    --anchor-scale-reference-kk "${ANCHOR_SCALE_REFERENCE_KK}"
    --target-weights "${TARGET_WEIGHTS}"
    --anchor-weights "${ANCHOR_WEIGHTS}"
  )

  if [[ "${POST_TRAIN_EVAL}" == "1" ]]; then
    log_file="${TRAIN_EVAL_LOG_DIR}/${ds}.log"
    TRAIN_ARGS+=(
      --post-train-eval
      --eval-datasets "${ds}"
      --eval-all-labels
      --global-label-datasets "${GLOBAL_LABEL_DATASETS}"
    )
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main "${TRAIN_ARGS[@]}" "${SAVE_ARGS[@]}" 2>&1 | tee "${log_file}"
    acc=$(grep -oE 'Top-1 accuracy: [0-9.]+' "${log_file}" | tail -n1 | awk '{print $3}' || true)
    printf '%s\t%s\n' "${ds}" "${acc:-NA}" >> "${TRAIN_EVAL_RESULTS}"
  else
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main "${TRAIN_ARGS[@]}" "${SAVE_ARGS[@]}"
  fi
done

if [[ "${POST_TRAIN_EVAL}" == "1" ]]; then
  python tools/summarize_results_tsv.py "${TRAIN_EVAL_RESULTS}" --append-mean
  echo "[post-train-summary] wrote ${TRAIN_EVAL_RESULTS}"
fi

echo "[merged-kv] wrote ${KV_OUT}"
