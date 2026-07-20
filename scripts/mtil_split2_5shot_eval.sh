#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
source "${SCRIPT_DIR}/mtil_split2_5shot_config.sh"

DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
FEW_SHOT=${FEW_SHOT:-5}
MODEL=${MODEL:-ViT-B/16}
BATCH_SIZE=${BATCH_SIZE:-64}
BATCH_SIZE_EVAL=${BATCH_SIZE_EVAL:-128}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mtil_split2_5shot}
EDITED_CKPT=${EDITED_CKPT:-${OUTPUT_DIR}/edited/split2_5shot_edited.pth}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/eval_logs}
RESULTS_TSV=${RESULTS_TSV:-${OUTPUT_DIR}/eval_results.tsv}
RESULT_PART_DIR=${RESULT_PART_DIR:-${OUTPUT_DIR}/eval_result_parts}
GLOBAL_LABEL_EVAL=${GLOBAL_LABEL_EVAL:-1}
GLOBAL_LABEL_DATASETS=${GLOBAL_LABEL_DATASETS:-$(IFS=,; echo "${MTIL_SPLIT2_DATASETS[*]}")}
GLOBAL_LABEL_ARGS=()
if [[ "${GLOBAL_LABEL_EVAL}" == "1" ]]; then
  GLOBAL_LABEL_ARGS+=(--eval-all-labels --global-label-datasets "${GLOBAL_LABEL_DATASETS}")
fi

read -r -a GPU_LIST <<< "${GPUS//,/ }"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  GPU_LIST=("${GPU}")
fi
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}" "${RESULT_PART_DIR}" "$(dirname "${RESULTS_TSV}")"
rm -f "${RESULT_PART_DIR}"/*.tsv
cd "${PROJECT_ROOT}"

run_eval() {
  local i=$1
  local gpu_id=$2
  local ds=${MTIL_SPLIT2_DATASETS[$i]}
  local log_file="${LOG_DIR}/${ds}.log"
  local part_file
  part_file=$(printf "%s/%02d_%s.tsv" "${RESULT_PART_DIR}" "$((i + 1))" "${ds}")

  echo "[eval] ${ds} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.main --eval-only \
    --model "${MODEL}" \
    --few_shot "${FEW_SHOT}" \
    --data-location "${DATA_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    --batch-size-eval "${BATCH_SIZE_EVAL}" \
    --train-mode ffn_out \
    --lr "${MTIL_SPLIT2_LR[$i]}" \
    --ls "${MTIL_SPLIT2_LS[$i]}" \
    --wd "${MTIL_SPLIT2_WD[$i]}" \
    --iterations "${MTIL_SPLIT2_ITERATIONS[$i]}" \
    --train-dataset "${ds}" \
    --eval-datasets "${ds}" \
    "${GLOBAL_LABEL_ARGS[@]}" \
    --load "${EDITED_CKPT}" 2>&1 | tee "${log_file}"

  local acc
  acc=$(grep -oE 'Top-1 accuracy: [0-9.]+' "${log_file}" | tail -n1 | awk '{print $3}' || true)
  printf '%s\t%s\n' "${ds}" "${acc:-NA}" > "${part_file}"
}

N=${#MTIL_SPLIT2_DATASETS[@]}
pids=()
for ((i=0; i<N; i++)); do
  gpu_id=${GPU_LIST[$((i % NUM_GPUS))]}
  run_eval "${i}" "${gpu_id}" &
  pids+=("$!")

  if [[ ${#pids[@]} -eq ${NUM_GPUS} ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

printf '%s\t%s\n' dataset top1 > "${RESULTS_TSV}"
for ((i=0; i<N; i++)); do
  ds=${MTIL_SPLIT2_DATASETS[$i]}
  part_file=$(printf "%s/%02d_%s.tsv" "${RESULT_PART_DIR}" "$((i + 1))" "${ds}")
  if [[ ! -f "${part_file}" ]]; then
    echo "[error] missing eval result: ${part_file}" >&2
    exit 1
  fi
  cat "${part_file}" >> "${RESULTS_TSV}"
done

python tools/summarize_results_tsv.py "${RESULTS_TSV}" --append-mean
