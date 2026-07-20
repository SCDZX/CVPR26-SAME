#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}

MODEL=${MODEL:-ViT-B/16}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mtil_split2_5shot}
BASE_CKPT=${BASE_CKPT:-${OUTPUT_DIR}/base_clip.pth}
KV_OUT=${KV_OUT:-${OUTPUT_DIR}/kv/split2_5shot_merged_kv.pkl}
EDITED_CKPT=${EDITED_CKPT:-${OUTPUT_DIR}/edited/split2_5shot_edited.pth}
LAM=${LAM:-300}
LAM_B=${LAM_B:-300}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-1.0}
EDIT_DEVICE=${EDIT_DEVICE:-cpu}
FORCE_EXPORT_BASE=${FORCE_EXPORT_BASE:-0}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${BASE_CKPT}")" "$(dirname "${EDITED_CKPT}")"

base_abs=$(readlink -f "${BASE_CKPT}")
edited_abs=$(readlink -f "${EDITED_CKPT}")
if [[ "${base_abs}" == "${edited_abs}" ]]; then
  echo "[error] BASE_CKPT and EDITED_CKPT point to the same file: ${base_abs}" >&2
  echo "[error] Refusing to overwrite the base CLIP checkpoint with edited weights." >&2
  exit 2
fi

if [[ "${FORCE_EXPORT_BASE}" == "1" || ! -f "${BASE_CKPT}" ]]; then
  python tools/export_clip_state.py --model "${MODEL}" --output "${BASE_CKPT}"
fi

python -m model_edit.edit_from_merged_kv \
  --model "${BASE_CKPT}" \
  --feature-file "${KV_OUT}" \
  --ckpt-out "${EDITED_CKPT}" \
  --lam "${LAM}" \
  --lam-b "${LAM_B}" \
  --anchor-weight "${ANCHOR_WEIGHT}" \
  --device "${EDIT_DEVICE}"
