#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/mtil_split2_5shot_finetune.sh"
bash "${SCRIPT_DIR}/mtil_split2_5shot_edit.sh"
bash "${SCRIPT_DIR}/mtil_split2_5shot_eval.sh"
