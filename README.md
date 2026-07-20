# CVPR26 Highlight: SAME: Sparse and Anchored Model Editing

Official implementation of **SAME: Sparse and Anchored Model Editing for Heterogeneous Incremental Learning under Limited Data** (CVPR 2026 Highlight).

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_SAME_Sparse_and_Anchored_Model_Editing_for_Heterogeneous_Incremental_Learning_CVPR_2026_paper.html)

This folder contains the MTIL split2 5-shot workflow used by SAME. The workflow trains task-specific sparse updates, accumulates one anchored KV statistics file, edits a base CLIP model, and evaluates the edited model under a shared class-incremental label space.

## Environment and Data

Please follow the MTIL setup requirements in [ZSCL](https://github.com/Thunderbeee/ZSCL/tree/main) for environment preparation, dataset organization, and CLIP model download.

This release assumes the datasets and the CLIP checkpoint are already prepared locally. Set the following paths before running the scripts:

```bash
cd /path/to/SAME
export DATA_ROOT=/path/to/datasets
export MODEL=/path/to/ViT-B-16.pt
```

## MTIL Split2 5-shot Tutorial

SAME exposes the split2 5-shot MTIL setting in this folder. The 11 MTIL datasets are split into two class-disjoint tasks, resulting in 22 split tasks such as `Aircraft_1_2` and `Aircraft_2_2`.

### 1. Train Sparse Task Updates and Save One KV File

The default hyperparameters are tuned for the 5-shot split2 setting. If you change the training setting, the training and editing hyperparameters should be adjusted for better performance.

```bash
bash scripts/mtil_split2_5shot_finetune.sh
```

The script independently fine-tunes each split task with its own task labels, in the order defined by `scripts/mtil_split2_5shot_config.sh`. By default, each in-memory fine-tuned model is evaluated immediately after training and the 22-task mean is summarized. The script then extracts KV statistics from the in-memory fine-tuned model and its pre-finetuning anchor model, then merges them into a single file. KV magnitude balancing uses fixed global reference values by default, so reordering tasks does not change the scale factors:

```text
outputs/mtil_split2_5shot/kv/split2_5shot_merged_kv.pkl
```

The independent fine-tuning evaluation summary is written to:

```text
outputs/mtil_split2_5shot/finetune_eval_results.tsv
```

By default, per-task fine-tuned checkpoints are not saved.

### 2. Edit the Base CLIP Model

```bash
bash scripts/mtil_split2_5shot_edit.sh
```

The edited checkpoint is saved to:

```text
outputs/mtil_split2_5shot/edited/split2_5shot_edited.pth
```

### 3. Evaluate with a Shared Global Label Space

Evaluation defaults to eight GPUs (`0,1,2,3,4,5,6,7`):

```bash
bash scripts/mtil_split2_5shot_eval.sh
```

To override the devices:

```bash
export GPUS=0,1
bash scripts/mtil_split2_5shot_eval.sh
```

Final evaluation uses a fixed global label space over all 22 split tasks by default. Each task keeps local labels internally, and the evaluator maps them to stable global class ids before computing accuracy. Results are written to:

```text
outputs/mtil_split2_5shot/eval_results.tsv
```

The final `mean_top1` is the arithmetic mean over the 22 split-task Top-1 accuracies.

## Expected Results

Due to the randomness in few-shot sampling, sparse gradient masking, and GPU execution, exact numbers may vary slightly across runs and environments. With the default split2 5-shot hyperparameters, the independent fine-tuning mean Top-1 is typically around `69`, and the final edited single-model mean Top-1 is expected to be above `65`.

## Important Options

- `DATA_ROOT`: dataset root.
- `MODEL`: CLIP checkpoint path or model name.
- `GPU`: single GPU id for training.
- `GPUS`: comma- or space-separated GPU ids for parallel evaluation, default `0,1,2,3,4,5,6,7`.
- `FEW_SHOT`: shots per class, default `5`.
- `MASK_RATIO`: random sparse gradient mask ratio, default `0.3`.
- `SAVE_MODELS`: whether to save per-task fine-tuned checkpoints, default `0`.
- `POST_TRAIN_EVAL`: evaluate each in-memory fine-tuned task model and summarize the 22-task mean during training, default `1`.
- `GLOBAL_LABEL_EVAL`: use the shared all-task label space when `1`, default `1`.
- `SCALE_REFERENCE`: KV magnitude balancing mode, default `global`.
- `TARGET_SCALE_REFERENCE_KK` and `ANCHOR_SCALE_REFERENCE_KK`: fixed global KV reference magnitudes used when `SCALE_REFERENCE=global`.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{
duan2026retain,
title={SAME: Sparse and Anchored Model Editing  for Heterogeneous Incremental Learning under Limited Data},
author={Zixuan Duan and Zeyu Zhang and Fengyuan Lu and Shaofeng Zhang and Wenbin Li and Qi Fan and Yang Gao },
booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
year={2026}
}
```

## Acknowledgement

This codebase is built upon [ZSCL](https://github.com/Thunderbeee/ZSCL/tree/main). We sincerely thank the authors for releasing their code and MTIL benchmark implementation.
