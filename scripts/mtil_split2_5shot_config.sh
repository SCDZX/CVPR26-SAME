#!/usr/bin/env bash

MTIL_SPLIT2_DATASETS=(
  Aircraft_1_2 Caltech101_1_2 Food_1_2 MNIST_1_2 OxfordPet_1_2 Flowers_1_2 SUN397_1_2 DTD_1_2 EuroSAT_1_2 CIFAR100_1_2 StanfordCars_1_2
  Aircraft_2_2 Caltech101_2_2 Food_2_2 MNIST_2_2 OxfordPet_2_2 Flowers_2_2 SUN397_2_2 DTD_2_2 EuroSAT_2_2 CIFAR100_2_2 StanfordCars_2_2
)

MTIL_SPLIT2_LR=(
  3e-4 1e-4 2e-7 3e-4 2e-7 2e-4 2e-6 2e-4 5e-5 3e-5 1e-6
  2e-4 1e-5 2e-7 5e-4 2e-7 2e-4 2e-6 2e-4 5e-5 3e-5 1e-6
)

MTIL_SPLIT2_ITERATIONS=(
  500 500 500 500 500 500 500 500 500 500 500
  500 500 500 500 500 500 500 500 500 500 500
)

MTIL_SPLIT2_LS=(
  0.2 0.1 0.05 0.2 0.07 0.15 0.13 0.23 0.2 0.15 0.1
  0.2 0.1 0.05 0.2 0.07 0.15 0.13 0.23 0.2 0.15 0.1
)

MTIL_SPLIT2_WD=(
  0.01 0 0 0 0.005 0.00005 0 0 0.01 0.005 0
  0.01 0 0 0 0.005 0.00005 0 0 0.01 0.005 0
)

# KV magnitude balancing.  The old tuning runs used Aircraft_1_2 as the
# first-task reference, whose logged kk_sum was about 2.12347e10 for target
# and 2.42692e10 for anchor.  Keeping these as explicit global references
# makes the merged KV invariant to task order changes.
MTIL_SPLIT2_SCALE_REFERENCE=${MTIL_SPLIT2_SCALE_REFERENCE:-global}
MTIL_SPLIT2_TARGET_SCALE_REFERENCE_KK=${MTIL_SPLIT2_TARGET_SCALE_REFERENCE_KK:-2.12347e10}
MTIL_SPLIT2_ANCHOR_SCALE_REFERENCE_KK=${MTIL_SPLIT2_ANCHOR_SCALE_REFERENCE_KK:-2.42692e10}

# Manual edit/KV balance for the split2 5-shot run.
# Target weights emphasize the five largest drops from independent fine-tuning
# and reduce tasks where the edited model is below the base CLIP baseline.
MTIL_SPLIT2_DEFAULT_TARGET_WEIGHTS="Aircraft_1_2:1,Caltech101_1_2:0.1,Food_1_2:0.1,Flowers_1_2:2,DTD_1_2:2,EuroSAT_1_2:2,Aircraft_2_2:2,Caltech101_2_2:0.1,OxfordPet_2_2:0.1,Flowers_2_2:2,DTD_2_2:2,EuroSAT_2_2:1,StanfordCars_2_2:1"

# Anchor weights are high for tasks where the base CLIP baseline was stronger
# than the edited model; all other split tasks keep the default 0.1 anchor.
MTIL_SPLIT2_DEFAULT_ANCHOR_WEIGHTS="Aircraft_1_2:0.1,Caltech101_1_2:2,Food_1_2:2,MNIST_1_2:0.1,OxfordPet_1_2:0.1,Flowers_1_2:0,SUN397_1_2:0.1,DTD_1_2:0.1,EuroSAT_1_2:0.1,CIFAR100_1_2:0.1,StanfordCars_1_2:0.1,Aircraft_2_2:0.1,Caltech101_2_2:2,Food_2_2:0.1,MNIST_2_2:0.1,OxfordPet_2_2:2,Flowers_2_2:0,SUN397_2_2:0.1,DTD_2_2:0.1,EuroSAT_2_2:0.1,CIFAR100_2_2:0.1,StanfordCars_2_2:2"
