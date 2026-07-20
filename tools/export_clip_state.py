#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import clip.clip as clip


def parse_args():
    parser = argparse.ArgumentParser(description="Export a CLIP model state_dict checkpoint.")
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_clip_model(model_name):
    loaded = clip.load(model_name, jit=False)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def main():
    args = parse_args()
    model = load_clip_model(args.model)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.output)
    print(f"Saved base CLIP checkpoint to {args.output}")


if __name__ == "__main__":
    main()
