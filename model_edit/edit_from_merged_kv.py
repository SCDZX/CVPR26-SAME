#!/usr/bin/env python3
import argparse
import collections
import os
import pickle

import torch
import torch.nn as nn


def load_feature_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ckpt_as_state(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
    elif isinstance(obj, collections.OrderedDict):
        state = obj
    elif isinstance(obj, nn.Module):
        state = obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    stripped = collections.OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped



def combine_feature_stats(target, anchor=None, anchor_weight=20.0):
    if anchor is None or anchor_weight == 0:
        return target

    combined = {"kk": {}, "kv": {}, "sum_x": {}, "sum_y": {}, "n": {}}
    if "sum_yy" in target:
        combined["sum_yy"] = {}

    for key in ("kk", "kv", "sum_x", "sum_y"):
        for name, value in target[key].items():
            combined[key][name] = value + anchor[key][name] * anchor_weight

    if "sum_yy" in combined:
        for name, value in target["sum_yy"].items():
            combined["sum_yy"][name] = value + anchor["sum_yy"][name] * anchor_weight

    for name, value in target["n"].items():
        combined["n"][name] = float(value) + float(anchor["n"][name]) * anchor_weight

    return combined


def save_state_with_key(state, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"state_dict": state}, path)
    print(f"[save] wrote edited checkpoint to {path}")


@torch.no_grad()
def update_state_dict_from_stats(state, feat, layer_names, lam, lam_b, device):
    dev = torch.device(device)
    for name in layer_names:
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key not in state:
            print(f"[skip] {weight_key} not found in checkpoint")
            continue

        w_old = state[weight_key].detach().cpu()
        has_bias = bias_key in state
        b_old = state[bias_key].detach().cpu() if has_bias else torch.zeros(w_old.size(0))

        out_dim, in_dim = w_old.shape
        compute_dtype = torch.float32 if w_old.dtype in (torch.float16, torch.bfloat16) else w_old.dtype

        kk = feat["kk"][name].to(dev, dtype=compute_dtype)
        kv = feat["kv"][name].to(dev, dtype=compute_dtype)
        sx = feat["sum_x"][name].to(dev, dtype=compute_dtype).reshape(in_dim, 1)
        sy = feat["sum_y"][name].to(dev, dtype=compute_dtype).reshape(1, out_dim)
        n = torch.tensor(float(feat["n"][name]), device=dev, dtype=compute_dtype)

        w0 = w_old.t().to(dev, dtype=compute_dtype)
        b0 = b_old.reshape(1, out_dim).to(dev, dtype=compute_dtype)

        a = torch.empty((in_dim + 1, in_dim + 1), device=dev, dtype=compute_dtype)
        a[:in_dim, :in_dim] = kk
        a[:in_dim, in_dim:] = sx
        a[in_dim:, :in_dim] = sx.t()
        a[in_dim, in_dim] = n

        b = torch.empty((in_dim + 1, out_dim), device=dev, dtype=compute_dtype)
        b[:in_dim, :] = kv
        b[in_dim:, :] = sy

        penalty = torch.zeros((in_dim + 1,), device=dev, dtype=compute_dtype)
        penalty[:in_dim] = lam
        penalty[in_dim] = lam_b

        a_reg = a + torch.diag(penalty)
        rhs = b + torch.vstack((w0, b0)) * penalty.unsqueeze(1)

        try:
            solution = torch.linalg.solve(a_reg, rhs)
        except RuntimeError as exc:
            print(f"[warn] solve failed for {name}: {exc}; falling back to lstsq")
            solution = torch.linalg.lstsq(a_reg, rhs).solution

        state[weight_key] = solution[:in_dim, :].t().contiguous().to(dtype=w_old.dtype, device="cpu")
        if has_bias:
            state[bias_key] = solution[in_dim, :].contiguous().to(dtype=b_old.dtype, device="cpu")
        print(f"[update] {name}: {tuple(w_old.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edit CLIP FFN output layers from one merged MTIL KV statistics file."
    )
    parser.add_argument("--model", required=True, help="Base CLIP checkpoint/state_dict.")
    parser.add_argument("--feature-file", required=True, help="Merged KV pickle file.")
    parser.add_argument("--ckpt-out", required=True, help="Output edited checkpoint.")
    parser.add_argument("--lam", type=float, default=300.0)
    parser.add_argument("--lam-b", type=float, default=300.0)
    parser.add_argument("--layer-names", default="auto")
    parser.add_argument("--anchor-weight", type=float, default=1.0, help="Weight for pre-finetuning KV anchor statistics.")
    parser.add_argument("--no-anchor", action="store_true", help="Ignore anchor statistics even if present in the KV file.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    state = load_ckpt_as_state(args.model)
    payload = load_feature_dict(args.feature_file)

    if "target" in payload:
        target = payload["target"]
        anchor = None if args.no_anchor else payload.get("anchor")
        if anchor is not None:
            print(f"[info] using pre-finetuning KV anchor with weight={args.anchor_weight:g}")
        else:
            print("[info] editing without KV anchor")
        feat = combine_feature_stats(target, anchor, anchor_weight=args.anchor_weight)
    else:
        feat = payload
        print("[info] legacy flat KV file detected; no separate anchor stats found")

    if args.layer_names.strip().lower() == "auto":
        layer_names = list(feat["kk"].keys())
    else:
        layer_names = [item.strip() for item in args.layer_names.split(",") if item.strip()]

    print(f"[info] editing {len(layer_names)} layers")
    update_state_dict_from_stats(
        state, feat, layer_names, lam=args.lam, lam_b=args.lam_b, device=args.device
    )
    save_state_with_key(state, args.ckpt_out)


if __name__ == "__main__":
    main()
