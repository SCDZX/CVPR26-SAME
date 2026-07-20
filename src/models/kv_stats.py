import os
import pickle

import clip.clip as clip
import torch

from ..datasets.common import maybe_dictionarize


DEFAULT_SPLIT2_DATASETS = [
    "Aircraft_1_2", "Caltech101_1_2", "Food_1_2", "MNIST_1_2",
    "OxfordPet_1_2", "Flowers_1_2", "SUN397_1_2", "DTD_1_2",
    "EuroSAT_1_2", "CIFAR100_1_2", "StanfordCars_1_2",
    "Aircraft_2_2", "Caltech101_2_2", "Food_2_2", "MNIST_2_2",
    "OxfordPet_2_2", "Flowers_2_2", "SUN397_2_2", "DTD_2_2",
    "EuroSAT_2_2", "CIFAR100_2_2", "StanfordCars_2_2",
]


def parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_weights_arg(weights_arg, dataset_names):
    weights = {name: 1.0 for name in dataset_names}
    if not weights_arg:
        return weights

    for item in str(weights_arg).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid weight item: {item}. Expected name:weight.")
        name, weight = item.split(":", 1)
        name = name.strip()
        if name not in weights:
            raise KeyError(f"Unknown dataset in weights: {name}")
        weights[name] = float(weight.strip())
    return weights


def make_clip_ffn_out_names(num_visual_blocks=12, num_text_blocks=12):
    names = []
    for i in range(num_visual_blocks):
        names.append(f"visual.transformer.resblocks.{i}.mlp.c_proj")
    for i in range(num_text_blocks):
        names.append(f"transformer.resblocks.{i}.mlp.c_proj")
    return names


def get_layer(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer not found: {layer_name}")


def shape_to_nc(tensor):
    if tensor.dim() == 4:
        return torch.flatten(tensor, 2).transpose(1, 2).reshape(-1, tensor.size(1))
    if tensor.dim() == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


class FeatureCollector:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.handles = []
        self.stats = {
            "kk": {name: None for name in layer_names},
            "kv": {name: None for name in layer_names},
            "sum_x": {name: None for name in layer_names},
            "sum_y": {name: None for name in layer_names},
            "n": {name: 0 for name in layer_names},
            "sum_yy": {name: 0 for name in layer_names},
        }

    def __enter__(self):
        for layer_name in self.layer_names:
            layer = get_layer(self.model, layer_name)
            self.handles.append(layer.register_forward_hook(self._hook(layer_name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer_name):
        def hook(module, fea_in, fea_out):
            x = fea_in[0] if isinstance(fea_in, (tuple, list)) else fea_in
            y = fea_out[0] if isinstance(fea_out, (tuple, list)) else fea_out
            x = shape_to_nc(x.detach())
            y = shape_to_nc(y.detach())

            kk = x.t() @ x
            kv = x.t() @ y
            sx = x.sum(0)
            sy = y.sum(0)
            yy = (y ** 2).sum()

            if self.stats["kk"][layer_name] is None:
                self.stats["kk"][layer_name] = kk.cpu()
                self.stats["kv"][layer_name] = kv.cpu()
                self.stats["sum_x"][layer_name] = sx.cpu()
                self.stats["sum_y"][layer_name] = sy.cpu()
                self.stats["sum_yy"][layer_name] = yy.cpu()
            else:
                self.stats["kk"][layer_name] += kk.cpu()
                self.stats["kv"][layer_name] += kv.cpu()
                self.stats["sum_x"][layer_name] += sx.cpu()
                self.stats["sum_y"][layer_name] += sy.cpu()
                self.stats["sum_yy"][layer_name] += yy.cpu()
            self.stats["n"][layer_name] += x.shape[0]
        return hook


@torch.no_grad()
def collect_feature_stats(model, dataloader, class_tokens, layer_names=None, device="cuda"):
    layer_names = layer_names or make_clip_ffn_out_names()
    device = torch.device(device)
    model = getattr(model, "module", model)
    model.to(device)
    was_training = model.training
    model.eval()
    class_tokens = class_tokens.to(device)

    with FeatureCollector(model, layer_names) as collector:
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].long().to(device)
            text_tokens = class_tokens.index_select(0, labels)
            _ = model(images, text_tokens)
        stats = collector.stats

    if was_training:
        model.train()
    return stats


def kk_sum(stats):
    return sum(float(t.sum().item()) for t in stats["kk"].values())


def init_empty_like(stats):
    merged = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            merged[key] = {
                name: torch.zeros_like(t) if torch.is_tensor(t) else 0.0
                for name, t in value.items()
            }
    return merged


def accumulate_scaled(merged, stats, scale):
    for key in ("kk", "kv", "sum_x", "sum_y", "sum_yy"):
        if key not in stats:
            continue
        for name, tensor in stats[key].items():
            merged[key][name] += tensor * scale
    for name, value in stats["n"].items():
        merged["n"][name] += float(value) * scale


def _empty_payload(dataset_names, scale_reference):
    return {
        "metadata": {
            "datasets": [],
            "dataset_order": dataset_names,
            "scale_reference": scale_reference,
            "has_anchor": True,
            "target_scales": {},
            "anchor_scales": {},
            "target_manual_weights": {},
            "anchor_manual_weights": {},
            "target_final_weights": {},
            "anchor_final_weights": {},
            "target_kk_sums": {},
            "anchor_kk_sums": {},
            "target_reference_dataset": None,
            "anchor_reference_dataset": None,
            "target_reference_kk": None,
            "anchor_reference_kk": None,
        }
    }


def _merge_side(
    payload,
    side,
    dataset_name,
    stats,
    manual_weight,
    scale_reference,
    scale_reference_kk=None,
):
    metadata = payload["metadata"]
    side_key = side
    ref_dataset_key = f"{side}_reference_dataset"
    ref_kk_key = f"{side}_reference_kk"
    scales_key = f"{side}_scales"
    manual_key = f"{side}_manual_weights"
    final_key = f"{side}_final_weights"
    kk_key = f"{side}_kk_sums"

    current_kk = kk_sum(stats)
    if payload.get(side_key) is None:
        payload[side_key] = init_empty_like(stats)

    if scale_reference == "global":
        reference_kk = metadata.get(ref_kk_key)
        if reference_kk is None:
            if scale_reference_kk is None:
                raise ValueError(
                    "--scale-reference global requires a fixed kk reference value"
                )
            reference_kk = float(scale_reference_kk)
            metadata[ref_dataset_key] = "__global__"
            metadata[ref_kk_key] = reference_kk
        scale = float(reference_kk) / max(current_kk, 1e-12)
    elif metadata.get(ref_kk_key) is None:
        metadata[ref_dataset_key] = dataset_name
        metadata[ref_kk_key] = current_kk
        scale = 1.0
    elif scale_reference == "first":
        scale = float(metadata[ref_kk_key]) / max(current_kk, 1e-12)
    elif scale_reference == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unknown scale_reference: {scale_reference}")

    final_weight = scale * manual_weight
    accumulate_scaled(payload[side_key], stats, final_weight)

    metadata[scales_key][dataset_name] = scale
    metadata[manual_key][dataset_name] = manual_weight
    metadata[final_key][dataset_name] = final_weight
    metadata[kk_key][dataset_name] = current_kk
    print(
        f"[merge-{side}] {dataset_name}: kk_sum={current_kk:.6g}, "
        f"scale={scale:.6g}, manual={manual_weight:.6g}, final={final_weight:.6g}"
    )


def append_to_merged_kv(
    output_path,
    dataset_name,
    target_stats,
    anchor_stats,
    dataset_names=None,
    target_weights_arg="",
    anchor_weights_arg=None,
    scale_reference="first",
    target_scale_reference_kk=None,
    anchor_scale_reference_kk=None,
):
    dataset_names = parse_csv(dataset_names) or DEFAULT_SPLIT2_DATASETS
    target_weights = parse_weights_arg(target_weights_arg, dataset_names)
    anchor_weights_arg = target_weights_arg if anchor_weights_arg is None else anchor_weights_arg
    anchor_weights = parse_weights_arg(anchor_weights_arg, dataset_names)

    if dataset_name not in target_weights:
        raise KeyError(f"{dataset_name} is not in the configured KV dataset order")

    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            payload = pickle.load(f)
    else:
        payload = _empty_payload(dataset_names, scale_reference)

    metadata = payload["metadata"]
    if dataset_name in metadata.get("datasets", []):
        raise ValueError(
            f"{dataset_name} already exists in {output_path}. "
            "Remove the merged KV file before rerunning from the beginning."
        )

    _merge_side(
        payload, "target", dataset_name, target_stats,
        target_weights[dataset_name], scale_reference, target_scale_reference_kk,
    )
    _merge_side(
        payload, "anchor", dataset_name, anchor_stats,
        anchor_weights[dataset_name], scale_reference, anchor_scale_reference_kk,
    )
    metadata["datasets"].append(dataset_name)
    metadata["has_anchor"] = True

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] merged KV updated: {output_path}")
