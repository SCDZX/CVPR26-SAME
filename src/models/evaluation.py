import clip

import torch
from tqdm import tqdm

from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def _as_template_list(templates):
    if isinstance(templates, list):
        return templates
    return [templates]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    zeroshot_weights = []
    for classname in classnames:
        texts = [
            template(classname) for template in _as_template_list(templates)
        ]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_classifier_from_global_entries(entries, model):
    zeroshot_weights = []
    for entry in entries:
        texts = [
            template(entry["classname"])
            for template in _as_template_list(entry["templates"])
        ]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def _resolve_global_label_datasets(args):
    dataset_names = getattr(args, "global_label_datasets", None)
    if dataset_names is None:
        dataset_names = getattr(datasets, "mtil_split2_dataset_names")
    dataset_names = [name.strip() for name in dataset_names if name.strip()]
    if not dataset_names:
        raise ValueError("The global label space is empty")
    return dataset_names


def build_global_label_eval_state(args, model):
    dataset_names = _resolve_global_label_datasets(args)
    entries = []
    label_maps = {}

    for dataset_name in dataset_names:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            num_workers=0,
        )

        local_to_global = {}
        for local_label, classname in enumerate(dataset.classnames):
            global_label = len(entries)
            local_to_global[local_label] = global_label
            entries.append(
                {
                    "dataset": dataset_name,
                    "local_label": local_label,
                    "global_label": global_label,
                    "classname": classname,
                    "templates": dataset.templates,
                }
            )
        label_maps[dataset_name] = local_to_global

    print(
        f"Using fixed global label space: {len(entries)} classes "
        f"from {len(dataset_names)} MTIL split tasks."
    )
    zeroshot_weights = zeroshot_classifier_from_global_entries(entries, model)
    return {
        "dataset_names": dataset_names,
        "entries": entries,
        "label_maps": label_maps,
        "zeroshot_weights": zeroshot_weights,
    }


def _map_targets_to_global(target, label_map):
    try:
        mapped = [label_map[int(label)] for label in target.detach().cpu().tolist()]
    except KeyError as exc:
        raise ValueError(
            f"Local label {exc.args[0]} is missing from the global label map"
        ) from exc
    return torch.tensor(mapped, device=target.device, dtype=target.dtype)


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights, label_map=None):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()
        if label_map is not None:
            target = _map_targets_to_global(target, label_map)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args, dataset_name, global_eval_state=None):
    model = image_classifier
    image_enc = None

    model.eval()

    label_map = None
    if global_eval_state is None:
        zeroshot_weights = zeroshot_classifier(
            dataset.classnames, dataset.templates, model
        )
    else:
        if dataset_name not in global_eval_state["label_maps"]:
            raise ValueError(
                f"{dataset_name} is not included in --global-label-datasets"
            )
        label_map = global_eval_state["label_maps"][dataset_name]
        global_ids = list(label_map.values())
        print(
            f"Evaluating {dataset_name} with fixed global ids "
            f"{min(global_ids)}..{max(global_ids)} for its local labels."
        )
        zeroshot_weights = global_eval_state["zeroshot_weights"]

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights, label_map)

    print(f"Top-1 accuracy: {top1:.2f}")


def evaluate(image_classifier, args, val_preprocess):
    if args.eval_datasets is None:
        return

    global_eval_state = None
    if getattr(args, "eval_all_labels", False):
        global_eval_state = build_global_label_eval_state(args, image_classifier)

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset(
            image_classifier, dataset, args, dataset_name, global_eval_state
        )
