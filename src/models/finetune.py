import copy
import os

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from .. import datasets, templates, utils
from ..datasets.common import maybe_dictionarize
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
from .kv_stats import append_to_merged_kv, collect_feature_stats, make_clip_ffn_out_names


def _as_int_label(label):
    if isinstance(label, torch.Tensor):
        return int(label.item())
    if isinstance(label, (tuple, list)):
        return _as_int_label(label[0])
    return int(label)


def _dataset_targets(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base_targets = _dataset_targets(dataset.dataset)
        return [base_targets[i] for i in dataset.indices]

    for attr in ("targets", "_labels", "labels", "_targets"):
        if hasattr(dataset, attr):
            return [_as_int_label(x) for x in getattr(dataset, attr)]

    for attr in ("samples", "_samples", "imgs"):
        if hasattr(dataset, attr):
            return [_as_int_label(x[1]) for x in getattr(dataset, attr)]

    return [_as_int_label(dataset[i][1]) for i in range(len(dataset))]


def _build_fewshot_loader(dataset, fallback_loader, args, num_workers):
    if args.few_shot is None or args.few_shot <= 0:
        return fallback_loader

    print("=====few-shot======")
    few_shot_data = {}
    for batch in fallback_loader:
        batch = maybe_dictionarize(batch)
        for image, label in zip(batch["images"], batch["labels"]):
            label = _as_int_label(label)
            if label not in few_shot_data:
                few_shot_data[label] = []
            if len(few_shot_data[label]) < args.few_shot:
                few_shot_data[label].append(image)

    few_shot_images = []
    few_shot_labels = []
    for label, images in few_shot_data.items():
        few_shot_images.extend(images)
        few_shot_labels.extend([label] * len(images))

    if not few_shot_images:
        raise ValueError("No samples were selected for few-shot training")

    few_shot_dataset = torch.utils.data.TensorDataset(
        torch.stack(few_shot_images), torch.tensor(few_shot_labels)
    )
    loader = DataLoader(few_shot_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(loader))
    return loader


def _select_trainable_params(model, train_mode):
    if train_mode == "text":
        print("[Training mode] Text Encoder")
        visual_params_name = [k for k, _ in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        named_params = [
            (k, v) for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif train_mode == "image":
        print("[Training mode] Image Encoder")
        named_params = list(model.visual.named_parameters())
    elif train_mode == "image_ffn_out":
        print("[Training mode] Image Encoder FFN output layers")
        keywords = ("mlp.c_proj", "mlp.fc2", "mlp.linear2")
        named_params = [
            (k, v)
            for k, v in model.named_parameters()
            if "visual" in k and any(key in k for key in keywords)
        ]
    elif train_mode == "ffn_out":
        print("[Training mode] Visual and text FFN output layers")
        keywords = ("mlp.c_proj", "mlp.fc2", "mlp.linear2")
        named_params = [
            (k, v) for k, v in model.named_parameters() if any(key in k for key in keywords)
        ]
    elif train_mode == "full_ffn":
        print("[Training mode] Full FFN layers")
        keywords = (
            "mlp.c_fc",
            "mlp.fc1",
            "mlp.linear1",
            "mlp.c_proj",
            "mlp.fc2",
            "mlp.linear2",
        )
        named_params = [
            (k, v) for k, v in model.named_parameters() if any(key in k for key in keywords)
        ]
    else:
        assert train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]
        named_params = [
            (k, v) for k, v in model.named_parameters() if k not in exclude_params_name
        ]

    if not named_params:
        raise ValueError(f"No trainable parameters found for train_mode={train_mode}")

    print(f"[INFO] Found {len(named_params)} trainable parameter tensors:")
    for name, param in named_params:
        print(f"  - {name}: {tuple(param.shape)}")
    return named_params



def _register_random_gradient_masks(named_params, mask_ratio, seed):
    if mask_ratio <= 0:
        return []
    if mask_ratio >= 1:
        raise ValueError("--random-mask-ratio must be in [0, 1)")

    handles = []
    print(f"[Mask] Randomly freezing {mask_ratio * 100:.1f}% of trainable parameter elements")
    for idx, (name, param) in enumerate(named_params):
        generator = torch.Generator(device=param.device)
        generator.manual_seed(int(seed) + idx)
        train_mask = (
            torch.rand(param.shape, device=param.device, generator=generator) >= mask_ratio
        ).to(dtype=param.dtype)
        frozen_ratio = 1.0 - float(train_mask.float().mean().item())
        print(f"  - {name}: frozen={frozen_ratio * 100:.2f}%")
        handles.append(param.register_hook(lambda grad, mask=train_mask: grad * mask))
    return handles


def finetune(args):
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
    if args.load is not None:
        utils.torch_load(model, args.load)

    should_collect_kv = args.merged_kv_output
    initial_model = copy.deepcopy(model) if should_collect_kv else None

    if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"):
        print("Using WiSE-FT with Loaded Model")
        model_fix, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if args.load is not None:
            utils.torch_load(model_fix, args.load)

    if args.we or args.moving_avg or args.we_wise:
        print("Averaging training")
        if args.moving_avg and args.mv_avg_model == "zeroshot": # mv+zeroshot
            we_model, _, _ =  clip.load(args.model, jit=False)
            we_model.cuda()
            we_n = 0
        else: #we; mv+m; mv+t; we_wise
            we_model = copy.deepcopy(model)
            we_model.cuda()
            we_n = 0
    if args.l2 > 0:
        print("L2 norm")
        l2_model = copy.deepcopy(model)
        l2_model.cuda()

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    train_loader = _build_fewshot_loader(
        dataset.train_dataset, dataset.train_loader, args, dataset.num_workers
    )

    # number of iterations
    num_batches = len(train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    named_params = _select_trainable_params(model, args.train_mode)
    params = [param for _, param in named_params]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    mask_seed = args.random_mask_seed if args.random_mask_seed is not None else args.seed
    _register_random_gradient_masks(named_params, args.random_mask_ratio, mask_seed)
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method
    if args.method == "ZSCL":
        # (Ref Model) get reference model
        print("[Method] ZSCL")
        if args.ref_model is None:
            if args.ref_wise:
                print("[ref_model] WiSE-Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
                for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                    param_q.data = param_q.data * (1 - args.ref_wise_alpha) + param_k.data * args.ref_wise_alpha
            else:    
                print("[ref_model] Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        else:
            print(f"[ref_model] {args.ref_model}")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(
                ref_model, args.ref_model
            )
        ref_model = ref_model.cuda()
        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model.eval()

        # (Ref Dataset) get reference dataset
        ref_dataset_cls = getattr(datasets, args.ref_dataset)
        print(f"[Ref Dataset] {args.ref_dataset}")
        if args.ref_dataset in ["ImageNetSM", "ImageNetSUB"]:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num=args.num,
            )
        else:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
        ref_iter = iter(ref_dataset.train_loader)

        # (Ref Text) get reference text
        if args.text_datasets is not None:
            print("[Ref Sentences] Text-Datasets")
            ref_texts = get_datasets_text(args.text_datasets, args)
        elif args.ref_sentences == "random":
            ref_texts = virtual_vocab()
            print("[Ref Sentences] Random Sentences")
        elif args.ref_sentences is not None:
            ref_sentences_cls = getattr(datasets, args.ref_sentences)
            print(f"[Ref Sentences] {args.ref_sentences}")
            ref_sentences = ref_sentences_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
            if args.ref_sentences == "conceptual_captions":
                # breakpoint()
                ref_texts = ref_sentences.train_dataset.captions
                ref_texts = clip.tokenize(ref_texts).cuda()

            else:
                ref_template = ref_sentences.template
                ref_texts = [ref_template(x) for x in ref_sentences.classnames]
                ref_texts = clip.tokenize(ref_texts).cuda()
        else:
            print(f"[Ref Sentences] {args.ref_dataset}")
            ref_template = ref_dataset.template
            ref_texts = [ref_template(x) for x in ref_dataset.classnames]
            ref_texts = clip.tokenize(ref_texts).cuda()
            
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if eval_iterations is not None and iteration % eval_iterations == 0:
            if args.we or args.we_wise:
                evaluate(we_model, args, val_preprocess)
            else:
                evaluate(model.module, args, val_preprocess)

        # training
        if iteration % num_batches == 0:
            data_iter = iter(train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        try:
            train_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            train_batch = next(data_iter)
        train_batch = maybe_dictionarize(train_batch)
        images, labels = train_batch["images"], train_batch["labels"]
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        if args.l2 > 0:
            loss_l2 = l2_loss(model, l2_model)
            loss += args.l2 * loss_l2

        if args.method == "ZSCL":

            # FIXME: only for ImageNet
            if args.ref_dataset in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                try:
                    ref_batch = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_batch = next(ref_iter)
                ref_images, ref_labels = ref_batch["images"], ref_batch["labels"]
            else:
                try:
                    ref_images, ref_labels = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_images, ref_labels = next(ref_iter)
            ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()
            # breakpoint()
            with torch.no_grad():
                # -- get ref text embedding --
                ref_embeddings = ref_model(None, ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # -- get ref image embedding --
                ref_out = ref_model(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            ref_out_current = model(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            # -- loss --
            logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
            loss_ZSCL = distillation(logits_ref, logits_current, T=args.T)

            # feature-space mse
            if args.feature_mse:
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(ref_out, ref_out_current)

            # -- final loss --
            if args.image_loss:
                if args.weight_adjust:
                    loss = loss + 0.5 * loss_ZSCL 
                else:
                    loss = loss + 1.0 * loss_ZSCL 

            # transpose loss
            if args.text_loss:
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_2
                else:
                    loss += loss_ZSCL_2
            
            if args.ablation_loss_2:
                logits_img_current = logit_scale.exp() * ref_out_current @ ref_out_current.t()
                logits_img_ref = logit_scale.exp() * ref_out @ ref_out.t()
                logits_img_current -= torch.diag(logits_img_current.diag() + 1e4)
                logits_img_ref -= torch.diag(logits_img_ref.diag() + 1e4)
                loss_ZSCL_3 = distillation(logits_img_ref, logits_img_current, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_3
                else:
                    loss += loss_ZSCL_3


        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # we
        if (args.we or args.moving_avg or args.we_wise) and iteration % args.avg_freq == 0:
            we_n += 1
            if args.moving_avg:
                if args.mv_avg_model == "t":
                    next_we_model = copy.deepcopy(model.module)
                    moving_avg(model.module, we_model, args.mv_avg_decay)
                    we_model = next_we_model.cuda()
                else: ### args.moving_avg_model == "n" or "zeroshot"
                    moving_avg(model.module, we_model, args.mv_avg_decay)
            elif args.we:
                merge_we(model.module, we_model, we_n)
            else:
                wise_we(model.module, we_model, we_n, model_fix, args.we_wise_alpha)

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
            if args.method == "ZSCL":
                print("Loss ZSCL:", loss_ZSCL.item())
            if args.l2 > 0:
                print("Loss L2:", loss_l2.item())

    if args.wise_merge:
        alpha = args.wise_ft_alpha
        if args.wise_ft_model == "zeroshot":
            wise_ft_model, _, _ =  clip.load(args.model, jit=False)
        else:
            wise_ft_model = copy.deepcopy(model_fix)
        wise_ft_model.cuda()
        for param_q, param_k in zip(model.module.parameters(), wise_ft_model.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)



    if args.we or args.we_wise:
        to_save_model = we_model
    else:
        to_save_model = model.module

    if args.post_train_eval:
        if args.eval_datasets is None:
            args.eval_datasets = [args.train_dataset]
        print(f"[post-train-eval] {args.train_dataset}")
        evaluate(to_save_model, args, val_preprocess)

    # Saving model is optional in the MTIL KV workflow; the default open-source
    # path keeps only the merged KV statistics and avoids 22 expert checkpoints.
    if args.save is not None:
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)

    if should_collect_kv:
        print(f"[kv] collecting target stats for {args.train_dataset}")
        layer_names = make_clip_ffn_out_names()
        target_stats = collect_feature_stats(
            to_save_model, train_loader, texts, layer_names=layer_names, device="cuda"
        )
        to_save_model.cpu()
        torch.cuda.empty_cache()

        print(f"[kv] collecting anchor stats for {args.train_dataset}")
        anchor_stats = collect_feature_stats(
            initial_model, train_loader, texts, layer_names=layer_names, device="cuda"
        )
        initial_model.cpu()
        torch.cuda.empty_cache()

        if args.merged_kv_output:
            append_to_merged_kv(
                output_path=args.merged_kv_output,
                dataset_name=args.train_dataset,
                target_stats=target_stats,
                anchor_stats=anchor_stats,
                dataset_names=args.merged_kv_datasets,
                target_weights_arg=args.target_weights,
                anchor_weights_arg=args.anchor_weights,
                scale_reference=args.scale_reference,
                target_scale_reference_kk=args.target_scale_reference_kk,
                anchor_scale_reference_kk=args.anchor_scale_reference_kk,
            )
