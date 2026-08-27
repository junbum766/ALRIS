from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import yaml
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from al.active_learning import (
    RandomSampling,
    RegionEntropy,
    full_gt_cost_separate,
    load_cost_tables_separate,
)
from engine.engine import inference, trainMetricGPU, validate
from model import build_segmenter_detris
from utils import config
from utils.dataset import RefDataset, RefDatasetCorrection
from utils.misc import (
    AverageMeter,
    ProgressMeter,
    init_random_seed,
    set_random_seed,
    setup_logger,
    worker_init_fn,
)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def gather_lists(lst):
    if dist.get_world_size() == 1:
        return lst
    buf = [None] * dist.get_world_size()
    dist.all_gather_object(buf, lst)
    return sum(buf, []) if dist.get_rank() == 0 else []


def write_split_summary(metrics_dict, out_dir):
    lines = []
    for split in ("val", "testA", "testB", "test"):
        if split in metrics_dict and (metrics_dict[split]["miou"] or metrics_dict[split]["oiou"]):
            miou_list = ", ".join(f"{x*100:.2f}" for x in metrics_dict[split]["miou"])
            oiou_list = ", ".join(f"{x*100:.2f}" for x in metrics_dict[split]["oiou"])
            lines.append(f"{split} mIoU : [{miou_list}]")
            lines.append(f"{split} oIoU : [{oiou_list}]")
    (Path(out_dir) / "summary.txt").write_text("\n".join(lines) + "\n")


def train_one_epoch(loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter("Batch", ":2.2f")
    data_time  = AverageMeter("Data",  ":2.2f")
    lr_meter   = AverageMeter("LR",    ":1.6f")
    loss_meter = AverageMeter("Loss",  ":2.4f")
    iou_meter  = AverageMeter("IoU",   ":2.2f")
    pr_meter   = AverageMeter("Prec@50", ":2.2f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, lr_meter, loss_meter, iou_meter, pr_meter],
        prefix=f"Train Epoch [{epoch}/{args.epochs}] ",
    )
    model.train()
    end = time.time()
    for it, (img, txt, tgt, idx, _) in enumerate(loader):
        img, txt, tgt = img.cuda(), txt.cuda(), tgt.cuda().unsqueeze(1)
        with amp.autocast():
            pred, tgt, loss = model(img, txt, tgt)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        iou, pr5, _ = trainMetricGPU(pred, tgt, 0.35, 0.5, return_per_sample=True)
        dist.all_reduce(loss.detach()); dist.all_reduce(iou); dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou  = iou  / dist.get_world_size()
        pr5  = pr5  / dist.get_world_size()
        loss_meter.update(loss.item(), img.size(0))
        iou_meter.update(iou.item(),   img.size(0))
        pr_meter.update(pr5.item(),    img.size(0))
        lr_meter.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end); end = time.time()
        if (it + 1) % args.print_freq == 0:
            progress.display(it + 1)


def main_worker(gpu: int, args):
    start_time = time.time()
    args.gpu  = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(gpu)
    setup_logger(args.output_dir, distributed_rank=gpu, filename="train.log", mode="a")
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(seconds=28800),
    )
    if args.rank == 0:
        (Path(args.output_dir) / "run_cfg.yaml").write_text(yaml.safe_dump(dict(args), sort_keys=False))
    dist.barrier()

    split_metrics = {"val": {"miou": [], "oiou": []}}
    if args.dataset in ("refcoco", "refcoco+"):
        split_metrics["testA"] = {"miou": [], "oiou": []}
        split_metrics["testB"] = {"miou": [], "oiou": []}
    if args.dataset == "refcocog_u":
        split_metrics["test"] = {"miou": [], "oiou": []}

    if args.rank == 0:
        mask_cost, text_cost = load_cost_tables_separate(
            args.dataset, args.mask_cost_json, args.text_cost_json, args.cost_key
        )
    else:
        mask_cost, text_cost = {}, {}
    obj = [mask_cost, text_cost]
    dist.broadcast_object_list(obj, src=0)
    mask_cost, text_cost = obj

    totals = full_gt_cost_separate(args.dataset)
    total_mask_cost = int(totals["mask"])
    total_text_cost = int(totals["text"])
    total_gt_cost   = total_mask_cost + total_text_cost

    mapping_path = f"datasets/meta/pseudo_to_gt_{args.dataset}.json"
    with open(mapping_path, "r") as f:
        pseudo_to_gt_map = {int(k): v for k, v in json.load(f).items()}

    ps_ds = RefDatasetCorrection(
        lmdb_dir=args.train_lmdb_gsam_llava,
        mask_dir=args.gsam_mask_root,
        dataset=args.dataset, split=args.train_split_gsam, mode="train",
        input_size=args.input_size, word_length=args.word_len,
    )
    cor_ds = RefDatasetCorrection(
        lmdb_dir=args.train_lmdb,
        mask_dir=args.train_mask_root_cor,
        dataset=args.dataset, split=args.train_split, mode="train",
        input_size=args.input_size, word_length=args.word_len,
        use_first_only=getattr(args, "use_first_only", False),
    )
    val_ds = RefDataset(
        lmdb_dir=args.val_lmdb, mask_dir=args.mask_root_gt,
        dataset=args.dataset, split=args.val_split, mode="val",
        input_size=args.input_size, word_length=args.word_len,
    )
    test_ds_val = RefDataset(
        lmdb_dir=args.test_lmdb_val, mask_dir=args.mask_root_gt,
        dataset=args.dataset, split="val", mode="test",
        input_size=args.input_size, word_length=args.word_len,
    )
    if args.dataset in ("refcoco", "refcoco+"):
        test_ds_testA = RefDataset(args.test_lmdb_testA, args.mask_root_gt, args.dataset, "testA", "test", args.input_size, args.word_len)
        test_ds_testB = RefDataset(args.test_lmdb_testB, args.mask_root_gt, args.dataset, "testB", "test", args.input_size, args.word_len)
    if args.dataset == "refcocog_u":
        test_ds_test = RefDataset(args.test_lmdb_test, args.mask_root_gt, args.dataset, "test", "test", args.input_size, args.word_len)

    N_pseudo = len(ps_ds)
    N_gt     = len(cor_ds)

    if args.rank == 0:
        logger.info("Building model (fresh init weights)...")
    init_weights = build_segmenter_detris(args)[0].state_dict()
    dist.barrier()

    cold_start = RandomSampling(args, dataset_size=N_pseudo, seed=args.manual_seed)
    if args.label_mode == "random":
        al = cold_start
    elif args.label_mode == "region_entropy":
        al = RegionEntropy(args=args, dataset=ps_ds,
                           dataset_size=N_pseudo, seed=args.manual_seed,
                           device="cuda")
    else:
        raise ValueError(
            f"Unknown label_mode {args.label_mode!r}. "
            "Release build supports only 'random' and 'region_entropy'.")

    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batch_size_val // args.ngpus_per_node,
        shuffle=False, num_workers=args.workers_val, pin_memory=True,
        sampler=data.distributed.DistributedSampler(val_ds, shuffle=False),
    )
    test_loader_val = data.DataLoader(test_ds_val, batch_size=1, shuffle=False, num_workers=1)
    if args.dataset in ("refcoco", "refcoco+"):
        test_loader_testA = data.DataLoader(test_ds_testA, batch_size=1, shuffle=False, num_workers=1)
        test_loader_testB = data.DataLoader(test_ds_testB, batch_size=1, shuffle=False, num_workers=1)
    if args.dataset == "refcocog_u":
        test_loader_test = data.DataLoader(test_ds_test, batch_size=1, shuffle=False, num_workers=1)

    cor_set: set = set()
    cum_cost = cum_mask_cost = cum_text_cost = 0.0
    total_sample = 0

    round_idx = 0
    while round_idx <= len(args.target_pcts):
        round_dir = Path(args.output_dir) / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if args.rank == 0:
            sampler_for_round = cold_start if round_idx == 0 else al
            target_pct = args.target_pcts[round_idx] if round_idx < len(args.target_pcts) else None
            if target_pct is None:
                logger.info("All AL rounds finished.")
                stop_flag = torch.tensor(1, device="cuda")
            else:
                target_cum_cost      = total_gt_cost  * target_pct
                target_cum_mask_cost = total_mask_cost * target_pct
                target_cum_text_cost = total_text_cost * target_pct
                round_budget      = int(target_cum_cost      - cum_cost)
                round_mask_budget = int(target_cum_mask_cost - cum_mask_cost)
                round_text_budget = int(target_cum_text_cost - cum_text_cost)
                logger.info(
                    f"[Round {round_idx}] target {target_pct*100:.1f}% — "
                    f"budget total={round_budget:,d} (mask={round_mask_budget:,d}, text={round_text_budget:,d})"
                )

                if round_idx > 0:
                    sampler_for_round.set_model(
                        model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                    )

                selected_pseudos, spent, m_spent, t_spent, sel_scores = (
                    sampler_for_round.select_query_with_rejection(
                        mask_cost=mask_cost, text_cost=text_cost,
                        round_idx=round_idx,
                        combined_budget=round_budget,
                        mask_budget=round_mask_budget,
                        text_budget=round_text_budget,
                        reject_cost=args.reject_cost,
                    )
                )

                if not selected_pseudos:
                    logger.warning("No samples fit within round budget — stopping.")
                    stop_flag = torch.tensor(1, device="cuda")
                else:
                    accepted_gts = {sampler_for_round.gt_index_of(p) for p in selected_pseudos
                                    if sampler_for_round.gt_index_of(p) != -1}
                    num_rejected = len(selected_pseudos) - len(accepted_gts)
                    cor_set.update(accepted_gts)
                    al.update(list(selected_pseudos))     # always track in main sampler
                    if cold_start is not al:
                        cold_start.update(list(selected_pseudos))

                    cum_cost      += int(spent)
                    cum_mask_cost += int(m_spent)
                    cum_text_cost += int(t_spent)
                    total_sample  += len(accepted_gts)
                    pct_total = 100.0 * cum_cost      / max(1, total_gt_cost)
                    pct_mask  = 100.0 * cum_mask_cost / max(1, total_mask_cost)
                    pct_text  = 100.0 * cum_text_cost / max(1, total_text_cost)
                    logger.info(
                        f"[Round {round_idx}] Queried {len(selected_pseudos)} "
                        f"(accept={len(accepted_gts)}, reject={num_rejected}) | "
                        f"GTs={total_sample}/{N_gt} | "
                        f"cum total={int(cum_cost):,d} ({pct_total:.2f}%), "
                        f"mask={int(cum_mask_cost):,d} ({pct_mask:.2f}%), "
                        f"text={int(cum_text_cost):,d} ({pct_text:.2f}%)"
                    )
                    (round_dir / "labeled_set.json").write_text(json.dumps(sorted(cor_set)))
                    (round_dir / "selected_score.json").write_text(
                        json.dumps({str(k): v for k, v in sorted(sel_scores.items())}, indent=2)
                    )
                    stop_flag = torch.tensor(0, device="cuda")
        else:
            stop_flag = torch.tensor(0, device="cuda")

        dist.broadcast(stop_flag, src=0)
        obj = [list(cor_set), cum_cost, cum_mask_cost, cum_text_cost,
               total_sample, list(al._labeled_mask)] if args.rank == 0 else [None]*6
        dist.broadcast_object_list(obj, src=0)
        if args.rank != 0:
            cor_set      = set(obj[0])
            cum_cost     = obj[1]
            cum_mask_cost = obj[2]
            cum_text_cost = obj[3]
            total_sample = obj[4]
            al._labeled_mask = list(obj[5])
            if cold_start is not al:
                cold_start._labeled_mask = list(obj[5])
        if stop_flag.item():
            break

        model, _ = build_segmenter_detris(args)
        model.load_state_dict(init_weights, strict=False)
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu], find_unused_parameters=True
        )

        train_ds = Subset(cor_ds, list(cor_set))
        sampler = data.distributed.DistributedSampler(train_ds, shuffle=True)
        loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size // args.ngpus_per_node,
            shuffle=False,
            num_workers=max(1, args.workers // args.ngpus_per_node),
            pin_memory=True,
            sampler=sampler,
            worker_init_fn=partial(
                worker_init_fn, num_workers=args.workers,
                rank=args.rank, seed=args.manual_seed,
            ),
            drop_last=False,
        )

        if hasattr(args, "epochs_list") and round_idx < len(args.epochs_list):
            args.epochs = int(args.epochs_list[round_idx])

        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
        scaler = amp.GradScaler()

        best_miou = best_oiou = 0.0
        best_miou_epoch = best_oiou_epoch = 0
        for ep in range(1, args.epochs + 1):
            sampler.set_epoch(ep)
            train_one_epoch(loader, model, optimizer, scheduler, scaler, ep, args)
            scheduler.step(ep)
            cur_miou, cur_oiou, _ = validate(val_loader, model, ep, args)
            if args.rank == 0:
                last_ckpt = os.path.join(round_dir, "last_model.pth")
                torch.save({
                    "epoch": ep, "miou": cur_miou, "oiou": cur_oiou,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, last_ckpt)
                if cur_miou > best_miou:
                    best_miou = cur_miou; best_miou_epoch = ep
                    torch.save(torch.load(last_ckpt),
                               os.path.join(round_dir, "best_model.pth"))
                if cur_oiou > best_oiou:
                    best_oiou = cur_oiou; best_oiou_epoch = ep
                logger.info(
                    f"[Round {round_idx}] Epoch {ep} mIoU={cur_miou:.2f} oIoU={cur_oiou:.2f} | "
                    f"best mIoU={best_miou:.2f}(ep{best_miou_epoch}), oIoU={best_oiou:.2f}(ep{best_oiou_epoch})"
                )
            dist.barrier()

        best_ckpt = os.path.join(round_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            best_state = strip_module_prefix(torch.load(best_ckpt, map_location="cpu")["state_dict"])
            model.module.load_state_dict(best_state, strict=False)
            if args.rank == 0:
                logger.info("Loaded best_model.pth for inference.")
        if args.rank == 0:
            val_miou, val_oiou, _ = inference(test_loader_val, model, args, "val")
            split_metrics["val"]["miou"].append(val_miou)
            split_metrics["val"]["oiou"].append(val_oiou)
            if args.dataset in ("refcoco", "refcoco+"):
                ta_miou, ta_oiou, _ = inference(test_loader_testA, model, args, "testA")
                tb_miou, tb_oiou, _ = inference(test_loader_testB, model, args, "testB")
                split_metrics["testA"]["miou"].append(ta_miou); split_metrics["testA"]["oiou"].append(ta_oiou)
                split_metrics["testB"]["miou"].append(tb_miou); split_metrics["testB"]["oiou"].append(tb_oiou)
            if args.dataset == "refcocog_u":
                t_miou, t_oiou, _ = inference(test_loader_test, model, args, "test")
                split_metrics["test"]["miou"].append(t_miou); split_metrics["test"]["oiou"].append(t_oiou)
            write_split_summary(split_metrics, args.output_dir)
        dist.barrier()

        round_idx += 1
        gc.collect(); torch.cuda.empty_cache()

    if args.rank == 0:
        logger.info(f"Training finished in {time.time() - start_time:.2f}s.")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)
    if cli_args.opts:
        cfg = config.merge_cfg_from_list(cfg, cli_args.opts)
    args = cfg

    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed)
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))


if __name__ == "__main__":
    main()
    sys.exit(0)
#