"""Stage-1 Complementary 1408px Mask R-CNN v2 Instance Detector (FP16 Weight Export + Multi-Fold Clustering)."""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import boxes as box_ops

from physics_v2 import IMG_SIZE, load_cached_physics_u8
from targets_v10 import StemTargets

DET_SIZE = 1408


@dataclass
class ProposalROI:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in native 2048x2048
    mask_f16: np.ndarray             # shape (y2-y1, x2-x1), float16 in [0, 1]
    s_det: float
    view_agree: float


def compute_active_disk_box(geom: dict[str, float]) -> tuple[int, int, int]:
    """Return (x0, y0, side) enclosing the solar chromosphere with 14 px margin."""
    cx, cy, R_sun = geom["cx"], geom["cy"], geom["R_sun"]
    half = R_sun + 14.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    side = int(round(2.0 * half))
    return x0, y0, max(512, side)


def _crop_and_resize_disk(hw3: np.ndarray, x0: int, y0: int, side: int, det_size: int = DET_SIZE) -> np.ndarray:
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[..., 0] = 185
    sx1, sy1 = max(0, x0), max(0, y0)
    sx2, sy2 = min(IMG_SIZE, x0 + side), min(IMG_SIZE, y0 + side)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x0, sy1 - y0
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        canvas[dy1:dy2, dx1:dx2] = hw3[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (det_size, det_size), interpolation=cv2.INTER_LINEAR)


def _apply_d4_img(img: np.ndarray, k_rot: int, hflip: bool) -> np.ndarray:
    out = np.rot90(img, k_rot, axes=(0, 1)) if k_rot else img
    if hflip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def _apply_d4_boxes(boxes: np.ndarray, size: int, k_rot: int, hflip: bool) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    for _ in range(k_rot % 4):
        x1, y1, x2, y2 = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
        b[:, 0] = y1
        b[:, 1] = size - x2
        b[:, 2] = y2
        b[:, 3] = size - x1
    if hflip:
        x1, x2 = b[:, 0].copy(), b[:, 2].copy()
        b[:, 0] = size - x2
        b[:, 2] = size - x1
    return b


class Stage1TrainDataset(Dataset):
    def __init__(
        self,
        stems: list[str],
        ram_physics: dict[str, np.ndarray] | None,
        cache_dir: Path,
        geom_map: dict[str, dict[str, float]],
        targets_map: dict[str, StemTargets],
        det_size: int = DET_SIZE,
        augment: bool = True,
    ) -> None:
        self.stems = list(stems)
        self.ram_physics = ram_physics
        self.cache_dir = cache_dir
        self.geom_map = geom_map
        self.targets_map = targets_map
        self.det_size = det_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stem = self.stems[idx]
        hw3 = self.ram_physics[stem] if (self.ram_physics and stem in self.ram_physics) else load_cached_physics_u8(self.cache_dir, stem)
        geom = self.geom_map[stem]
        x0, y0, side = compute_active_disk_box(geom)
        img_det = _crop_and_resize_disk(hw3, x0, y0, side, self.det_size)
        scale = float(self.det_size) / float(side)

        st = self.targets_map[stem]
        rng = np.random.default_rng()
        k_rot = int(rng.integers(0, 4)) if self.augment else 0
        hflip = bool(rng.integers(0, 2)) if self.augment else False
        if k_rot or hflip:
            img_det = _apply_d4_img(img_det, k_rot, hflip)

        valid_crops: list[tuple[int, int, int, int, np.ndarray]] = []
        boxes_list: list[list[float]] = []
        for inst in st.instances:
            bx1, by1, bx2, by2 = inst.bbox
            crop = inst.mask_crop
            nx1 = int(np.clip(np.floor((bx1 - x0) * scale), 0, self.det_size - 1))
            ny1 = int(np.clip(np.floor((by1 - y0) * scale), 0, self.det_size - 1))
            nx2 = int(np.clip(np.ceil((bx2 - x0) * scale), nx1 + 1, self.det_size))
            ny2 = int(np.clip(np.ceil((by2 - y0) * scale), ny1 + 1, self.det_size))
            sub_res = cv2.resize(crop, (nx2 - nx1, ny2 - ny1), interpolation=cv2.INTER_NEAREST)
            sub_d4 = np.rot90(sub_res, k_rot, axes=(0, 1)) if k_rot else sub_res
            if hflip:
                sub_d4 = sub_d4[:, ::-1]
            sub_d4 = np.ascontiguousarray(sub_d4)
            rbox = _apply_d4_boxes(np.array([[nx1, ny1, nx2, ny2]], dtype=np.float32), self.det_size, k_rot, hflip)[0]
            rx1, ry1, rx2, ry2 = int(round(rbox[0])), int(round(rbox[1])), int(round(rbox[2])), int(round(rbox[3]))
            ys, xs = np.nonzero(sub_d4)
            if xs.size < 6:
                continue
            fx1, fy1, fx2, fy2 = rx1 + int(xs.min()), ry1 + int(ys.min()), rx1 + int(xs.max()) + 1, ry1 + int(ys.max()) + 1
            if fx2 - fx1 < 2 or fy2 - fy1 < 2:
                continue
            valid_crops.append((rx1, ry1, rx2, ry2, sub_d4))
            boxes_list.append([fx1, fy1, fx2, fy2])

        ign_boxes_list = []
        if st.ignore_crop is not None:
            ix1, iy1, ix2, iy2, icrop = st.ignore_crop
            nx1 = int(np.clip(np.floor((ix1 - x0) * scale), 0, self.det_size - 1))
            ny1 = int(np.clip(np.floor((iy1 - y0) * scale), 0, self.det_size - 1))
            nx2 = int(np.clip(np.ceil((ix2 - x0) * scale), nx1 + 1, self.det_size))
            ny2 = int(np.clip(np.ceil((iy2 - y0) * scale), ny1 + 1, self.det_size))
            ign_sub = cv2.resize(icrop, (nx2 - nx1, ny2 - ny1), interpolation=cv2.INTER_NEAREST)
            n_cc, _, stats, _ = cv2.connectedComponentsWithStats(ign_sub, connectivity=8)
            for c_id in range(1, n_cc):
                cx_i, cy_i, iw, ih, ia = stats[c_id]
                if ia >= 10 and iw >= 2 and ih >= 2:
                    ign_boxes_list.append([nx1 + cx_i, ny1 + cy_i, nx1 + cx_i + iw, ny1 + cy_i + ih])

        if valid_crops:
            masks_arr = np.zeros((len(valid_crops), self.det_size, self.det_size), dtype=np.uint8)
            for k_c, (rx1, ry1, rx2, ry2, sub_d4) in enumerate(valid_crops):
                masks_arr[k_c, ry1:ry2, rx1:rx2] = sub_d4
            boxes_arr = np.asarray(boxes_list, dtype=np.float32)
        else:
            masks_arr = np.zeros((0, self.det_size, self.det_size), dtype=np.uint8)
            boxes_arr = np.zeros((0, 4), dtype=np.float32)

        ign_arr = np.asarray(ign_boxes_list, dtype=np.float32) if ign_boxes_list else np.zeros((0, 4), dtype=np.float32)
        if k_rot or hflip:
            ign_arr = _apply_d4_boxes(ign_arr, self.det_size, k_rot, hflip)

        if self.augment and rng.random() < 0.5:
            gamma = float(rng.uniform(0.92, 1.08))
            img_f = img_det.astype(np.float32)
            img_f[..., :2] = np.clip(img_f[..., :2] * gamma, 0.0, 255.0)
            img_det = img_f.astype(np.uint8)

        img_t = torch.from_numpy(img_det.transpose(2, 0, 1).copy()).float() * (1.0 / 255.0)
        target = {
            "boxes": torch.from_numpy(boxes_arr.copy()),
            "labels": torch.ones((len(boxes_arr),), dtype=torch.int64),
            "masks": torch.from_numpy(masks_arr),
            "ignore_boxes": torch.from_numpy(ign_arr.copy()),
        }
        return img_t, target


def _collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return [b[0] for b in batch], [b[1] for b in batch]


def _install_ignore_shield(roi_heads: nn.Module) -> None:
    def _shielded_select(self, proposals, targets):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        ign_boxes = [t.get("ignore_boxes") for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        for img_id in range(len(proposals)):
            ib = ign_boxes[img_id]
            if ib is not None and ib.numel() > 0:
                bg_mask = labels[img_id] == 0
                if bg_mask.any():
                    bg_idx = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
                    bg_props = proposals[img_id][bg_idx]
                    ious = box_ops.box_iou(bg_props, ib.to(dtype=dtype, device=device))
                    max_iou = ious.max(dim=1).values
                    cx = 0.5 * (bg_props[:, 0] + bg_props[:, 2])[:, None]
                    cy = 0.5 * (bg_props[:, 1] + bg_props[:, 3])[:, None]
                    inside = (
                        (cx >= ib[None, :, 0].to(device))
                        & (cx <= ib[None, :, 2].to(device))
                        & (cy >= ib[None, :, 1].to(device))
                        & (cy <= ib[None, :, 3].to(device))
                    ).any(dim=1)
                    hit = (max_iou >= 0.25) | inside
                    if hit.any():
                        labels[img_id][bg_idx[hit]] = -1

        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        for img_id in range(len(proposals)):
            inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][inds]
            labels[img_id] = labels[img_id][inds]
            matched_idxs[img_id] = matched_idxs[img_id][inds]
            gb = gt_boxes[img_id]
            if gb.numel() == 0:
                gb = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gb[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    roi_heads.select_training_samples = types.MethodType(_shielded_select, roi_heads)


def build_stage1_model(pretrained: bool = True, det_size: int = DET_SIZE) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=det_size,
        max_size=det_size,
        image_mean=[0.725, 0.045, 0.030],
        image_std=[0.180, 0.115, 0.095],
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=800,
        box_detections_per_img=140,
        box_score_thresh=0.05,
        box_nms_thresh=0.55,
    )
    ratios = (0.15, 0.33, 0.6, 1.0, 1.67, 3.0, 6.5)
    anchor_sizes = ((12,), (24,), (48,), (96,), (192,))
    model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=(ratios,) * 5)

    conv_depth = len(model.rpn.head.conv) if hasattr(model.rpn.head, "conv") else 2
    old_conv_state = copy.deepcopy(model.rpn.head.conv.state_dict()) if hasattr(model.rpn.head, "conv") else None
    model.rpn.head = RPNHead(model.backbone.out_channels, len(ratios), conv_depth=conv_depth)
    if old_conv_state is not None:
        model.rpn.head.conv.load_state_dict(old_conv_state)

    model.roi_heads.mask_roi_pool.output_size = (28, 28)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    _install_ignore_shield(model.roi_heads)
    return model


def save_state_dict_fp16(state_dict: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp16 = {
        k: (v.detach().cpu().half() if torch.is_tensor(v) and v.is_floating_point() else (v.detach().cpu() if torch.is_tensor(v) else v))
        for k, v in state_dict.items()
    }
    torch.save(fp16, path)


def load_stage1_detector_from_fp16(ckpt_path: Path, device: torch.device, det_size: int = DET_SIZE) -> nn.Module:
    model = build_stage1_model(pretrained=False, det_size=det_size).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd32)
    model.to(device).eval()
    return model


class _EMA:
    def __init__(self, model: nn.Module, max_decay: float = 0.997) -> None:
        self.max_decay = max_decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.steps = 0
        self.ema_params = [p for p in self.ema.parameters() if p.dtype.is_floating_point]
        self.src_params = [p for p in model.parameters() if p.dtype.is_floating_point]
        self.ema_bufs = list(self.ema.buffers())
        self.src_bufs = list(model.buffers())

    @torch.no_grad()
    def update(self) -> None:
        self.steps += 1
        d = min(self.max_decay, (1.0 + self.steps) / (10.0 + self.steps))
        torch._foreach_lerp_(self.ema_params, [p.detach() for p in self.src_params], 1.0 - d)
        for eb, sb in zip(self.ema_bufs, self.src_bufs):
            eb.copy_(sb.detach())


class _ThreadedBatchLoader:
    """Thread-pool prefetcher sharing read-only RAM physics in a single process (0 MB COW overhead)."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, threads: int = 4) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.threads = max(1, int(threads))

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.dataset) / float(self.batch_size)))

    def __iter__(self):
        n = len(self.dataset)
        order = np.random.permutation(n).tolist() if self.shuffle else list(range(n))
        batches = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.threads <= 1:
            for idxs in batches:
                yield _collate_fn([self.dataset[i] for i in idxs])
            return
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            q: deque = deque()
            max_prefetch = min(len(batches), 2)
            bi = 0
            while bi < max_prefetch:
                q.append([ex.submit(self.dataset.__getitem__, idx) for idx in batches[bi]])
                bi += 1
            while q:
                futs = q.popleft()
                if bi < len(batches):
                    q.append([ex.submit(self.dataset.__getitem__, idx) for idx in batches[bi]])
                    bi += 1
                yield _collate_fn([f.result() for f in futs])


def train_stage1_detector(
    train_stems: list[str],
    ram_physics: dict[str, np.ndarray] | None,
    cache_dir: Path,
    geom_map: dict[str, dict[str, float]],
    targets_map: dict[str, StemTargets],
    out_dir: Path,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 8,
    det_size: int = DET_SIZE,
    smoke: bool = False,
    fold_tag: str = "v10_f0_seed42",
    export_name: str = "stage1_mrcnn_f0_fp16.pt",
) -> tuple[nn.Module, list[dict[str, float]]]:
    ckpt_cache = cache_dir / f"stage1_{fold_tag}.pt"
    hist_cache = cache_dir / f"stage1_{fold_tag}_history.json"
    w_dir = out_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    export_path = w_dir / export_name

    if not smoke and os.environ.get("REUSE_STAGE1_CACHE", "1") == "1" and ckpt_cache.is_file():
        sd = torch.load(ckpt_cache, map_location="cpu", weights_only=False)
        save_state_dict_fp16(sd, export_path)
        model = load_stage1_detector_from_fp16(export_path, device, det_size=det_size)
        history = json.loads(hist_cache.read_text(encoding="utf-8")) if hist_cache.is_file() else []
        print(f"[stage1_mrcnn] loaded cached EMA weights -> {export_path.name} ({len(history)} epochs)", flush=True)
        return model, history

    if smoke:
        epochs = 1
        batch_size = min(2, max(1, len(train_stems)))

    ds = Stage1TrainDataset(
        stems=train_stems,
        ram_physics=ram_physics,
        cache_dir=cache_dir,
        geom_map=geom_map,
        targets_map=targets_map,
        det_size=det_size,
        augment=(not smoke),
    )
    dl = _ThreadedBatchLoader(ds, batch_size=batch_size, shuffle=True, threads=(1 if smoke else 4))

    model = build_stage1_model(pretrained=(not smoke), det_size=det_size).to(device)
    ema = _EMA(model)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    base_lr = 2.4e-4
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": base_lr * 0.5}, {"params": head_params, "lr": base_lr}],
        weight_decay=1e-2,
    )
    total_steps = max(1, epochs * len(dl))
    warmup_steps = min(max(1, len(dl)), max(1, total_steps // 6))

    def _lr_mult(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return max(0.1, float(step_idx + 1) / float(warmup_steps))
        prog = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.015, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult)
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    history: list[dict[str, float]] = []
    t_start = time.time()
    step = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_tot, ep_box_cls, ep_mask, ep_gn, n_bat = 0.0, 0.0, 0.0, 0.0, 0
        for imgs, tgts in dl:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                opt.step()

            sched.step()
            ema.update()
            step += 1
            n_bat += 1
            ep_tot += float(loss.detach())
            l_bc = float(
                (loss_dict["loss_classifier"].detach() if "loss_classifier" in loss_dict else 0.0)
                + (loss_dict["loss_box_reg"].detach() if "loss_box_reg" in loss_dict else 0.0)
            )
            l_mk = float(loss_dict["loss_mask"].detach()) if "loss_mask" in loss_dict else 0.0
            ep_box_cls += l_bc
            ep_mask += l_mk
            ep_gn += gn if math.isfinite(gn) else 0.0

        lr_cur = float(opt.param_groups[1]["lr"])
        row = {
            "stage": "stage1",
            "epoch": ep,
            "step": step,
            "steps": total_steps,
            "loss": ep_tot / max(1, n_bat),
            "bce": ep_box_cls / max(1, n_bat),
            "dice": ep_mask / max(1, n_bat),
            "lr": lr_cur,
            "gn": ep_gn / max(1, n_bat),
            "minutes": (time.time() - t_start) / 60.0,
        }
        history.append(row)
        print(
            f"[stage1_mrcnn] epoch {ep}/{epochs} step {step}/{total_steps} loss {row['loss']:.4f} "
            f"box_cls {row['bce']:.4f} mask {row['dice']:.4f} lr {lr_cur:.2e} t {row['minutes']:.2f}m",
            flush=True,
        )

    ema_sd = ema.ema.state_dict()
    save_state_dict_fp16(ema_sd, export_path)
    if not smoke:
        torch.save(ema_sd, ckpt_cache)
        hist_cache.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return ema.ema, history


def _view_transform(img_t: torch.Tensor, v: int) -> torch.Tensor:
    if v == 1:
        return torch.flip(img_t, dims=[2])
    if v == 2:
        return torch.flip(img_t, dims=[1])
    if v == 3:
        return torch.flip(img_t, dims=[1, 2])
    return img_t


def _invert_box_coords(b: np.ndarray, v: int, det_size: int) -> np.ndarray:
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if v in (1, 3):
        x1, x2 = float(det_size) - x2, float(det_size) - x1
    if v in (2, 3):
        y1, y2 = float(det_size) - y2, float(det_size) - y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _invert_crop_gpu(crop_hw: torch.Tensor, v: int) -> torch.Tensor:
    out = crop_hw
    if v in (1, 3):
        out = torch.flip(out, dims=[1])
    if v in (2, 3):
        out = torch.flip(out, dims=[0])
    return out


def cluster_multiview_proposals(
    candidates: list[dict[str, Any]],
    iou_thr: float = 0.45,
    n_views: int = 4,
    max_keep: int = 120,
) -> list[ProposalROI]:
    """Cluster view-tagged Mask R-CNN proposals across TTA views (and/or folds)."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: -c["score"])
    used = [False] * len(candidates)
    out: list[ProposalROI] = []

    for i, c0 in enumerate(candidates):
        if used[i]:
            continue
        used[i] = True
        cluster = [c0]
        bx1, by1, bx2, by2 = c0["bbox"]
        a1 = float((c0["mask"] > 0.35).sum())
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            cj = candidates[j]
            jx1, jy1, jx2, jy2 = cj["bbox"]
            ix1, iy1 = max(bx1, jx1), max(by1, jy1)
            ix2, iy2 = min(bx2, jx2), min(by2, jy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            m1_sub = c0["mask"][iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1] > 0.35
            m2_sub = cj["mask"][iy1 - jy1 : iy2 - jy1, ix1 - jx1 : ix2 - jx1] > 0.35
            inter = float(np.logical_and(m1_sub, m2_sub).sum())
            if inter <= 0.0:
                continue
            a2 = float((cj["mask"] > 0.35).sum())
            iou = inter / max(1.0, a1 + a2 - inter)
            if iou >= iou_thr:
                used[j] = True
                cluster.append(cj)

        ux1 = min(c["bbox"][0] for c in cluster)
        uy1 = min(c["bbox"][1] for c in cluster)
        ux2 = max(c["bbox"][2] for c in cluster)
        uy2 = max(c["bbox"][3] for c in cluster)
        acc = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.float32)
        w_sum = 0.0
        best_by_view: dict[int, float] = {}
        for c in cluster:
            cx1, cy1, cx2, cy2 = c["bbox"]
            sc = float(c["score"])
            acc[cy1 - uy1 : cy2 - uy1, cx1 - ux1 : cx2 - ux1] += sc * c["mask"].astype(np.float32)
            w_sum += sc
            v = int(c["view"])
            if sc > best_by_view.get(v, 0.0):
                best_by_view[v] = sc
        fused = (acc / max(1e-6, w_sum)).astype(np.float16)
        ys, xs = np.nonzero(fused > 0.28)
        if xs.size < 16:
            continue
        tx1, ty1 = int(xs.min()), int(ys.min())
        tx2, ty2 = int(xs.max()) + 1, int(ys.max()) + 1
        s_det = sum(best_by_view.values()) / float(max(1, n_views))
        v_agree = float(len(best_by_view)) / float(max(1, n_views))
        if s_det < 0.08:
            continue
        out.append(
            ProposalROI(
                bbox=(ux1 + tx1, uy1 + ty1, ux1 + tx2, uy1 + ty2),
                mask_f16=fused[ty1:ty2, tx1:tx2].copy(),
                s_det=float(s_det),
                view_agree=float(v_agree),
            )
        )
        if len(out) >= max_keep:
            break
    out.sort(key=lambda p: -p.s_det)
    return out


@torch.inference_mode()
def predict_stage1_raw_candidates(
    model: nn.Module,
    stems: list[str],
    cache_dir: Path,
    geom_map: dict[str, dict[str, float]],
    device: torch.device,
    ram_physics: dict[str, np.ndarray] | None = None,
    det_size: int = DET_SIZE,
    tta_views: tuple[int, ...] = (0, 1, 2, 3),
    score_thresh: float = 0.10,
    view_offset: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Return unclustered view-tagged raw Mask R-CNN candidates (native 2048 coords)."""
    model.eval()
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    results: dict[str, list[dict[str, Any]]] = {}

    for stem in stems:
        hw3 = ram_physics[stem] if (ram_physics and stem in ram_physics) else load_cached_physics_u8(cache_dir, stem)
        geom = geom_map[stem]
        x0, y0, side = compute_active_disk_box(geom)
        scale = float(det_size) / float(side)
        img_det = _crop_and_resize_disk(hw3, x0, y0, side, det_size)
        base_t = torch.from_numpy(img_det.transpose(2, 0, 1).copy()).float() * (1.0 / 255.0)
        base_t = base_t.to(device)

        views_tensors = [_view_transform(base_t, v) for v in tta_views]
        cands: list[dict[str, Any]] = []
        for start in range(0, len(views_tensors), 2):
            sub_v = tta_views[start : start + 2]
            sub_t = views_tensors[start : start + 2]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                preds = model(sub_t)
            for v, pred in zip(sub_v, preds):
                scores_t = pred["scores"].float()
                keep_idx = torch.nonzero(scores_t >= score_thresh, as_tuple=False).squeeze(1)
                if keep_idx.numel() == 0:
                    continue
                boxes_np = pred["boxes"][keep_idx].float().cpu().numpy()
                scores_np = scores_t[keep_idx].cpu().numpy()
                masks_t = pred["masks"][keep_idx, 0]

                for k_i in range(len(scores_np)):
                    bx1, by1, bx2, by2 = boxes_np[k_i]
                    vx1 = max(0, int(np.floor(bx1)) - 2)
                    vy1 = max(0, int(np.floor(by1)) - 2)
                    vx2 = min(det_size, int(np.ceil(bx2)) + 2)
                    vy2 = min(det_size, int(np.ceil(by2)) + 2)
                    if vx2 <= vx1 + 1 or vy2 <= vy1 + 1:
                        continue
                    sub_gpu = masks_t[k_i, vy1:vy2, vx1:vx2].float()
                    if float(sub_gpu.max()) < 0.25:
                        continue
                    sub_gpu = _invert_crop_gpu(sub_gpu, v)

                    ux1, uy1, ux2, uy2 = _invert_box_coords(np.array([vx1, vy1, vx2, vy2]), v, det_size)
                    nx1 = int(np.clip(np.floor(x0 + ux1 / scale), 0, IMG_SIZE - 1))
                    ny1 = int(np.clip(np.floor(y0 + uy1 / scale), 0, IMG_SIZE - 1))
                    nx2 = int(np.clip(np.ceil(x0 + ux2 / scale), nx1 + 2, IMG_SIZE))
                    ny2 = int(np.clip(np.ceil(y0 + uy2 / scale), ny1 + 2, IMG_SIZE))
                    pcx, pcy = 0.5 * (nx1 + nx2), 0.5 * (ny1 + ny2)
                    if math.hypot(pcx - geom["cx"], pcy - geom["cy"]) > 0.996 * geom["R_sun"]:
                        continue

                    nat_t = F.interpolate(
                        sub_gpu.unsqueeze(0).unsqueeze(0),
                        size=(ny2 - ny1, nx2 - nx1),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    cands.append({
                        "view": view_offset + int(v),
                        "score": float(scores_np[k_i]),
                        "bbox": (nx1, ny1, nx2, ny2),
                        "mask": nat_t.to(torch.float16).cpu().numpy(),
                    })
        results[stem] = cands
    return results


@torch.inference_mode()
def predict_stage1_proposals(
    model: nn.Module,
    stems: list[str],
    cache_dir: Path,
    geom_map: dict[str, dict[str, float]],
    device: torch.device,
    ram_physics: dict[str, np.ndarray] | None = None,
    det_size: int = DET_SIZE,
    tta_views: tuple[int, ...] = (0, 1, 2, 3),
    score_thresh: float = 0.10,
) -> dict[str, list[ProposalROI]]:
    raw = predict_stage1_raw_candidates(
        model, stems, cache_dir, geom_map, device, ram_physics=ram_physics,
        det_size=det_size, tta_views=tta_views, score_thresh=score_thresh, view_offset=0,
    )
    return {s: cluster_multiview_proposals(cands, iou_thr=0.45, n_views=len(tta_views)) for s, cands in raw.items()}
