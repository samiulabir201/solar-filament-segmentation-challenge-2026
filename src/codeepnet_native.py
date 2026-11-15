"""Full-resolution 1:1 Heterogeneous Cooperative Co-DeepNet v10 (Thin-Filament Focal Boost + Threaded Prefetch + FP16 Export)."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from physics_v2 import (
    IMG_SIZE,
    SLICE_SIZE,
    active_disk_slice_origin,
    build_native_5ch_gpu,
    transform_local_center_d4,
)
from stage1_detector import save_state_dict_fp16
from targets_v10 import StemTargets


def _gn(ch: int) -> nn.GroupNorm:
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class SCSEBlock(nn.Module):
    """Concurrent spatial + channel squeeze-and-excitation."""

    def __init__(self, ch: int, r: int = 8) -> None:
        super().__init__()
        mid = max(4, ch // r)
        self.c_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.c_se(x) + x * self.s_se(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, dil: int = 1) -> None:
        super().__init__()
        pad = ((k - 1) // 2) * dil
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, dilation=dil, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, 3)
        self.c2 = ConvGNAct(out_ch, out_ch, 3)
        self.scse = SCSEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scse(self.c2(self.c1(x)))


class RibbonBarbBlock(nn.Module):
    """Multi-orientation anisotropic conv block (3x3 + 1x9 + 9x1 + 3x3 dilated) for fine barbs & thin ribbons."""

    def __init__(self, in_ch: int, out_ch: int, dil: int = 2) -> None:
        super().__init__()
        b = max(4, out_ch // 4)
        cat_ch = b * 4
        self.proj = nn.Conv2d(in_ch, b, kernel_size=1, bias=False)
        self.br_iso = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.br_h = nn.Conv2d(b, b, kernel_size=(1, 9), padding=(0, 4), bias=False)
        self.br_v = nn.Conv2d(b, b, kernel_size=(9, 1), padding=(4, 0), bias=False)
        self.br_dil = nn.Conv2d(b, b, kernel_size=3, padding=dil, dilation=dil, bias=False)
        self.norm_act = nn.Sequential(_gn(cat_ch), nn.SiLU(inplace=True))
        self.out_conv = ConvGNAct(cat_ch, out_ch, 3)
        self.scse = SCSEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        h = torch.cat([self.br_iso(z), self.br_h(z), self.br_v(z), self.br_dil(z)], dim=1)
        h = self.scse(self.out_conv(self.norm_act(h)))
        return h + self.skip(x)


class CrossBridge(nn.Module):
    """Bidirectional zero-init gated feature exchange between Peer A (ConvNeXt) and Peer B (RibbonBarbUNet)."""

    def __init__(self, ch_a: int, ch_b: int) -> None:
        super().__init__()
        self.b2a = nn.Sequential(ConvGNAct(ch_b, ch_a, 1), nn.Conv2d(ch_a, ch_a, 1))
        self.a2b = nn.Sequential(ConvGNAct(ch_a, ch_b, 1), nn.Conv2d(ch_b, ch_b, 1))
        nn.init.zeros_(self.b2a[-1].weight)
        nn.init.zeros_(self.b2a[-1].bias)
        nn.init.zeros_(self.a2b[-1].weight)
        nn.init.zeros_(self.a2b[-1].bias)
        self.alpha_b2a = nn.Parameter(torch.tensor(0.0))
        self.alpha_a2b = nn.Parameter(torch.tensor(0.0))

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ga = torch.tanh(self.alpha_b2a)
        gb = torch.tanh(self.alpha_a2b)
        return fa + ga * self.b2a(fb), fb + gb * self.a2b(fa)


class NativeCoDeepNet(nn.Module):
    """Full-resolution 1:1 cooperative dual-network taking strictly 5 physical H-alpha channels (zero shortcut leakage)."""

    def __init__(self, in_ch: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_a = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k_384",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        assert hasattr(self.backbone_a, "stem_0"), "Expected timm ConvNeXt features_only stem_0"
        old_conv = self.backbone_a.stem_0
        assert isinstance(old_conv, nn.Conv2d) and old_conv.in_channels == 3
        new_conv = nn.Conv2d(
            in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            nn.init.normal_(new_conv.weight[:, 3:], mean=0.0, std=0.005)
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone_a.stem_0 = new_conv

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406, 0.0, 0.0]).view(1, 5, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225, 1.0, 1.0]).view(1, 5, 1, 1))

        self.s1_stem_a = nn.Sequential(ConvGNAct(in_ch, 24, 3), ConvGNAct(24, 24, 3))
        self.s2_stem_a = nn.Sequential(ConvGNAct(24, 48, 3, stride=2), ConvGNAct(48, 48, 3))

        self.up16_a = DecBlock(768 + 384, 256)
        self.up8_a = DecBlock(256 + 192, 160)
        self.up4_a = DecBlock(160 + 96, 96)
        self.up2_a = DecBlock(96 + 48, 64)
        self.up1_a = DecBlock(64 + 24, 32)

        self.enc1_b = RibbonBarbBlock(in_ch, 24, dil=2)
        self.enc2_b = nn.Sequential(nn.AvgPool2d(2), RibbonBarbBlock(24, 48, dil=2))
        self.enc4_b = nn.Sequential(nn.AvgPool2d(2), RibbonBarbBlock(48, 96, dil=2))
        self.enc8_b = nn.Sequential(nn.AvgPool2d(2), RibbonBarbBlock(96, 144, dil=2))
        self.enc16_b = nn.Sequential(nn.AvgPool2d(2), RibbonBarbBlock(144, 192, dil=3))

        self.dec8_b = RibbonBarbBlock(192 + 144, 144, dil=2)
        self.dec4_b = RibbonBarbBlock(144 + 96, 96, dil=2)
        self.dec2_b = RibbonBarbBlock(96 + 48, 48, dil=2)
        self.dec1_b = RibbonBarbBlock(48 + 24, 24, dil=2)

        self.bridge16 = CrossBridge(256, 192)
        self.bridge8 = CrossBridge(160, 144)
        self.bridge4 = CrossBridge(96, 96)
        self.bridge2 = CrossBridge(64, 48)

        self.head_a = nn.Conv2d(32, 3, 1)
        self.head_b = nn.Conv2d(24, 3, 1)
        self.fuse_dec = nn.Sequential(
            ConvGNAct(32 + 24 + in_ch, 32, 3),
            ConvGNAct(32, 32, 3),
        )
        self.head_f = nn.Conv2d(32, 3, 1)

        bias_vec = torch.tensor([-4.6, -5.0, -5.0])
        for hd in (self.head_a, self.head_b, self.head_f):
            nn.init.normal_(hd.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                hd.bias.copy_(bias_vec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xn = (x - self.im_mean) / self.im_std

        c0, c1, c2, c3 = self.backbone_a(xn)
        a_s1 = self.s1_stem_a(xn)
        a_s2 = self.s2_stem_a(a_s1)

        b_s1 = self.enc1_b(xn)
        b_s2 = self.enc2_b(b_s1)
        b_s4 = self.enc4_b(b_s2)
        b_s8 = self.enc8_b(b_s4)
        b_s16 = self.enc16_b(b_s8)

        d16_a = self.up16_a(torch.cat([F.interpolate(c3, size=c2.shape[-2:], mode="bilinear", align_corners=False), c2], dim=1))
        d16_a, d16_b = self.bridge16(d16_a, b_s16)

        d8_a = self.up8_a(torch.cat([F.interpolate(d16_a, size=c1.shape[-2:], mode="bilinear", align_corners=False), c1], dim=1))
        d8_b = self.dec8_b(torch.cat([F.interpolate(d16_b, size=b_s8.shape[-2:], mode="bilinear", align_corners=False), b_s8], dim=1))
        d8_a, d8_b = self.bridge8(d8_a, d8_b)

        d4_a = self.up4_a(torch.cat([F.interpolate(d8_a, size=c0.shape[-2:], mode="bilinear", align_corners=False), c0], dim=1))
        d4_b = self.dec4_b(torch.cat([F.interpolate(d8_b, size=b_s4.shape[-2:], mode="bilinear", align_corners=False), b_s4], dim=1))
        d4_a, d4_b = self.bridge4(d4_a, d4_b)

        d2_a = self.up2_a(torch.cat([F.interpolate(d4_a, size=a_s2.shape[-2:], mode="bilinear", align_corners=False), a_s2], dim=1))
        d2_b = self.dec2_b(torch.cat([F.interpolate(d4_b, size=b_s2.shape[-2:], mode="bilinear", align_corners=False), b_s2], dim=1))
        d2_a, d2_b = self.bridge2(d2_a, d2_b)

        fa = self.up1_a(torch.cat([F.interpolate(d2_a, size=a_s1.shape[-2:], mode="bilinear", align_corners=False), a_s1], dim=1))
        fb = self.dec1_b(torch.cat([F.interpolate(d2_b, size=b_s1.shape[-2:], mode="bilinear", align_corners=False), b_s1], dim=1))

        la = self.head_a(fa)
        lb = self.head_b(fb)
        ff = self.fuse_dec(torch.cat([fa, fb, x], dim=1))
        lf = self.head_f(ff)
        return lf, la, lb


class NativeCropDataset(Dataset):
    """RAM-backed 768x768 native 1:1 crop dataset with thread-local RNG for lock-free threaded prefetch."""

    def __init__(
        self,
        train_stems: list[str],
        ram_physics: dict[str, np.ndarray],
        bitmasks: dict[str, np.ndarray],
        geom_map: dict[str, dict[str, float]],
        targets_map: dict[str, StemTargets],
        crop_size: int = 768,
        crops_per_stem: int = 3,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.stems = list(train_stems)
        self.ram_physics = ram_physics
        self.bitmasks = bitmasks
        self.geom_map = geom_map
        self.targets_map = targets_map
        self.crop_size = crop_size
        self.crops_per_stem = crops_per_stem
        self.augment = augment
        self.seed = int(seed)
        self._tls = threading.local()

        self.stem_inst_pts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in self.stems:
            insts = targets_map[s].instances
            if insts:
                pts = np.asarray(
                    [[0.5 * (it.bbox[0] + it.bbox[2]), 0.5 * (it.bbox[1] + it.bbox[3])] for it in insts],
                    dtype=np.float32,
                )
                ws = np.asarray([it.sampling_weight for it in insts], dtype=np.float64)
                ws /= ws.sum()
                self.stem_inst_pts[s] = (pts, ws)

    def _rng(self) -> np.random.Generator:
        rng = getattr(self._tls, "rng", None)
        if rng is None:
            rng = np.random.default_rng((self.seed + threading.get_ident()) & 0xFFFFFFFF)
            self._tls.rng = rng
        return rng

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_stem

    def __getitem__(self, idx: int):
        rng = self._rng()
        stem = self.stems[idx % len(self.stems)]
        hw3 = self.ram_physics[stem]
        bm = self.bitmasks[stem]
        g = self.geom_map[stem]
        st = self.targets_map[stem]
        cs = self.crop_size
        cx_s, cy_s, r_sun = float(g["cx"]), float(g["cy"]), float(g["R_sun"])

        sx0, sy0 = active_disk_slice_origin(cx_s, cy_s, SLICE_SIZE, IMG_SIZE)
        min_x, max_x = sx0, min(IMG_SIZE - cs, sx0 + SLICE_SIZE - cs)
        min_y, max_y = sy0, min(IMG_SIZE - cs, sy0 + SLICE_SIZE - cs)
        if max_x < min_x:
            min_x, max_x = 0, IMG_SIZE - cs
        if max_y < min_y:
            min_y, max_y = 0, IMG_SIZE - cs

        if stem in self.stem_inst_pts and rng.random() < 0.70:
            pts, ws = self.stem_inst_pts[stem]
            pick = int(rng.choice(len(pts), p=ws))
            px, py = pts[pick]
            jx = float(rng.integers(-180, 181))
            jy = float(rng.integers(-180, 181))
            x0 = int(np.clip(round(px + jx - 0.5 * cs), min_x, max_x))
            y0 = int(np.clip(round(py + jy - 0.5 * cs), min_y, max_y))
        else:
            best_xy = (int(rng.integers(min_x, max_x + 1)), int(rng.integers(min_y, max_y + 1)))
            best_score = -1.0
            for _ in range(4):
                tx = int(rng.integers(min_x, max_x + 1))
                ty = int(rng.integers(min_y, max_y + 1))
                ccx, ccy = tx + 0.5 * cs, ty + 0.5 * cs
                rho = math.hypot(ccx - cx_s, ccy - cy_s) / max(100.0, r_sun)
                if rho > 0.92:
                    continue
                sub_d = hw3[ty : ty + cs : 16, tx : tx + cs : 16, 1]
                sc = float((sub_d > 50).mean()) + (0.25 if 0.55 <= rho <= 0.90 else 0.0)
                if sc > best_score:
                    best_score = sc
                    best_xy = (tx, ty)
            x0, y0 = best_xy

        hw3_crop = hw3[y0 : y0 + cs, x0 : x0 + cs]
        bm_crop = bm[y0 : y0 + cs, x0 : x0 + cs]
        cx_loc = cx_s - float(x0)
        cy_loc = cy_s - float(y0)

        if self.augment:
            k_rot = int(rng.integers(0, 4))
            hflip = bool(rng.integers(0, 2))
            if k_rot:
                hw3_crop = np.rot90(hw3_crop, k_rot, axes=(0, 1))
                bm_crop = np.rot90(bm_crop, k_rot, axes=(0, 1))
            if hflip:
                hw3_crop = hw3_crop[:, ::-1]
                bm_crop = bm_crop[:, ::-1]
            cx_loc, cy_loc = transform_local_center_d4(cx_loc, cy_loc, cs, cs, k_rot, hflip)

        hw3_c = np.ascontiguousarray(hw3_crop.transpose(2, 0, 1))
        bm_c = np.ascontiguousarray(bm_crop)
        meta = np.array([cx_loc, cy_loc, r_sun, 1.0 if st.is_clean_stem else 0.0], dtype=np.float32)
        return torch.from_numpy(hw3_c), torch.from_numpy(bm_c), torch.from_numpy(meta)


class _ThreadedCropLoader:
    """Thread-pool batch prefetcher sharing read-only RAM physics & bitmasks (0 MB fork COW overhead)."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        threads: int = 4,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.threads = max(1, int(threads))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last and n >= self.batch_size:
            return max(1, n // self.batch_size)
        return max(1, math.ceil(n / float(self.batch_size)))

    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n).tolist() if self.shuffle else list(range(n))
        batches = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.drop_last and len(batches) > 1 and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.threads <= 1:
            for idxs in batches:
                yield default_collate([self.dataset[i] for i in idxs])
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
                yield default_collate([f.result() for f in futs])


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    if labels_flat.numel() == 0 or not (labels_flat > 0.5).any():
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat.float() * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = (labels_flat[perm] > 0.5).long()
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def _masked_soft_iou_or_dice(probs: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, is_dice: bool = False) -> torch.Tensor:
    p = probs * valid
    t = targets * valid
    inter = (p * t).sum()
    if is_dice:
        return 1.0 - (2.0 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
    return 1.0 - (inter + 0.5) / (p.sum() + t.sum() - inter + 0.5)


def _focal_fg_pixelwise(logits_fg: torch.Tensor, y_fg: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits_fg, y_fg, reduction="none")
    prob = torch.sigmoid(logits_fg)
    p_t = y_fg * prob + (1.0 - y_fg) * (1.0 - prob)
    mod = (1.0 - p_t).clamp(min=0.0, max=1.0) ** gamma
    return bce * mod


def _compute_head_loss(
    logits: torch.Tensor,
    y_fg: torch.Tensor,
    y_core: torch.Tensor,
    y_spine: torch.Tensor,
    valid: torch.Tensor,
    thin_weight_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute float32 PQ-aligned loss for one 3-channel head [fg, core, spine] with bit-4 thin boost."""
    l32 = logits.float()
    l_fg, l_core, l_spine = l32[:, 0], l32[:, 1], l32[:, 2]
    v = valid.float()
    v_sum = v.sum().clamp(min=1.0)

    foc_map = _focal_fg_pixelwise(l_fg, y_fg, gamma=2.0)
    if thin_weight_map is not None:
        foc_map = foc_map * thin_weight_map
    loss_foc = (foc_map * v).sum() / v_sum

    p_fg = torch.sigmoid(l_fg)
    loss_iou = _masked_soft_iou_or_dice(p_fg, y_fg, v, is_dice=False)

    lov_acc = l_fg.new_tensor(0.0)
    b_cnt = 0
    for bi in range(l_fg.shape[0]):
        vb = v[bi] > 0.5
        if vb.any() and (y_fg[bi][vb] > 0.5).any():
            act = vb & ((y_fg[bi] > 0.5) | (l_fg[bi].detach() > -1.0))
            lov_acc = lov_acc + _lovasz_hinge_flat(l_fg[bi][act], y_fg[bi][act])
            b_cnt += 1
    loss_lov = lov_acc / max(1, b_cnt)

    pw = l32.new_tensor(4.0)
    bce_core = (F.binary_cross_entropy_with_logits(l_core, y_core, pos_weight=pw, reduction="none") * v).sum() / v_sum
    dice_core = _masked_soft_iou_or_dice(torch.sigmoid(l_core), y_core, v, is_dice=True)
    loss_core = 0.5 * bce_core + 0.5 * dice_core

    bce_sp = (F.binary_cross_entropy_with_logits(l_spine, y_spine, pos_weight=pw, reduction="none") * v).sum() / v_sum
    dice_sp = _masked_soft_iou_or_dice(torch.sigmoid(l_spine), y_spine, v, is_dice=True)
    loss_spine = 0.5 * bce_sp + 0.5 * dice_sp

    total = 6.0 * loss_foc + 1.0 * loss_iou + 0.35 * loss_lov + 0.25 * loss_core + 0.25 * loss_spine
    return total, loss_foc.detach(), loss_iou.detach()


class _ModelEMA:
    def __init__(self, model: nn.Module, max_decay: float = 0.998) -> None:
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


def load_native_codeepnet_from_fp16(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Instantiate NativeCoDeepNet strictly offline (pretrained=False) and restore fp16 weights in float32."""
    model = NativeCoDeepNet(in_ch=5, pretrained=False).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd32)
    model.to(device).eval()
    return model


def train_native_codeepnet(
    train_stems: list[str],
    ram_physics: dict[str, np.ndarray],
    bitmasks: dict[str, np.ndarray],
    geom_map: dict[str, dict[str, float]],
    targets_map: dict[str, StemTargets],
    out_dir: Path,
    cache_dir: Path,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 8,
    grad_accum: int = 2,
    crop_size: int = 768,
    crops_per_stem: int = 3,
    smoke: bool = False,
    seed: int = 42,
    fold_tag: str = "v10_f0_seed42",
    export_name: str = "codeepnet_native_f0_fp16.pt",
) -> tuple[nn.Module, list[dict[str, float]]]:
    ckpt_cache = cache_dir / f"codeepnet_native_{fold_tag}.pt"
    hist_cache = cache_dir / f"codeepnet_native_{fold_tag}_history.json"
    w_dir = out_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    export_path = w_dir / export_name

    if not smoke and os.environ.get("REUSE_NATIVE_CACHE", "1") == "1" and ckpt_cache.is_file():
        sd = torch.load(ckpt_cache, map_location="cpu", weights_only=False)
        save_state_dict_fp16(sd, export_path)
        model = load_native_codeepnet_from_fp16(export_path, device)
        history = json.loads(hist_cache.read_text(encoding="utf-8")) if hist_cache.is_file() else []
        print(f"[codeepnet_native] loaded cached EMA weights -> {export_path.name} ({len(history)} epochs)", flush=True)
        return model, history

    if smoke:
        epochs = 2
        batch_size = min(2, max(1, len(train_stems)))
        grad_accum = 1
        crop_size = 512
        crops_per_stem = 1

    ds = NativeCropDataset(
        train_stems=train_stems,
        ram_physics=ram_physics,
        bitmasks=bitmasks,
        geom_map=geom_map,
        targets_map=targets_map,
        crop_size=crop_size,
        crops_per_stem=crops_per_stem,
        seed=seed,
        augment=(not smoke),
    )
    dl = _ThreadedCropLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(not smoke and len(ds) >= batch_size),
        threads=(1 if smoke else 4),
        seed=seed,
    )

    model = NativeCoDeepNet(in_ch=5, pretrained=(not smoke)).to(device)
    ema = _ModelEMA(model, max_decay=0.998)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone_a.") and not name.startswith("backbone_a.stem_0."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1.8e-4}, {"params": head_params, "lr": 5.0e-4}],
        weight_decay=1.0e-2,
    )
    steps_per_ep = max(1, math.ceil(len(dl) / max(1, grad_accum)))
    total_opt_steps = max(1, epochs * steps_per_ep)
    warmup_steps = max(1, min(int(1.5 * steps_per_ep), total_opt_steps // 4))

    def _lr_mult(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return max(0.05, float(step_idx + 1) / float(warmup_steps))
        prog = float(step_idx - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return max(1.0e-6 / 5.0e-4, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    history: list[dict[str, float]] = []
    t_start = time.time()
    opt_step = 0
    co_start_ep = max(4, epochs // 4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_loss, ep_foc, ep_iou, ep_cons, ep_gn, n_bat, ep_opt = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        micro = 0

        for b_idx, (hw3_u8, bmask_u8, meta) in enumerate(dl):
            hw3_u8 = hw3_u8.to(device, non_blocking=True)
            bmask_u8 = bmask_u8.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            x_5ch, on_disk = build_native_5ch_gpu(hw3_u8, meta[:, :3])
            y_fg = ((bmask_u8 & 1) > 0).float()
            y_core = ((bmask_u8 & 2) > 0).float()
            y_spine = ((bmask_u8 & 4) > 0).float()
            w_thin = torch.where((bmask_u8 & 16) > 0, 2.0, 1.0).float()
            v_base = (((bmask_u8 & 8) > 0) & (on_disk > 0.5)).float()
            is_clean = meta[:, 3] > 0.5

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                lf, la, lb = model(x_5ch)

            v_a = v_base.clone()
            v_b = v_base.clone()
            if ep >= co_start_ep and (~is_clean).any():
                with torch.no_grad():
                    pa0 = torch.sigmoid(la[:, 0].float())
                    pb0 = torch.sigmoid(lb[:, 0].float())
                    fa_map = _focal_fg_pixelwise(la[:, 0].float(), y_fg, gamma=2.0)
                    fb_map = _focal_fg_pixelwise(lb[:, 0].float(), y_fg, gamma=2.0)
                    for bi in torch.nonzero(~is_clean, as_tuple=False).squeeze(1).tolist():
                        bg_valid = (y_fg[bi] < 0.5) & (v_base[bi] > 0.5)
                        if int(bg_valid.sum()) < 256:
                            continue
                        thr_a = torch.quantile(fa_map[bi][bg_valid], 0.994)
                        thr_b = torch.quantile(fb_map[bi][bg_valid], 0.994)
                        forgive_a = bg_valid & (pb0[bi] > 0.60) & (fa_map[bi] >= thr_a)
                        forgive_b = bg_valid & (pa0[bi] > 0.60) & (fb_map[bi] >= thr_b)
                        v_a[bi][forgive_a] = 0.0
                        v_b[bi][forgive_b] = 0.0
            v_f = torch.minimum(v_a, v_b)

            loss_f, foc_f, iou_f = _compute_head_loss(lf, y_fg, y_core, y_spine, v_f, thin_weight_map=w_thin)
            loss_a, _, _ = _compute_head_loss(la, y_fg, y_core, y_spine, v_a, thin_weight_map=w_thin)
            loss_b, _, _ = _compute_head_loss(lb, y_fg, y_core, y_spine, v_b, thin_weight_map=w_thin)

            if ep >= 3:
                pa = torch.sigmoid(la.float())
                pb = torch.sigmoid(lb.float())
                vm3 = v_f.unsqueeze(1).expand_as(pa)
                vm_sum = vm3.sum().clamp(min=1.0)
                c_a2b = (((pa - pb.detach()) ** 2) * vm3).sum() / vm_sum
                c_b2a = (((pb - pa.detach()) ** 2) * vm3).sum() / vm_sum
                loss_cons = 0.15 * (c_a2b + c_b2a)
            else:
                loss_cons = loss_f.new_tensor(0.0)

            total_loss = 0.45 * loss_f + 0.275 * loss_a + 0.275 * loss_b + loss_cons
            scaled = total_loss / float(max(1, grad_accum))

            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            micro += 1
            is_last = (b_idx + 1) == len(dl)
            if micro >= grad_accum or is_last:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    scaler.step(opt)
                    scaler.update()
                else:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update()
                opt_step += 1
                ep_opt += 1
                micro = 0
                ep_gn += gn if math.isfinite(gn) else 0.0

            ep_loss += float(total_loss.detach())
            ep_foc += float(foc_f)
            ep_iou += float(iou_f)
            ep_cons += float(loss_cons.detach())
            n_bat += 1

        lr_cur = float(opt.param_groups[1]["lr"])
        alpha_ab = float(torch.tanh(ema.ema.bridge16.alpha_a2b).detach())
        row = {
            "stage": "codeepnet_native",
            "epoch": ep,
            "step": opt_step,
            "steps": total_opt_steps,
            "loss": ep_loss / max(1, n_bat),
            "bce": ep_foc / max(1, n_bat),
            "dice": ep_iou / max(1, n_bat),
            "cons": ep_cons / max(1, n_bat),
            "lr": lr_cur,
            "gn": ep_gn / max(1, ep_opt),
            "alpha16": alpha_ab,
            "minutes": (time.time() - t_start) / 60.0,
        }
        history.append(row)
        print(
            f"[codeepnet_native] epoch {ep}/{epochs} step {opt_step}/{total_opt_steps} "
            f"loss {row['loss']:.4f} foc {row['bce']:.4f} soft_iou {row['dice']:.4f} "
            f"cons {row['cons']:.4f} lr {lr_cur:.2e} t {row['minutes']:.2f}m",
            flush=True,
        )

    ema_sd = ema.ema.state_dict()
    save_state_dict_fp16(ema_sd, export_path)
    if not smoke:
        torch.save(ema_sd, ckpt_cache)
        hist_cache.write_text(json.dumps(history, indent=2), encoding="utf-8")
    ema_model = ema.ema
    del model, opt, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ema_model, history


@torch.inference_mode()
def predict_native_full_disk(
    model: nn.Module,
    hw3_2048_u8: np.ndarray,
    geom: dict[str, float],
    device: torch.device,
    slice_size: int = SLICE_SIZE,
    tta_views: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 1:1 pure integer slice 1888x1888 4x flip TTA inference with fp16-safe logit clamping."""
    model.eval()
    cx, cy, r_sun = float(geom["cx"]), float(geom["cy"]), float(geom["R_sun"])
    x0, y0 = active_disk_slice_origin(cx, cy, slice_sz=slice_size, img_size=IMG_SIZE)
    hw3_slice = np.ascontiguousarray(hw3_2048_u8[y0 : y0 + slice_size, x0 : x0 + slice_size].transpose(2, 0, 1))
    hw3_t = torch.from_numpy(hw3_slice).unsqueeze(0).to(device, non_blocking=True)
    meta_t = torch.tensor([[cx - float(x0), cy - float(y0), r_sun]], dtype=torch.float32, device=device)
    x_5ch, on_disk = build_native_5ch_gpu(hw3_t, meta_t)

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    acc_fg = torch.zeros((slice_size, slice_size), dtype=torch.float32, device=device)
    acc_core = torch.zeros_like(acc_fg)
    acc_spine = torch.zeros_like(acc_fg)
    acc_dis = torch.zeros_like(acc_fg)

    for v in tta_views:
        xv = x_5ch
        if v == 1:
            xv = torch.flip(x_5ch, dims=[3])
        elif v == 2:
            xv = torch.flip(x_5ch, dims=[2])
        elif v == 3:
            xv = torch.flip(x_5ch, dims=[2, 3])

        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                lf, la, lb = model(xv)
        except RuntimeError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lf, la, lb = model(xv)
            except RuntimeError:
                lf, la, lb = model(xv.float())

        lf32 = torch.nan_to_num(lf[0].float(), nan=0.0, posinf=20.0, neginf=-20.0)
        la32 = torch.nan_to_num(la[0].float(), nan=0.0, posinf=20.0, neginf=-20.0)
        lb32 = torch.nan_to_num(lb[0].float(), nan=0.0, posinf=20.0, neginf=-20.0)

        pf = torch.sigmoid(lf32)
        pa = torch.sigmoid(la32)
        pb = torch.sigmoid(lb32)

        p3 = 0.50 * pf + 0.25 * pa + 0.25 * pb
        dis = torch.abs(pa[0] - pb[0])

        if v == 1:
            p3 = torch.flip(p3, dims=[2])
            dis = torch.flip(dis, dims=[1])
        elif v == 2:
            p3 = torch.flip(p3, dims=[1])
            dis = torch.flip(dis, dims=[0])
        elif v == 3:
            p3 = torch.flip(p3, dims=[1, 2])
            dis = torch.flip(dis, dims=[0, 1])

        acc_fg += p3[0]
        acc_core += p3[1]
        acc_spine += p3[2]
        acc_dis += dis

    inv_n = 1.0 / float(len(tta_views))
    od = on_disk[0]
    p_fg = (acc_fg * inv_n * od).cpu().numpy()
    p_core = (acc_core * inv_n * od).cpu().numpy()
    p_spine = (acc_spine * inv_n * od).cpu().numpy()
    p_dis = (acc_dis * inv_n * od).cpu().numpy()
    return x0, y0, p_fg, p_core, p_spine, p_dis
