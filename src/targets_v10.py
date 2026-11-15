"""Zero-bloat SDF consensus (psi >= 0.0), strict singleton filtering, bit-4 thin mask & fast threaded bitmask builder."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mu_coco
from scipy import ndimage

from physics_v2 import IMG_SIZE, compute_solar_geometry


@dataclass
class InstanceTarget:
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2) inclusive-exclusive in [0, 2048]
    rle: dict[str, Any]
    mask_crop: np.ndarray                # uint8 (y2-y1, x2-x1)
    spine_crop: np.ndarray               # uint8 (y2-y1, x2-x1)
    n_annotators: int
    weight: float
    is_consensus: bool
    area: int
    sampling_weight: float               # higher for thin/near-limb filaments
    is_thin: bool = False                # thin ribbon/filament flag for focal bit-4 weighting


@dataclass
class StemTargets:
    stem: str
    n_annotators: int
    instances: list[InstanceTarget]
    ignore_rle: dict[str, Any] | None
    ignore_crop: tuple[int, int, int, int, np.ndarray] | None
    is_clean_stem: bool


def _ann_to_rle(ann: dict[str, Any], h: int = IMG_SIZE, w: int = IMG_SIZE) -> dict[str, Any]:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mu_coco.merge(mu_coco.frPyObjects(seg, h, w))
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mu_coco.frPyObjects(seg, h, w)
    else:
        rle = dict(seg)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def _parse_spine_pts(ann: dict[str, Any]) -> np.ndarray | None:
    sp = ann.get("spine")
    if not sp or not isinstance(sp, list):
        return None
    if isinstance(sp[0], (int, float)):
        n = (len(sp) // 2) * 2
        if n < 4:
            return None
        pts = np.asarray(sp[:n], dtype=np.float32).reshape(-1, 2)
    else:
        pts = np.asarray(sp, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return None
    return np.round(pts).astype(np.int32)


def _mask_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _box_iou_fast(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])


def render_spine_crop(mask_crop: np.ndarray, spine_pts: list[np.ndarray], ox: int, oy: int) -> np.ndarray:
    """Render clipped 3-px spine inside a cropped mask window (H, W) with origin (ox, oy)."""
    ch, cw = mask_crop.shape
    out = np.zeros((ch, cw), dtype=np.uint8)
    for pts in spine_pts:
        if len(pts) >= 2:
            shifted = pts.copy()
            shifted[:, 0] -= ox
            shifted[:, 1] -= oy
            cv2.polylines(out, [shifted], isClosed=False, color=1, thickness=3)
    dil = cv2.dilate(mask_crop.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    out &= dil
    if out.sum() == 0 and mask_crop.sum() > 0:
        edt = ndimage.distance_transform_edt(mask_crop > 0)
        mx = np.maximum(ndimage.maximum_filter(edt, size=3), 1e-6)
        out = ((edt > 0) & (edt >= 0.85 * mx)).astype(np.uint8)
    return out


def _inst_sampling_weight(
    bbox: tuple[int, int, int, int],
    area: int,
    geom: dict[str, float] | None,
) -> tuple[float, bool]:
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    mean_w = float(area) / float(max(bw, bh))
    is_thin = mean_w <= 6.5
    w = 1.0
    if is_thin:
        w *= 1.65
    if geom is not None:
        icx = 0.5 * (bbox[0] + bbox[2])
        icy = 0.5 * (bbox[1] + bbox[3])
        rho = math.hypot(icx - geom["cx"], icy - geom["cy"]) / max(100.0, geom["R_sun"])
        if rho >= 0.78:
            w *= 1.5
    return float(min(w, 2.5)), bool(is_thin)


def build_stem_targets_v10(
    labels_json_path: Path,
    allowed_stems: set[str],
    vault_stems: set[str],
    val_stems: set[str],
    cache_dir: Path,
    load_physics_fn,
    geom_map: dict[str, dict[str, float]] | None = None,
) -> tuple[
    dict[str, StemTargets],
    dict[str, dict[str, list[dict]]],
    dict[str, dict[str, list[dict]]],
    dict[str, dict[str, list[dict]]],
    dict[str, dict[str, list[dict]]],
    dict[int, dict[str, dict[str, list[dict]]]],
]:
    """Parse annotations for allowed_stems only (strict zero-vault isolation)."""
    assert len(allowed_stems & vault_stems) == 0, "Vault stems leaked into allowed_stems!"
    raw = json.loads(labels_json_path.read_text(encoding="utf-8"))

    img_to_stem_ann: dict[str, str] = {}
    stem_ann_ids: dict[str, list[str]] = {}
    for im in raw["images"]:
        fname = str(im.get("file_name", "")).replace("\\", "/")
        raw_stem = Path(fname).stem
        base_stem = re.sub(r"_e\d+$", "", raw_stem)
        if not base_stem and im.get("stem"):
            base_stem = str(im["stem"])
        if base_stem in vault_stems or base_stem not in allowed_stems:
            continue
        img_id = str(im["id"])
        img_to_stem_ann[img_id] = base_stem
        stem_ann_ids.setdefault(base_stem, []).append(img_id)

    anns_by_img_id: dict[str, list[dict[str, Any]]] = {iid: [] for iid in img_to_stem_ann}
    for ann in raw["annotations"]:
        iid = str(ann["image_id"])
        if iid in anns_by_img_id:
            anns_by_img_id[iid].append(ann)

    targets_by_stem: dict[str, StemTargets] = {}
    raw_val_gt: dict[str, dict[str, list[dict]]] = {}
    cons_val_gt: dict[str, dict[str, list[dict]]] = {}
    pure_v2_val_gt: dict[str, dict[str, list[dict]]] = {}
    a1_val_gt: dict[str, dict[str, list[dict]]] = {}
    as_subsets_val_gt: dict[int, dict[str, dict[str, list[dict]]]] = {1: {}, 2: {}, 3: {}}

    k3 = np.ones((3, 3), dtype=np.uint8)

    for stem in sorted(allowed_stems):
        entry_ids = sorted(stem_ann_ids.get(stem, []))
        n_ann = len(entry_ids)
        geom = geom_map.get(stem) if geom_map else None
        if n_ann == 0:
            targets_by_stem[stem] = StemTargets(stem, 0, [], None, None, is_clean_stem=False)
            continue

        by_ann_rles: dict[str, list[dict]] = {}
        flat_items: list[dict[str, Any]] = []
        for a_idx, eid in enumerate(entry_ids):
            rles_here = []
            for ann in anns_by_img_id[eid]:
                r = _ann_to_rle(ann)
                area = int(mu_coco.area(r))
                if area <= 0:
                    continue
                rles_here.append(r)
                bb = mu_coco.toBbox(r)
                x1 = max(0, int(np.floor(bb[0])))
                y1 = max(0, int(np.floor(bb[1])))
                x2 = min(IMG_SIZE, int(np.ceil(bb[0] + bb[2])))
                y2 = min(IMG_SIZE, int(np.ceil(bb[1] + bb[3])))
                flat_items.append({
                    "ann_idx": a_idx,
                    "rle": r,
                    "area": area,
                    "bbox": (x1, y1, x2, y2),
                    "spine": _parse_spine_pts(ann),
                })
            by_ann_rles[eid] = rles_here

        if stem in val_stems:
            raw_val_gt[stem] = by_ann_rles
            if n_ann == 1:
                a1_val_gt[stem] = by_ann_rles
            as_subsets_val_gt[min(3, n_ann)][stem] = by_ann_rles

        if n_ann == 1:
            insts = []
            for it in flat_items:
                bx1, by1, bx2, by2 = it["bbox"]
                mcrop = mu_coco.decode(it["rle"])[by1:by2, bx1:bx2].astype(np.uint8)
                sp_list = [it["spine"]] if it["spine"] is not None else []
                scrop = render_spine_crop(mcrop > 0, sp_list, bx1, by1)
                sw_val, is_thin = _inst_sampling_weight(it["bbox"], it["area"], geom)
                insts.append(
                    InstanceTarget(
                        bbox=it["bbox"],
                        rle=it["rle"],
                        mask_crop=mcrop,
                        spine_crop=scrop,
                        n_annotators=1,
                        weight=1.0,
                        is_consensus=True,
                        area=it["area"],
                        sampling_weight=sw_val,
                        is_thin=is_thin,
                    )
                )
            targets_by_stem[stem] = StemTargets(stem, 1, insts, None, None, is_clean_stem=False)
            if stem in val_stems:
                cons_val_gt[stem] = {"consensus": [it.rle for it in insts]}
            continue

        m = len(flat_items)
        parent = list(range(m))

        def _find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def _union(u: int, v: int) -> None:
            ru, rv = _find(u), _find(v)
            if ru != rv:
                parent[rv] = ru

        for i in range(m):
            for j in range(i + 1, m):
                if flat_items[i]["ann_idx"] == flat_items[j]["ann_idx"]:
                    continue
                if not _box_iou_fast(flat_items[i]["bbox"], flat_items[j]["bbox"]):
                    continue
                iou = float(mu_coco.iou([flat_items[i]["rle"]], [flat_items[j]["rle"]], [0])[0, 0])
                if iou >= 0.15:
                    _union(i, j)
                elif iou > 0.0:
                    inter = iou * (flat_items[i]["area"] + flat_items[j]["area"]) / (1.0 + iou)
                    ov_min = inter / max(1.0, min(flat_items[i]["area"], flat_items[j]["area"]))
                    if ov_min >= 0.35:
                        _union(i, j)

        clusters: dict[int, list[dict[str, Any]]] = {}
        for i in range(m):
            clusters.setdefault(_find(i), []).append(flat_items[i])

        hw3 = load_physics_fn(cache_dir, stem)
        d_multi_u8 = hw3[..., 1]
        ignore_full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        insts = []
        has_rescued_singleton = False

        for comp in clusters.values():
            distinct_anns = sorted({it["ann_idx"] for it in comp})
            votes = len(distinct_anns)
            ux1 = max(0, min(it["bbox"][0] for it in comp) - 2)
            uy1 = max(0, min(it["bbox"][1] for it in comp) - 2)
            ux2 = min(IMG_SIZE, max(it["bbox"][2] for it in comp) + 2)
            uy2 = min(IMG_SIZE, max(it["bbox"][3] for it in comp) + 2)

            if votes >= 2:
                sdf_acc = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.float32)
                w_sum = 0.0
                for a_id in distinct_anns:
                    merged_rle = mu_coco.merge([it["rle"] for it in comp if it["ann_idx"] == a_id], intersect=False)
                    sub = mu_coco.decode(merged_rle)[uy1:uy2, ux1:ux2] > 0
                    sub_area = int(sub.sum())
                    if sub.any() and (~sub).any():
                        phi_a = (ndimage.distance_transform_edt(sub) - 0.5) - (ndimage.distance_transform_edt(~sub) - 0.5)
                    elif sub.any():
                        phi_a = np.ones_like(sub, dtype=np.float32) * 10.0
                    else:
                        phi_a = np.zeros_like(sub, dtype=np.float32) - 10.0
                    w_a = 1.0 / math.sqrt(max(16.0, float(sub_area)))
                    sdf_acc += w_a * phi_a.astype(np.float32)
                    w_sum += w_a

                psi = sdf_acc / max(1e-6, w_sum)
                cons_crop = (psi >= 0.0).astype(np.uint8)
                if cons_crop.sum() < 15:
                    best_it = max(comp, key=lambda it: it["area"])
                    cons_crop = mu_coco.decode(best_it["rle"])[uy1:uy2, ux1:ux2].astype(np.uint8)
                if cons_crop.sum() < 15:
                    continue
                full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                full[uy1:uy2, ux1:ux2] = cons_crop
                bb_new = _mask_bbox(cons_crop)
                if bb_new is None:
                    continue
                rle_new = mu_coco.encode(np.asfortranarray(full))
                spines = [it["spine"] for it in comp if it["spine"] is not None]
                tb = (ux1 + bb_new[0], uy1 + bb_new[1], ux1 + bb_new[2], uy1 + bb_new[3])
                mcrop = cons_crop[bb_new[1] : bb_new[3], bb_new[0] : bb_new[2]].copy()
                scrop = render_spine_crop(mcrop > 0, spines, tb[0], tb[1])
                area_new = int(mcrop.sum())
                sw_val, is_thin = _inst_sampling_weight(tb, area_new, geom)
                insts.append(
                    InstanceTarget(
                        bbox=tb,
                        rle=rle_new,
                        mask_crop=mcrop,
                        spine_crop=scrop,
                        n_annotators=n_ann,
                        weight=1.0,
                        is_consensus=True,
                        area=area_new,
                        sampling_weight=sw_val,
                        is_thin=is_thin,
                    )
                )
            else:
                it = comp[0]
                bx1, by1, bx2, by2 = it["bbox"]
                sub_mask = mu_coco.decode(it["rle"])[by1:by2, bx1:bx2] > 0
                if not sub_mask.any():
                    continue
                d_vals = d_multi_u8[by1:by2, bx1:bx2][sub_mask]
                med_d = float(np.median(d_vals)) / 255.0

                rescue_a2 = (n_ann == 2) and (it["area"] >= 120) and (med_d >= 0.40)
                if rescue_a2:
                    has_rescued_singleton = True
                    mcrop = sub_mask.astype(np.uint8)
                    sp_list = [it["spine"]] if it["spine"] is not None else []
                    scrop = render_spine_crop(mcrop > 0, sp_list, bx1, by1)
                    sw_val, is_thin = _inst_sampling_weight(it["bbox"], it["area"], geom)
                    insts.append(
                        InstanceTarget(
                            bbox=it["bbox"],
                            rle=it["rle"],
                            mask_crop=mcrop,
                            spine_crop=scrop,
                            n_annotators=n_ann,
                            weight=0.75,
                            is_consensus=False,
                            area=it["area"],
                            sampling_weight=sw_val,
                            is_thin=is_thin,
                        )
                    )
                else:
                    py1, px1 = max(0, by1 - 2), max(0, bx1 - 2)
                    py2, px2 = min(IMG_SIZE, by2 + 2), min(IMG_SIZE, bx2 + 2)
                    sub_pad = mu_coco.decode(it["rle"])[py1:py2, px1:px2].astype(np.uint8)
                    ignore_full[py1:py2, px1:px2] |= cv2.dilate(sub_pad, k3, iterations=1)

        for inst in insts:
            bx1, by1, bx2, by2 = inst.bbox
            ignore_full[by1:by2, bx1:bx2] &= ~(inst.mask_crop > 0)

        ign_rle = mu_coco.encode(np.asfortranarray(ignore_full)) if ignore_full.any() else None
        ign_bb = _mask_bbox(ignore_full)
        ign_crop = (
            (ign_bb[0], ign_bb[1], ign_bb[2], ign_bb[3], ignore_full[ign_bb[1] : ign_bb[3], ign_bb[0] : ign_bb[2]].copy())
            if ign_bb is not None
            else None
        )
        targets_by_stem[stem] = StemTargets(
            stem=stem,
            n_annotators=n_ann,
            instances=insts,
            ignore_rle=ign_rle,
            ignore_crop=ign_crop,
            is_clean_stem=(n_ann >= 2 and not has_rescued_singleton),
        )
        if stem in val_stems:
            cons_val_gt[stem] = {"consensus": [it.rle for it in insts]}
            if n_ann >= 2:
                pure_v2_val_gt[stem] = {"v2": [it.rle for it in insts if it.is_consensus]}

    return targets_by_stem, raw_val_gt, cons_val_gt, pure_v2_val_gt, a1_val_gt, as_subsets_val_gt


def build_train_bitmasks(
    train_stems: list[str],
    targets_map: dict[str, StemTargets],
    geom_map: dict[str, dict[str, float]],
    workers: int = 6,
) -> dict[str, np.ndarray]:
    """Pack [bit0=fg, bit1=core, bit2=spine, bit3=valid, bit4=thin_filament] into uint8 per train stem."""

    def _one(s: str) -> tuple[str, np.ndarray]:
        st = targets_map[s]
        g = geom_map[s]
        on_disk, _ = compute_solar_geometry(g["cx"], g["cy"], g["R_sun"], IMG_SIZE)
        bmask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        bmask[on_disk] = 8  # bit 3 = valid

        if st.ignore_crop is not None:
            ix1, iy1, ix2, iy2, icrop = st.ignore_crop
            bmask[iy1:iy2, ix1:ix2][icrop > 0] = 0

        for inst in st.instances:
            bx1, by1, bx2, by2 = inst.bbox
            mc = inst.mask_crop > 0
            if not mc.any():
                continue
            edt_in = ndimage.distance_transform_edt(mc)
            core = (edt_in >= 2.2) & mc
            if not core.any():
                core = inst.spine_crop > 0
            sp = inst.spine_crop > 0
            thin_flag = inst.is_thin or (float(edt_in.max()) <= 3.5)
            win = bmask[by1:by2, bx1:bx2]
            win[mc] |= np.uint8(1 | 8 | (16 if thin_flag else 0))
            win[core] |= np.uint8(2)
            win[sp] |= np.uint8(4)
        return s, bmask

    out: dict[str, np.ndarray] = {}
    n_w = max(1, min(workers, os.cpu_count() or 4, len(train_stems) or 1))
    with ThreadPoolExecutor(max_workers=n_w) as ex:
        for s, bm in ex.map(_one, train_stems):
            out[s] = bm
    return out
