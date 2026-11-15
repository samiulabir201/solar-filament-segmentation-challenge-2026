"""Topology-safe Hybrid Native 1:1 Instance Formation (Geodesic Hysteresis + 24-d Meta-Scorer) & Dark-UI Diagnostics."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

import referee_metric
from physics_v2 import IMG_SIZE, SLICE_SIZE, load_cached_physics_u8
from stage1_detector import ProposalROI

# Pre-allocated reusable 2048x2048 dirty-window compositing buffers
_OWNER_BUF = np.full((IMG_SIZE, IMG_SIZE), -1, dtype=np.int32)
_SCORE_BUF = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
_TMP_MASK = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8, order="F")

TAU_HYST_GRID: tuple[tuple[float, float], ...] = (
    (0.46, 0.00),
    (0.50, 0.00),
    (0.54, 0.00),
    (0.50, 0.06),
    (0.54, 0.06),
    (0.54, 0.10),
    (0.58, 0.08),
    (0.58, 0.12),
)


def tau_pair_key(tau_seed: float, delta_hyst: float) -> str:
    return f"{float(tau_seed):.2f}_{float(delta_hyst):.2f}"


FEATURE_NAMES = (
    "p_fg_mean",
    "p_fg_q85",
    "p_fg_max",
    "p_core_mean",
    "p_spine_cov",
    "peer_agree",
    "boundary_drop",
    "s_det_best",
    "det_iou_best",
    "s1_view_frac",
    "d_multi_med",
    "d_barb_med",
    "t_ribbon_med",
    "log_area",
    "thinness",
    "mu_cent",
    "p_fg_x_agree",
    "t_rib_q90",
    "d_multi_q90",
    "snr_local",
    "edge_contrast",
    "elongation",
    "circularity",
    "plage_ring_q95",
)


@dataclass
class HybridCandidate:
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2) in native 2048x2048
    mask_u8: np.ndarray                  # (y2-y1, x2-x1) uint8 in {0, 1}
    features: np.ndarray                 # (24,) float32
    v_zero: float
    is_healed: bool = False
    is_rescued: bool = False
    v_hat: float = 0.0


def _mask_bbox_u8(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _find_closest_points(m1_pts: np.ndarray, m2_pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Find closest (y, x) pixel pair between two boundary point sets."""
    if len(m1_pts) > 96:
        m1_pts = m1_pts[np.linspace(0, len(m1_pts) - 1, 96, dtype=int)]
    if len(m2_pts) > 96:
        m2_pts = m2_pts[np.linspace(0, len(m2_pts) - 1, 96, dtype=int)]
    diff = m1_pts[:, None, :] - m2_pts[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    flat = int(np.argmin(d2))
    i0, j0 = divmod(flat, d2.shape[1])
    return float(math.sqrt(float(d2[i0, j0]))), m1_pts[i0], m2_pts[j0]


def _contour_yx(mask_crop: np.ndarray, ox: int, oy: int) -> np.ndarray:
    b = mask_crop.astype(np.uint8)
    er = cv2.erode(b, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = b & (~er)
    ys, xs = np.nonzero(edge if edge.any() else b)
    return np.stack([ys + oy, xs + ox], axis=1).astype(np.float32)


def extract_stem_candidates_multitau(
    stem: str,
    x0: int,
    y0: int,
    p_fg: np.ndarray,
    p_core: np.ndarray,
    p_spine: np.ndarray,
    p_disagree: np.ndarray,
    hw3_2048_u8: np.ndarray,
    geom: dict[str, float],
    s1_props: list[ProposalROI],
    tau_hyst_grid: tuple[tuple[float, float], ...] = TAU_HYST_GRID,
) -> dict[str, list[HybridCandidate]]:
    """Build crisp 1:1 native connected components via Geodesic Hysteresis + R-CNN touch splitting + waist healing."""
    sh, sw = p_fg.shape
    hw3_sl = np.asarray(hw3_2048_u8[y0 : y0 + sh, x0 : x0 + sw], dtype=np.uint8)
    i_flat_f = hw3_sl[..., 0].astype(np.float32)
    bg_15 = cv2.boxFilter(i_flat_f, -1, (15, 15), normalize=True)
    bg_sq_15 = cv2.boxFilter(i_flat_f * i_flat_f, -1, (15, 15), normalize=True)
    loc_sd_sl = np.sqrt(np.maximum(bg_sq_15 - bg_15 * bg_15, 4.0))
    plage_sl = np.clip((i_flat_f - bg_15) * (6.0 / 255.0), 0.0, 1.0)
    d_barb_sl = np.clip((bg_15 - i_flat_f) * (10.0 / 255.0), 0.0, 1.0)
    d_multi_sl = hw3_sl[..., 1].astype(np.float32) * (1.0 / 255.0)
    t_rib_sl = hw3_sl[..., 2].astype(np.float32) * (1.0 / 255.0)

    # Pre-project Stage-1 proposals into slice-local coordinates
    local_s1: list[dict[str, Any]] = []
    for pr in s1_props:
        bx1, by1, bx2, by2 = pr.bbox
        sx1 = max(0, bx1 - x0)
        sy1 = max(0, by1 - y0)
        sx2 = min(sw, bx2 - x0)
        sy2 = min(sh, by2 - y0)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        pm = pr.mask_f16[(sy1 + y0 - by1) : (sy2 + y0 - by1), (sx1 + x0 - bx1) : (sx2 + x0 - bx1)] >= 0.48
        area_d = int(pm.sum())
        if area_d < 16:
            continue
        local_s1.append({
            "sbbox": (sx1, sy1, sx2, sy2),
            "mask": pm,
            "area": area_d,
            "s_det": float(pr.s_det),
            "view_agree": float(pr.view_agree),
        })

    cx_s, cy_s, r_sun = float(geom["cx"]), float(geom["cy"]), max(100.0, float(geom["R_sun"]))
    k3 = np.ones((3, 3), dtype=np.uint8)
    k3_bool = np.ones((3, 3), dtype=bool)
    out_by_tau: dict[str, list[HybridCandidate]] = {}

    for tau_seed, delta_hyst in tau_hyst_grid:
        tkey = tau_pair_key(tau_seed, delta_hyst)
        spine_gate = max(0.28, float(tau_seed) - 0.12)
        m_seed = (p_fg >= tau_seed) | ((p_spine >= 0.48) & (p_fg >= spine_gate))
        if delta_hyst > 1e-6:
            low_fg = max(0.24, float(tau_seed) - float(delta_hyst))
            low_sp_fg = max(0.22, float(tau_seed) - float(delta_hyst) - 0.04)
            m_low = (p_fg >= low_fg) | ((p_spine >= 0.42) & (p_fg >= low_sp_fg))
            m_bin = ndimage.binary_propagation(m_seed, mask=(m_low | m_seed), structure=k3_bool)
        else:
            m_bin = m_seed
        n_cc, lbl_map, stats, _ = cv2.connectedComponentsWithStats(m_bin.astype(np.uint8), connectivity=8)

        # Raw components -> list of (sbbox, mask_crop, is_healed, is_rescued)
        raw_comps: list[tuple[tuple[int, int, int, int], np.ndarray, bool, bool]] = []
        for cid in range(1, n_cc):
            cx1, cy1, cw, ch, area_c = [int(v) for v in stats[cid]]
            if area_c < 20:
                continue
            cx2, cy2 = cx1 + cw, cy1 + ch
            cmask = (lbl_map[cy1:cy2, cx1:cx2] == cid)

            # 1. Detector-Verified Touch Separation (only when >=2 disjoint R-CNN objects overlap >=18% each)
            split_done = False
            if area_c >= 140 and len(local_s1) >= 2:
                overlaps = []
                for di, det in enumerate(local_s1):
                    if det["s_det"] < 0.44:
                        continue
                    dx1, dy1, dx2, dy2 = det["sbbox"]
                    ix1, iy1 = max(cx1, dx1), max(cy1, dy1)
                    ix2, iy2 = min(cx2, dx2), min(cy2, dy2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    sub_c = cmask[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1]
                    sub_d = det["mask"][iy1 - dy1 : iy2 - dy1, ix1 - dx1 : ix2 - dx1]
                    inter = int(np.logical_and(sub_c, sub_d).sum())
                    if inter >= 60 and float(inter) / float(det["area"]) >= 0.18:
                        overlaps.append((det["s_det"], di, det))

                if len(overlaps) >= 2:
                    overlaps.sort(key=lambda t: -t[0])
                    chosen_dets: list[dict[str, Any]] = []
                    for _sc, _di, cand_d in overlaps:
                        dx1, dy1, dx2, dy2 = cand_d["sbbox"]
                        disjoint_all = True
                        for prev in chosen_dets:
                            px1, py1, px2, py2 = prev["sbbox"]
                            ix1, iy1 = max(dx1, px1), max(dy1, py1)
                            ix2, iy2 = min(dx2, px2), min(dy2, py2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter_dd = float(
                                    np.logical_and(
                                        cand_d["mask"][iy1 - dy1 : iy2 - dy1, ix1 - dx1 : ix2 - dx1],
                                        prev["mask"][iy1 - py1 : iy2 - py1, ix1 - px1 : ix2 - px1],
                                    ).sum()
                                )
                                iou_dd = inter_dd / max(1.0, float(cand_d["area"] + prev["area"]) - inter_dd)
                                if iou_dd >= 0.12:
                                    disjoint_all = False
                                    break
                        if disjoint_all:
                            chosen_dets.append(cand_d)

                    if len(chosen_dets) >= 2:
                        # Compute weighted Voronoi costs inside cmask across disjoint seeds
                        cost_stack = []
                        valid_seeds = 0
                        p_fg_c = p_fg[cy1:cy2, cx1:cx2]
                        for det in chosen_dets[:4]:
                            seed = np.zeros_like(cmask, dtype=bool)
                            dx1, dy1, dx2, dy2 = det["sbbox"]
                            ix1, iy1 = max(cx1, dx1), max(cy1, dy1)
                            ix2, iy2 = min(cx2, dx2), min(cy2, dy2)
                            seed[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1] = (
                                det["mask"][iy1 - dy1 : iy2 - dy1, ix1 - dx1 : ix2 - dx1]
                                & cmask[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1]
                            )
                            if int(seed.sum()) >= 15:
                                valid_seeds += 1
                                dmap = ndimage.distance_transform_edt(~seed).astype(np.float32) * (1.20 - p_fg_c)
                                cost_stack.append(dmap)
                        if valid_seeds >= 2:
                            owner = np.argmin(np.stack(cost_stack, axis=0), axis=0)
                            parts = []
                            for k_part in range(len(cost_stack)):
                                pm_k = cmask & (owner == k_part)
                                n_pk, l_pk, st_pk, _ = cv2.connectedComponentsWithStats(pm_k.astype(np.uint8), connectivity=8)
                                for sub_id in range(1, n_pk):
                                    sx_i, sy_i, sw_i, sh_i, sa_i = [int(v) for v in st_pk[sub_id]]
                                    if sa_i >= 20:
                                        sub_m = (l_pk[sy_i : sy_i + sh_i, sx_i : sx_i + sw_i] == sub_id)
                                        parts.append(((cx1 + sx_i, cy1 + sy_i, cx1 + sx_i + sw_i, cy1 + sy_i + sh_i), sub_m, False, False))
                            if len(parts) >= 2:
                                raw_comps.extend(parts)
                                split_done = True

            if not split_done:
                raw_comps.append(((cx1, cy1, cx2, cy2), cmask, False, False))

        # 2. Waist-Only Split Healing
        n_raw = len(raw_comps)
        parent = list(range(n_raw))

        def _find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def _union(u: int, v: int) -> None:
            ru, rv = _find(u), _find(v)
            if ru != rv:
                parent[rv] = ru

        contours = [_contour_yx(m, bb[0], bb[1]) for bb, m, _, _ in raw_comps]
        comp_areas = [int(m.sum()) for _, m, _, _ in raw_comps]

        # Precompute which high-conf dets overlap which raw_comp by >= 45%
        comp_strong_dets: list[set[int]] = [set() for _ in range(n_raw)]
        for i, (bb, m, _, _) in enumerate(raw_comps):
            bx1, by1, bx2, by2 = bb
            ai = max(1, comp_areas[i])
            for di, det in enumerate(local_s1):
                if det["s_det"] < 0.35:
                    continue
                dx1, dy1, dx2, dy2 = det["sbbox"]
                ix1, iy1 = max(bx1, dx1), max(by1, dy1)
                ix2, iy2 = min(bx2, dx2), min(by2, dy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = int(
                    np.logical_and(
                        m[iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1],
                        det["mask"][iy1 - dy1 : iy2 - dy1, ix1 - dx1 : ix2 - dx1],
                    ).sum()
                )
                if float(inter) / float(ai) >= 0.45:
                    comp_strong_dets[i].add(di)

        bridges_by_root: dict[int, list[tuple[tuple[int, int, int, int], np.ndarray]]] = {}
        edges: list[tuple[int, int, tuple[int, int, int, int], np.ndarray]] = []
        for i in range(n_raw):
            ix1, iy1, ix2, iy2 = raw_comps[i][0]
            for j in range(i + 1, n_raw):
                jx1, jy1, jx2, jy2 = raw_comps[j][0]
                if max(ix1, jx1) - min(ix2, jx2) > 20 or max(iy1, jy1) - min(iy2, jy2) > 20:
                    continue
                # Veto if claimed by two distinct high-conf Stage-1 detections
                di_set, dj_set = comp_strong_dets[i], comp_strong_dets[j]
                if di_set and dj_set and not (di_set & dj_set):
                    # Check whether those separate dets are actually disjoint
                    veto = False
                    for a_di in di_set:
                        da = local_s1[a_di]
                        ax1, ay1, ax2, ay2 = da["sbbox"]
                        for b_dj in dj_set:
                            db = local_s1[b_dj]
                            bx1_, by1_, bx2_, by2_ = db["sbbox"]
                            ox1, oy1 = max(ax1, bx1_), max(ay1, by1_)
                            ox2, oy2 = min(ax2, bx2_), min(ay2, by2_)
                            if ox2 <= ox1 or oy2 <= oy1:
                                veto = True
                                break
                            inter_ab = float(
                                np.logical_and(
                                    da["mask"][oy1 - ay1 : oy2 - ay1, ox1 - ax1 : ox2 - ax1],
                                    db["mask"][oy1 - by1_ : oy2 - by1_, ox1 - bx1_ : ox2 - bx1_],
                                ).sum()
                            )
                            if inter_ab / max(1.0, float(da["area"] + db["area"]) - inter_ab) < 0.15:
                                veto = True
                                break
                        if veto:
                            break
                    if veto:
                        continue

                dist, pt_i, pt_j = _find_closest_points(contours[i], contours[j])
                if dist > 20.0:
                    continue

                lx1 = max(0, int(min(pt_i[1], pt_j[1])) - 2)
                ly1 = max(0, int(min(pt_i[0], pt_j[0])) - 2)
                lx2 = min(sw, int(max(pt_i[1], pt_j[1])) + 3)
                ly2 = min(sh, int(max(pt_i[0], pt_j[0])) + 3)
                line_m = np.zeros((ly2 - ly1, lx2 - lx1), dtype=np.uint8)
                cv2.line(
                    line_m,
                    (int(round(pt_i[1])) - lx1, int(round(pt_i[0])) - ly1),
                    (int(round(pt_j[1])) - lx1, int(round(pt_j[0])) - ly1),
                    color=1,
                    thickness=3,
                )
                lbool = line_m > 0
                if not lbool.any():
                    continue
                fg_line = float(p_fg[ly1:ly2, lx1:lx2][lbool].mean())
                sp_line = float(p_spine[ly1:ly2, lx1:lx2][lbool].mean())

                shared_strong = [d for d in (di_set & dj_set) if local_s1[d]["s_det"] >= 0.42]
                case_a = bool(shared_strong) and (fg_line >= 0.18)
                case_b = (dist <= 10.0) and (fg_line >= 0.28) and (sp_line >= 0.22)
                if case_a or case_b:
                    gate = 0.16 if case_a else 0.20
                    waist_m = lbool & (p_fg[ly1:ly2, lx1:lx2] >= gate)
                    edges.append((i, j, (lx1, ly1, lx2, ly2), waist_m))
                    _union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n_raw):
            groups.setdefault(_find(i), []).append(i)
        for i, _j, lbb, wm in edges:
            bridges_by_root.setdefault(_find(i), []).append((lbb, wm))

        merged_comps: list[tuple[tuple[int, int, int, int], np.ndarray, bool, bool]] = []
        for root_id, idxs in groups.items():
            if len(idxs) == 1 and root_id not in bridges_by_root:
                merged_comps.append(raw_comps[idxs[0]])
                continue
            ux1 = min(raw_comps[k][0][0] for k in idxs)
            uy1 = min(raw_comps[k][0][1] for k in idxs)
            ux2 = max(raw_comps[k][0][2] for k in idxs)
            uy2 = max(raw_comps[k][0][3] for k in idxs)
            for lbb, _ in bridges_by_root.get(root_id, []):
                ux1, uy1 = min(ux1, lbb[0]), min(uy1, lbb[1])
                ux2, uy2 = max(ux2, lbb[2]), max(uy2, lbb[3])
            canvas = np.zeros((uy2 - uy1, ux2 - ux1), dtype=bool)
            for k in idxs:
                cx1, cy1, cx2, cy2 = raw_comps[k][0]
                canvas[cy1 - uy1 : cy2 - uy1, cx1 - ux1 : cx2 - ux1] |= raw_comps[k][1]
            for (lx1, ly1, lx2, ly2), wm in bridges_by_root.get(root_id, []):
                canvas[ly1 - uy1 : ly2 - uy1, lx1 - ux1 : lx2 - ux1] |= wm
            merged_comps.append(((ux1, uy1, ux2, uy2), canvas, True, False))

        # 3. Uncovered High-Confidence R-CNN Rescue
        for det in local_s1:
            if det["s_det"] < 0.48:
                continue
            dx1, dy1, dx2, dy2 = det["sbbox"]
            dm = det["mask"]
            ov_px = int(np.logical_and(dm, m_bin[dy1:dy2, dx1:dx2]).sum())
            if float(ov_px) / float(max(1, det["area"])) >= 0.20:
                continue
            res_m = dm & (p_fg[dy1:dy2, dx1:dx2] >= 0.20)
            ra = int(res_m.sum())
            if ra >= 28 and float(p_fg[dy1:dy2, dx1:dx2][res_m].mean()) >= 0.26:
                rbb = _mask_bbox_u8(res_m.astype(np.uint8))
                if rbb is not None:
                    rx1, ry1, rx2, ry2 = rbb
                    merged_comps.append((
                        (dx1 + rx1, dy1 + ry1, dx1 + rx2, dy1 + ry2),
                        res_m[ry1:ry2, rx1:rx2].copy(),
                        False,
                        True,
                    ))

        # 4. Compute orthogonal 17-d features + v_zero for every candidate
        cand_list: list[HybridCandidate] = []
        for (sx1, sy1, sx2, sy2), m_crop, is_healed, is_rescued in merged_comps:
            mc = m_crop > 0
            area = int(mc.sum())
            if area < 20:
                continue
            fg_vals = p_fg[sy1:sy2, sx1:sx2][mc]
            p_fg_mean = float(fg_vals.mean())
            p_fg_q85 = float(np.quantile(fg_vals, 0.85))
            p_fg_max = float(fg_vals.max())
            p_core_mean = float(p_core[sy1:sy2, sx1:sx2][mc].mean())
            p_spine_cov = float((p_spine[sy1:sy2, sx1:sx2][mc] >= 0.35).mean())
            peer_agree = 1.0 - float(p_disagree[sy1:sy2, sx1:sx2][mc].mean())

            # Boundary sharpness: inner perimeter p_fg minus 1-px outer ring p_fg
            px1, py1 = max(0, sx1 - 1), max(0, sy1 - 1)
            px2, py2 = min(sw, sx2 + 1), min(sh, sy2 + 1)
            pad_m = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            pad_m[sy1 - py1 : sy2 - py1, sx1 - px1 : sx2 - px1] = mc.astype(np.uint8)
            er_m = cv2.erode(pad_m, k3, iterations=1)
            dil_m = cv2.dilate(pad_m, k3, iterations=1)
            in_b = (pad_m > 0) & (er_m == 0)
            out_r = (dil_m > 0) & (pad_m == 0)
            fg_pad = p_fg[py1:py2, px1:px2]
            in_p = float(fg_pad[in_b].mean()) if in_b.any() else p_fg_mean
            out_p = float(fg_pad[out_r].mean()) if out_r.any() else 0.0
            boundary_drop = float(np.clip(in_p - out_p, -0.5, 1.0))

            s_det_best, det_iou_best, s1_view_frac = 0.0, 0.0, 0.0
            for det in local_s1:
                dx1, dy1, dx2, dy2 = det["sbbox"]
                ix1, iy1 = max(sx1, dx1), max(sy1, dy1)
                ix2, iy2 = min(sx2, dx2), min(sy2, dy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = float(
                    np.logical_and(
                        mc[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1],
                        det["mask"][iy1 - dy1 : iy2 - dy1, ix1 - dx1 : ix2 - dx1],
                    ).sum()
                )
                if inter <= 0.0:
                    continue
                iou_d = inter / max(1.0, float(area + det["area"]) - inter)
                score_match = det["s_det"] * max(0.35, iou_d)
                if score_match > s_det_best * max(0.35, det_iou_best):
                    s_det_best = float(det["s_det"])
                    det_iou_best = float(iou_d)
                    s1_view_frac = float(det["view_agree"])

            d_multi_med = float(np.median(d_multi_sl[sy1:sy2, sx1:sx2][mc]))
            d_barb_med = float(np.median(d_barb_sl[sy1:sy2, sx1:sx2][mc]))
            t_rib_med = float(np.median(t_rib_sl[sy1:sy2, sx1:sx2][mc]))
            log_area = float(math.log1p(area))
            bw, bh = max(1, sx2 - sx1), max(1, sy2 - sy1)
            thinness = float(min(1.0, float(area) / float(bw * bh)))
            gx_c = float(x0) + 0.5 * (sx1 + sx2)
            gy_c = float(y0) + 0.5 * (sy1 + sy2)
            rho = math.hypot(gx_c - cx_s, gy_c - cy_s) / r_sun
            mu_cent = float(math.sqrt(max(0.0, 1.0 - rho * rho)))
            p_fg_x_agree = p_fg_mean * peer_agree

            t_rib_q90 = float(np.quantile(t_rib_sl[sy1:sy2, sx1:sx2][mc], 0.90))
            d_multi_q90 = float(np.quantile(d_multi_sl[sy1:sy2, sx1:sx2][mc], 0.90))
            snr_local = float(
                np.clip(
                    (float(bg_15[sy1:sy2, sx1:sx2][mc].mean()) - float(i_flat_f[sy1:sy2, sx1:sx2][mc].mean()))
                    / max(1.0, float(loc_sd_sl[sy1:sy2, sx1:sx2][mc].mean())),
                    -2.0,
                    12.0,
                )
            )
            if_pad = i_flat_f[py1:py2, px1:px2]
            in_i = float(if_pad[in_b].mean()) if in_b.any() else 0.0
            out_i = float(if_pad[out_r].mean()) if out_r.any() else in_i
            edge_contrast = float(np.clip((out_i - in_i) / 32.0, -1.0, 4.0))
            ys_m, xs_m = np.nonzero(mc)
            if len(xs_m) >= 3:
                dx_m = xs_m.astype(np.float64) - float(xs_m.mean())
                dy_m = ys_m.astype(np.float64) - float(ys_m.mean())
                cxx = float((dx_m * dx_m).mean())
                cyy = float((dy_m * dy_m).mean())
                cxy = float((dx_m * dy_m).mean())
                tr = cxx + cyy
                disc = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
                l1 = max(1.0, 0.5 * (tr + disc))
                l2 = max(1.0, 0.5 * (tr - disc))
                elongation = float(min(25.0, math.sqrt(l1 / l2)))
            else:
                elongation = 1.0
            perim = max(1.0, float(in_b.sum()))
            circularity = float(min(1.0, 4.0 * math.pi * float(area) / (perim * perim)))
            pl_pad = plage_sl[py1:py2, px1:px2]
            plage_ring_q95 = float(np.quantile(pl_pad[out_r], 0.95)) if out_r.any() else 0.0

            feats = np.array(
                [
                    p_fg_mean, p_fg_q85, p_fg_max, p_core_mean, p_spine_cov,
                    peer_agree, boundary_drop, s_det_best, det_iou_best, s1_view_frac,
                    d_multi_med, d_barb_med, t_rib_med, log_area, thinness,
                    mu_cent, p_fg_x_agree,
                    t_rib_q90, d_multi_q90, snr_local, edge_contrast, elongation, circularity, plage_ring_q95,
                ],
                dtype=np.float32,
            )
            v_zero = (
                0.45 * p_fg_x_agree
                + 0.35 * (s_det_best * max(0.40, det_iou_best))
                + 0.20 * max(0.0, boundary_drop)
            )
            cand_list.append(
                HybridCandidate(
                    bbox=(x0 + sx1, y0 + sy1, x0 + sx2, y0 + sy2),
                    mask_u8=mc.astype(np.uint8),
                    features=feats,
                    v_zero=float(v_zero),
                    is_healed=is_healed,
                    is_rescued=is_rescued,
                    v_hat=float(v_zero),
                )
            )

        out_by_tau[tkey] = cand_list
    return out_by_tau


def _fit_ridge_and_logistic(
    X_tr: np.ndarray,
    y_iou: np.ndarray,
    ridge_alpha: float = 40.0,
    logit_lam: float = 5.0,
) -> dict[str, np.ndarray]:
    """Fit strongly regularized standardized Ridge + L2 Newton Logistic in pure NumPy (< 2 ms)."""
    mu_x = X_tr.mean(axis=0)
    sd_x = np.maximum(X_tr.std(axis=0), 1e-4)
    Xs = np.clip((X_tr - mu_x) / sd_x, -5.0, 5.0).astype(np.float64)
    n, d = Xs.shape
    Xb = np.column_stack([np.ones(n, dtype=np.float64), Xs])

    reg = np.eye(d + 1, dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0
    w_ridge = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y_iou.astype(np.float64))

    y_bin = (y_iou > 0.0).astype(np.float64)
    w_log = np.zeros(d + 1, dtype=np.float64)
    reg_log = np.eye(d + 1, dtype=np.float64) * logit_lam
    reg_log[0, 0] = 1e-4
    for _ in range(12):
        z = np.clip(Xb @ w_log, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-6)
        grad = Xb.T @ (p - y_bin) + reg_log @ w_log
        hess = (Xb.T * w[None, :]) @ Xb + reg_log
        step = np.linalg.solve(hess, grad)
        w_log -= step
        if float(np.max(np.abs(step))) < 1e-5:
            break

    return {"mu": mu_x, "sd": sd_x, "w_ridge": w_ridge, "w_log": w_log}


def _predict_metascorer(models: list[dict[str, np.ndarray]], feats: np.ndarray) -> float:
    acc = 0.0
    for m in models:
        xs = np.clip((feats.astype(np.float64) - m["mu"]) / m["sd"], -5.0, 5.0)
        xb = np.concatenate([[1.0], xs])
        y_r = float(np.clip(xb @ m["w_ridge"], 0.0, 1.0))
        z_l = float(np.clip(xb @ m["w_log"], -20.0, 20.0))
        p_l = 1.0 / (1.0 + math.exp(-z_l))
        acc += 0.65 * y_r + 0.35 * (0.72 * p_l)
    return acc / float(max(1, len(models)))


def _predecode_val_gt(
    val_gt: dict[str, dict[str, list[dict]]],
) -> dict[str, list[list[tuple[tuple[int, int, int, int], np.ndarray, int]]]]:
    """Decode each annotator RLE once into compact bbox crops for fast exact IoU matching."""
    out: dict[str, list[list[tuple[tuple[int, int, int, int], np.ndarray, int]]]] = {}
    for stem, by_ann in val_gt.items():
        ann_lists = []
        for rles in by_ann.values():
            items = []
            for r in rles:
                bb = mu.toBbox(r)
                x1 = max(0, int(math.floor(bb[0])))
                y1 = max(0, int(math.floor(bb[1])))
                x2 = min(IMG_SIZE, int(math.ceil(bb[0] + bb[2])))
                y2 = min(IMG_SIZE, int(math.ceil(bb[1] + bb[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                mc = mu.decode(r)[y1:y2, x1:x2] > 0
                a = int(mc.sum())
                if a > 0:
                    items.append(((x1, y1, x2, y2), mc, a))
            ann_lists.append(items)
        out[stem] = ann_lists
    return out


def _fast_stem_counts(
    gt_ann_lists: list[list[tuple[tuple[int, int, int, int], np.ndarray, int]]],
    preds: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
) -> dict[str, Any]:
    tot = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "split": 0, "merge": 0}
    np_len = len(preds)
    for g_list in gt_ann_lists:
        ng_len = len(g_list)
        if np_len == 0 or ng_len == 0:
            tot["fp"] += np_len
            tot["fn"] += ng_len
            continue
        iou = np.zeros((np_len, ng_len), dtype=np.float64)
        inter_m = np.zeros((np_len, ng_len), dtype=np.float64)
        ap = np.array([p[2] for p in preds], dtype=np.float64)
        ag = np.array([g[2] for g in g_list], dtype=np.float64)
        for pi, ((px1, py1, px2, py2), pm, pa) in enumerate(preds):
            for gi, ((gx1, gy1, gx2, gy2), gm, ga) in enumerate(g_list):
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = float(
                    np.logical_and(
                        pm[iy1 - py1 : iy2 - py1, ix1 - px1 : ix2 - px1],
                        gm[iy1 - gy1 : iy2 - gy1, ix1 - gx1 : ix2 - gx1],
                    ).sum()
                )
                if inter > 0.0:
                    inter_m[pi, gi] = inter
                    iou[pi, gi] = inter / max(1.0, float(pa + ga) - inter)

        used_p, used_g, iou_sum = set(), set(), 0.0
        for p_idx, g_idx in sorted(zip(*np.where(iou > 0.50)), key=lambda pg: -iou[pg]):
            if p_idx not in used_p and g_idx not in used_g:
                used_p.add(p_idx)
                used_g.add(g_idx)
                iou_sum += float(iou[p_idx, g_idx])
        part_of = inter_m >= 0.5 * ap[:, None]
        covers = inter_m >= 0.5 * ag[None, :]
        tot["tp"] += len(used_p)
        tot["fp"] += np_len - len(used_p)
        tot["fn"] += ng_len - len(used_g)
        tot["iou_sum"] += iou_sum
        tot["split"] += int((part_of.sum(axis=0) > 1).sum())
        tot["merge"] += int((covers.sum(axis=1) > 1).sum())
    return tot


def _compose_disjoint_crops(
    cands: list[HybridCandidate],
    scores: list[float],
    keep_thresh: float,
    min_area: int,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, int]]:
    """Rank candidates by descending utility score and resolve strict pairwise-disjoint crops."""
    selected = [
        (c, sc)
        for c, sc in zip(cands, scores)
        if sc >= keep_thresh and int(c.mask_u8.sum()) >= min_area
    ]
    if not selected:
        return []
    selected.sort(key=lambda item: -item[1])

    ux0 = min(c.bbox[0] for c, _ in selected)
    uy0 = min(c.bbox[1] for c, _ in selected)
    ux1 = max(c.bbox[2] for c, _ in selected)
    uy1 = max(c.bbox[3] for c, _ in selected)
    out: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []
    try:
        for idx, (c, sc) in enumerate(selected):
            bx1, by1, bx2, by2 = c.bbox
            ov = _OWNER_BUF[by1:by2, bx1:bx2]
            sv = _SCORE_BUF[by1:by2, bx1:bx2]
            mb = c.mask_u8 > 0
            # Higher-ranked candidate already claimed > 45% of this candidate -> suppress duplicate
            free_m = mb & (ov < 0)
            if int(free_m.sum()) < min_area or float(free_m.sum()) < 0.55 * float(mb.sum()):
                continue
            ov[free_m] = idx
            sv[free_m] = float(sc)

        for idx, (c, _sc) in enumerate(selected):
            bx1, by1, bx2, by2 = c.bbox
            inst_m = (_OWNER_BUF[by1:by2, bx1:bx2] == idx)
            area_c = int(inst_m.sum())
            if area_c < min_area:
                continue
            out.append((c.bbox, inst_m.copy(), area_c))
    finally:
        _OWNER_BUF[uy0:uy1, ux0:ux1] = -1
        _SCORE_BUF[uy0:uy1, ux0:ux1] = 0.0
    return out


def encode_crops_to_rles(crops: list[tuple[tuple[int, int, int, int], np.ndarray, int]]) -> list[dict]:
    rles: list[dict] = []
    for (bx1, by1, bx2, by2), inst_m, _area in crops:
        try:
            _TMP_MASK[by1:by2, bx1:bx2] = inst_m.astype(np.uint8)
            counts = mu.encode(_TMP_MASK)["counts"].decode()
            rles.append(referee_metric.rle(counts))
        finally:
            _TMP_MASK[by1:by2, bx1:bx2] = 0
    return rles


def _compute_empirical_targets_for_stem(
    cands: list[HybridCandidate],
    gt_ann_lists: list[list[tuple[tuple[int, int, int, int], np.ndarray, int]]],
) -> np.ndarray:
    """Compute empirical expected PQ numerator contribution y_c in [0, 1] per candidate."""
    if not cands:
        return np.zeros((0,), dtype=np.float32)
    a_s = max(1, len(gt_ann_lists))
    order = sorted(range(len(cands)), key=lambda i: -cands[i].v_zero)
    y = np.zeros((len(cands),), dtype=np.float32)
    for ann_g in gt_ann_lists:
        used_g: set[int] = set()
        for ci in order:
            px1, py1, px2, py2 = cands[ci].bbox
            pm = cands[ci].mask_u8 > 0
            pa = int(pm.sum())
            best_gi, best_iou = -1, 0.50
            for gi, ((gx1, gy1, gx2, gy2), gm, ga) in enumerate(ann_g):
                if gi in used_g:
                    continue
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = float(
                    np.logical_and(
                        pm[iy1 - py1 : iy2 - py1, ix1 - px1 : ix2 - px1],
                        gm[iy1 - gy1 : iy2 - gy1, ix1 - gx1 : ix2 - gx1],
                    ).sum()
                )
                if inter <= 0.0:
                    continue
                iou_v = inter / max(1.0, float(pa + ga) - inter)
                if iou_v > best_iou:
                    best_iou = iou_v
                    best_gi = gi
            if best_gi >= 0:
                used_g.add(best_gi)
                y[ci] += float(best_iou) / float(a_s)
    return y


def _stem_day(stem: str) -> float:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        dt = datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return (dt - epoch).total_seconds() / 86400.0
    except ValueError:
        return 15000.0


def _assign_solar_rotation_subfolds(stems: list[str], n_subfolds: int = 5) -> dict[str, int]:
    """Group stems by 27-day solar rotation buckets, then partition evenly into 5 chronological subfolds."""
    ordered = sorted(stems, key=lambda s: (_stem_day(s), s))
    buckets: dict[int, list[str]] = {}
    for s in ordered:
        b = int(_stem_day(s) // 27.0)
        buckets.setdefault(b, []).append(s)
    subfold_of: dict[str, int] = {}
    if len(buckets) >= n_subfolds:
        rot_keys = sorted(buckets.keys())
        target = len(ordered) / float(n_subfolds)
        cur_sf = 0
        cnt = 0
        for bk in rot_keys:
            for s in buckets[bk]:
                subfold_of[s] = min(n_subfolds - 1, cur_sf)
                cnt += 1
            if cnt >= target * (cur_sf + 1) and cur_sf < n_subfolds - 1:
                cur_sf += 1
    else:
        for i, s in enumerate(ordered):
            subfold_of[s] = min(n_subfolds - 1, (i * n_subfolds) // max(1, len(ordered)))
    return subfold_of


def crossfit_hybrid_select(
    val_stems: list[str],
    cands_by_stem_tau: dict[str, dict[str, list[HybridCandidate]]],
    val_gt: dict[str, dict[str, list[dict]]],
    tau_hyst_grid: tuple[tuple[float, float], ...] = TAU_HYST_GRID,
) -> tuple[dict[str, Any], float, float, list[dict[str, Any]], dict[str, list[dict]], list[dict[str, np.ndarray]]]:
    """Fit 5-subfold chronological solar-rotation Ridge+Logistic expected-PQ scorer + evaluate zero-fit baseline."""
    decoded_gt = _predecode_val_gt(val_gt)
    ordered = sorted(val_stems, key=lambda s: (_stem_day(s), s))
    subfold_of = _assign_solar_rotation_subfolds(ordered, n_subfolds=5)
    unique_subfolds = sorted(set(subfold_of.values()))
    tau_keys = [tau_pair_key(ts, dh) for ts, dh in tau_hyst_grid]

    # 1. Zero-fit analytical baseline (0 fitted parameters!)
    zero_key = "0.50_0.06" if "0.50_0.06" in tau_keys else tau_keys[len(tau_keys) // 2]
    zf_counts: list[dict[str, Any]] = []
    for s in ordered:
        cs = cands_by_stem_tau.get(s, {}).get(zero_key, [])
        crops = _compose_disjoint_crops(cs, [c.v_zero for c in cs], keep_thresh=0.24, min_area=32)
        zf_counts.append(_fast_stem_counts(decoded_gt.get(s, []), crops))
    zero_fit_pq = float(referee_metric.pq(referee_metric.total(zf_counts)))

    keep_grid = [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.31]
    area_grid = [24, 32, 44]

    # Precompute empirical y_c targets for every stem & hysteresis key
    emp_targets: dict[tuple[str, str], np.ndarray] = {}
    for s in ordered:
        for tkey in tau_keys:
            cs = cands_by_stem_tau.get(s, {}).get(tkey, [])
            emp_targets[(s, tkey)] = _compute_empirical_targets_for_stem(cs, decoded_gt.get(s, []))

    oof_rles: dict[str, list[dict]] = {}
    sweep_rows: list[dict[str, Any]] = []
    subfold_summaries: list[dict[str, Any]] = []
    fold_models_for_test: list[dict[str, np.ndarray]] = []
    chosen_taus: list[str] = []
    chosen_keeps: list[float] = []
    chosen_areas: list[int] = []

    if len(ordered) >= 10 and len(unique_subfolds) >= 2:
        oof_counts: list[dict[str, Any]] = []
        for sf in unique_subfolds:
            fit_stems = [s for s in ordered if subfold_of[s] != sf]
            eval_stems = [s for s in ordered if subfold_of[s] == sf]
            best_sf: tuple[float, str, float, int, dict[str, np.ndarray]] | None = None

            for tkey in tau_keys:
                X_list, y_list = [], []
                for s in fit_stems:
                    cs = cands_by_stem_tau.get(s, {}).get(tkey, [])
                    if cs:
                        X_list.append(np.stack([c.features for c in cs], axis=0))
                        y_list.append(emp_targets[(s, tkey)])
                if not X_list:
                    continue
                model_sf = _fit_ridge_and_logistic(np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0))

                stem_scores_fit = {
                    s: [_predict_metascorer([model_sf], c.features) for c in cands_by_stem_tau.get(s, {}).get(tkey, [])]
                    for s in fit_stems
                }
                ts_val = float(tkey.split("_")[0])
                dh_val = float(tkey.split("_")[1])
                for ma in area_grid:
                    for kt in keep_grid:
                        cnts = []
                        for s in fit_stems:
                            cs = cands_by_stem_tau.get(s, {}).get(tkey, [])
                            crops = _compose_disjoint_crops(cs, stem_scores_fit[s], keep_thresh=kt, min_area=ma)
                            cnts.append(_fast_stem_counts(decoded_gt.get(s, []), crops))
                        pq_fit = float(referee_metric.pq(referee_metric.total(cnts)))
                        if sf == unique_subfolds[0]:
                            sweep_rows.append({
                                "tau_key": tkey,
                                "tau_fg": ts_val,
                                "delta_hyst": dh_val,
                                "keep_thresh": float(kt),
                                "min_area": int(ma),
                                "pq": pq_fit,
                            })
                        if best_sf is None or pq_fit > best_sf[0]:
                            best_sf = (pq_fit, tkey, float(kt), int(ma), model_sf)

            assert best_sf is not None
            pq_fit, tkey_k, kt_k, ma_k, model_k = best_sf
            fold_models_for_test.append(model_k)
            chosen_taus.append(tkey_k)
            chosen_keeps.append(kt_k)
            chosen_areas.append(ma_k)
            subfold_summaries.append({
                "subfold": int(sf),
                "train_pq": float(pq_fit),
                "tau_key": tkey_k,
                "tau_fg": float(tkey_k.split("_")[0]),
                "delta_hyst": float(tkey_k.split("_")[1]),
                "keep_thresh": float(kt_k),
                "min_area": int(ma_k),
            })

            for s in eval_stems:
                cs = cands_by_stem_tau.get(s, {}).get(tkey_k, [])
                scs = [_predict_metascorer([model_k], c.features) for c in cs]
                for c, sc in zip(cs, scs):
                    c.v_hat = sc
                crops = _compose_disjoint_crops(cs, scs, keep_thresh=kt_k, min_area=ma_k)
                oof_counts.append(_fast_stem_counts(decoded_gt.get(s, []), crops))
                oof_rles[s] = encode_crops_to_rles(crops)

        oof_pq = float(referee_metric.pq(referee_metric.total(oof_counts)))
    else:
        tkey_k = zero_key
        kt_k = 0.22
        ma_k = 24
        X_list = [np.stack([c.features for c in cands_by_stem_tau.get(s, {}).get(tkey_k, [])], axis=0) for s in ordered if cands_by_stem_tau.get(s, {}).get(tkey_k, [])]
        y_list = [emp_targets[(s, tkey_k)] for s in ordered if len(emp_targets[(s, tkey_k)]) > 0]
        if X_list:
            m0 = _fit_ridge_and_logistic(np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0))
            fold_models_for_test.append(m0)
        for s in ordered:
            cs = cands_by_stem_tau.get(s, {}).get(tkey_k, [])
            scs = [_predict_metascorer(fold_models_for_test, c.features) for c in cs] if fold_models_for_test else [c.v_zero for c in cs]
            crops = _compose_disjoint_crops(cs, scs, keep_thresh=kt_k, min_area=ma_k)
            oof_rles[s] = encode_crops_to_rles(crops)
        chosen_taus, chosen_keeps, chosen_areas = [tkey_k], [kt_k], [ma_k]
        oof_pq = zero_fit_pq

    modal_key = max(set(chosen_taus), key=chosen_taus.count)
    ts_m, dh_m = [float(x) for x in modal_key.split("_")]
    med_keep = float(np.median(chosen_keeps))
    med_area = int(round(float(np.median(chosen_areas))))
    best_params = {
        "tau_key": modal_key,
        "tau_seed": float(ts_m),
        "delta_hyst": float(dh_m),
        "tau_fg": float(ts_m),
        "keep_thresh": float(med_keep),
        "min_area": int(med_area),
        "zero_fit_pq": float(zero_fit_pq),
        "oof_pq": float(oof_pq),
        "subfolds": subfold_summaries,
    }
    return best_params, oof_pq, zero_fit_pq, sweep_rows, oof_rles, fold_models_for_test


def save_metascorer_json(path: Path, best_params: dict[str, Any], models: list[dict[str, np.ndarray]]) -> None:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "best_params": best_params,
        "models": [
            {
                "mu": m["mu"].astype(float).tolist(),
                "sd": m["sd"].astype(float).tolist(),
                "w_ridge": m["w_ridge"].astype(float).tolist(),
                "w_log": m["w_log"].astype(float).tolist(),
            }
            for m in models
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metascorer_json(path: Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = [
        {
            "mu": np.asarray(m["mu"], dtype=np.float64),
            "sd": np.asarray(m["sd"], dtype=np.float64),
            "w_ridge": np.asarray(m["w_ridge"], dtype=np.float64),
            "w_log": np.asarray(m["w_log"], dtype=np.float64),
        }
        for m in data.get("models", [])
    ]
    return data["best_params"], models


def decode_test_stem_rles(
    cands: list[HybridCandidate],
    models: list[dict[str, np.ndarray]],
    keep_thresh: float,
    min_area: int,
) -> list[dict]:
    if not cands:
        return []
    scores = [_predict_metascorer(models, c.features) for c in cands] if models else [c.v_zero for c in cands]
    crops = _compose_disjoint_crops(cands, scores, keep_thresh=keep_thresh, min_area=min_area)
    return encode_crops_to_rles(crops)


def count_overlapping_stems(rles_by_stem: dict[str, list[dict]]) -> int:
    bad = 0
    for rles in rles_by_stem.values():
        if len(rles) < 2:
            continue
        inter = np.asarray(mu.iou(rles, rles, [1] * len(rles)), dtype=np.float64)
        np.fill_diagonal(inter, 0.0)
        if np.any(inter > 0.0):
            bad += 1
    return bad


def write_predictions_csv(rles_by_stem: dict[str, list[dict]], stems: list[str], out_csv: str | Path) -> Path:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(stems):
            idx_out = 0
            for r in rles_by_stem.get(stem, []):
                if not r or int(mu.area(r)) <= 0:
                    continue
                counts = r["counts"].decode() if isinstance(r["counts"], (bytes, bytearray)) else str(r["counts"])
                if not counts:
                    continue
                w.writerow([f"{stem}_{idx_out}", counts])
                idx_out += 1
    return out_path


def _style_dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#f8fafc")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.grid(color="#1e293b", alpha=0.65, linewidth=0.6)


def render_dark_ui_plots(
    out_dir: Path,
    val_stems: list[str],
    cache_dir: Path,
    geom_map: dict[str, dict[str, float]],
    s1_history: list[dict[str, float]],
    native_history: list[dict[str, float]],
    cands_by_stem_tau: dict[str, dict[str, list[HybridCandidate]]],
    oof_rles: dict[str, list[dict]],
    raw_val_gt: dict[str, dict[str, list[dict]]],
    cons_val_gt: dict[str, dict[str, list[dict]]],
    sweep_rows: list[dict[str, Any]],
    summary_metrics: dict[str, Any],
    raw_res: dict[str, Any],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tau_show = str(summary_metrics["best_params"]["tau_key"])

    # Plot 1: Native 5-Ch Physics & Unbiased Zero-Level Consensus Audit
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#0b1020")
    fig.suptitle("029 Native 1:1 Co-DeepNet — Active-Disk Slice, Multi-Scale Barb Physics & Hysteresis", color="#f8fafc", fontsize=13, fontweight="bold")
    if val_stems:
        pick = val_stems[len(val_stems) // 2]
        hw3 = load_cached_physics_u8(cache_dir, pick)
        g = geom_map[pick]
        x0, y0 = int(np.clip(round(g["cx"] - 0.5 * SLICE_SIZE), 0, IMG_SIZE - SLICE_SIZE)), int(np.clip(round(g["cy"] - 0.5 * SLICE_SIZE), 0, IMG_SIZE - SLICE_SIZE))
        sl = hw3[y0 : y0 + SLICE_SIZE, x0 : x0 + SLICE_SIZE]
        ds = cv2.resize(sl, (640, 640), interpolation=cv2.INTER_AREA)
        axes[0, 0].imshow(ds[..., 0], cmap="gray")
        axes[0, 0].set_title(f"{pick}: Ch0 I_flat (1888x1888 1:1 slice)")
        axes[0, 1].imshow(ds[..., 1], cmap="magma")
        axes[0, 1].set_title("Ch1 D_multi (Plage-Safe Dark Depth)")

        c_list = cons_val_gt.get(pick, {}).get("consensus", [])
        if c_list:
            bb = mu.toBbox(c_list[0])
            cx_z = int(np.clip(bb[0] + 0.5 * bb[2], 160, IMG_SIZE - 160))
            cy_z = int(np.clip(bb[1] + 0.5 * bb[3], 160, IMG_SIZE - 160))
        else:
            cx_z, cy_z = IMG_SIZE // 2, IMG_SIZE // 2
        zc = hw3[cy_z - 160 : cy_z + 160, cx_z - 160 : cx_z + 160]
        bg15 = cv2.boxFilter(zc[..., 0].astype(np.float32), -1, (15, 15))
        barb = np.clip((bg15 - zc[..., 0].astype(np.float32)) * (10.0 / 255.0), 0.0, 1.0)
        axes[1, 0].imshow(barb, cmap="inferno")
        axes[1, 0].set_title("Native 1:1 15x15 Fine Barb Channel (Zero Blurring)")
    else:
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_title("No validation stems")

    areas_val = [int(c.mask_u8.sum()) for s in val_stems for c in cands_by_stem_tau.get(s, {}).get(tau_show, [])]
    if areas_val:
        axes[1, 1].hist(np.log10(np.clip(areas_val, 10, 1e6)), bins=28, color="#38bdf8", alpha=0.85, edgecolor="#0b1020")
    axes[1, 1].set_title(f"Candidate log10(Area) at hyst={tau_show} (n={len(areas_val)})")
    axes[1, 1].set_xlabel("log10(Area px)")
    for ax in axes.flat:
        _style_dark_ax(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plots_dir / "01_v10_native_physics_and_hysteresis.png", dpi=150, facecolor="#0b1020")
    plt.close(fig)

    # Plot 2: Training Curves (Native Co-DeepNet 20 ep + Stage-1 Mask R-CNN 12 ep)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#0b1020")
    fig.suptitle("029 Multi-Fold Training Convergence (Last Fold Cosine Schedules)", color="#f8fafc", fontsize=13, fontweight="bold")
    if native_history:
        ep_n = [r["epoch"] for r in native_history]
        axes[0].plot(ep_n, [r["loss"] for r in native_history], "-o", color="#38bdf8", label="Total Loss")
        axes[0].plot(ep_n, [r["dice"] for r in native_history], "-s", color="#34d399", label="Soft-IoU Loss")
        axes[0].plot(ep_n, [r["bce"] for r in native_history], "-^", color="#f59e0b", label="Focal Loss (Thin x2)")
        axes[0].plot(ep_n, [r["cons"] for r in native_history], "--", color="#a78bfa", label="Peer MSE")
        axes[0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    axes[0].set_title("Native 1:1 Co-DeepNet (ConvNeXt-Small + StripBarbUNet)")
    axes[0].set_xlabel("Epoch")

    if s1_history:
        ep_s = [r["epoch"] for r in s1_history]
        axes[1].plot(ep_s, [r["loss"] for r in s1_history], "-o", color="#f43f5e", label="Total R-CNN Loss")
        axes[1].plot(ep_s, [r["bce"] for r in s1_history], "-s", color="#f59e0b", label="Box+Cls")
        axes[1].plot(ep_s, [r["dice"] for r in s1_history], "-^", color="#38bdf8", label="Mask")
        axes[1].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    axes[1].set_title("Stage-1 1408px Mask R-CNN v2 (Full 12-ep Cosine)")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        _style_dark_ax(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plots_dir / "02_v10_multifold_training_curves.png", dpi=150, facecolor="#0b1020")
    plt.close(fig)

    # Plot 3: 24-Feature Meta-Scorer Sweep, Utility Calibration & Per-Stem PQ
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#0b1020")
    fig.suptitle(
        f"029 24-d Calibration & PQ: pooled_pq={summary_metrics.get('self_val_pq', 0):.4f} | "
        f"fold0_pq={summary_metrics.get('fold0_val_pq', 0) or 0:.4f} | pure_v2={summary_metrics.get('pure_votes2_consensus_pq', 0):.4f}",
        color="#f8fafc", fontsize=12, fontweight="bold",
    )
    if sweep_rows:
        palette = ["#38bdf8", "#34d399", "#f59e0b", "#f43f5e", "#a78bfa", "#ec4899", "#22d3ee", "#eab308"]
        for tkey_s, col in zip(sorted({str(r["tau_key"]) for r in sweep_rows}), palette):
            sub_r = [r for r in sweep_rows if str(r["tau_key"]) == tkey_s and r["min_area"] == int(summary_metrics["best_params"]["min_area"])]
            if sub_r:
                sub_r.sort(key=lambda r: r["keep_thresh"])
                axes[0].plot([r["keep_thresh"] for r in sub_r], [r["pq"] for r in sub_r], "-o", color=col, label=tkey_s)
        axes[0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=7, ncol=2)
    axes[0].set_title("Subfold-0 PQ Sweep (keep_thresh vs tau_hyst)")
    axes[0].set_xlabel("keep_thresh (Expected PQ Gain)")

    all_c = [c for s in val_stems for c in cands_by_stem_tau.get(s, {}).get(tau_show, [])]
    if all_c:
        v0 = np.array([c.v_zero for c in all_c])
        vh = np.array([c.v_hat for c in all_c])
        healed = np.array([c.is_healed for c in all_c])
        axes[1].scatter(v0[~healed][:800], vh[~healed][:800], s=10, alpha=0.55, color="#38bdf8", label="Single CC")
        if healed.any():
            axes[1].scatter(v0[healed][:400], vh[healed][:400], s=16, alpha=0.75, color="#34d399", label="Waist-Healed")
        axes[1].axvline(float(summary_metrics["best_params"]["keep_thresh"]), color="#f43f5e", linestyle="--", label="keep_thresh")
        axes[1].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    axes[1].set_title("Zero-Fit Utility vs Out-of-Subfold v_hat (24-d)")
    axes[1].set_xlabel("v_zero")
    axes[1].set_ylabel("v_hat")

    per_s = raw_res.get("per_stem", {})
    if per_s:
        pqs = sorted([referee_metric.pq(c) for c in per_s.values()])
        axes[2].hist(pqs, bins=24, color="#a78bfa", alpha=0.85, edgecolor="#0b1020")
        axes[2].axvline(float(np.mean(pqs)), color="#34d399", linestyle="--", label=f"Mean={np.mean(pqs):.3f}")
        axes[2].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    axes[2].set_title(f"Per-Stem Validation PQ (n={len(per_s)})")
    axes[2].set_xlabel("Stem PQ")

    for ax in axes:
        _style_dark_ax(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(plots_dir / "03_v10_metascorer_24d_calibration.png", dpi=150, facecolor="#0b1020")
    plt.close(fig)

    # Plot 4: Native 1:1 Overlays on Diverse Validation Frames
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), facecolor="#0b1020")
    fig.suptitle("029 3-Fold Native 1:1 Predictions (Cyan) vs Consensus/GT (Amber)", color="#f8fafc", fontsize=13, fontweight="bold")
    scored_stems = sorted(per_s.items(), key=lambda kv: referee_metric.pq(kv[1])) if per_s else []
    if len(scored_stems) >= 4:
        picks = [scored_stems[-1][0], scored_stems[len(scored_stems) * 2 // 3][0], scored_stems[len(scored_stems) // 3][0], scored_stems[0][0]]
    else:
        picks = val_stems[:4]

    for ax_i, ax in enumerate(axes.flat):
        if ax_i >= len(picks):
            ax.set_visible(False)
            continue
        s = picks[ax_i]
        hw3 = load_cached_physics_u8(cache_dir, s)
        base = cv2.resize(hw3[..., 0], (640, 640), interpolation=cv2.INTER_AREA)
        rgb = np.stack([base, base, base], axis=-1)
        gt_rles = cons_val_gt.get(s, {}).get("consensus") or next(iter(raw_val_gt.get(s, {}).values()), [])
        pr_rles = oof_rles.get(s, [])
        gt_u = np.zeros((640, 640), dtype=bool)
        for r in gt_rles:
            gt_u |= cv2.resize(mu.decode(r), (640, 640), interpolation=cv2.INTER_NEAREST) > 0
        pr_u = np.zeros((640, 640), dtype=bool)
        for r in pr_rles:
            pr_u |= cv2.resize(mu.decode(r), (640, 640), interpolation=cv2.INTER_NEAREST) > 0
        gt_edge = gt_u & (~(cv2.erode(gt_u.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
        pr_edge = pr_u & (~(cv2.erode(pr_u.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
        rgb[gt_edge] = [245, 158, 11]
        rgb[pr_edge] = [56, 189, 248]
        rgb[gt_edge & pr_edge] = [52, 211, 153]
        ax.imshow(rgb)
        spq = referee_metric.pq(per_s[s]) if s in per_s else 0.0
        ax.set_title(f"{s} | PQ={spq:.3f} | pred={len(pr_rles)} vs gt={len(gt_rles)}")
        _style_dark_ax(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plots_dir / "04_v10_native_1to1_overlays.png", dpi=150, facecolor="#0b1020")
    plt.close(fig)
