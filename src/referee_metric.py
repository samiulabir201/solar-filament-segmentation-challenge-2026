"""The referee. Panoptic Quality on COCO RLE masks (true positive = IoU > 0.5, one-to-one), the fragmentation counts
the judges also read (one ground-truth filament split over several predictions, one prediction merging several
filaments), and a paired bootstrap over images that decides whether a change beat its parent.
ponytail: PQ follows the competition's formula; the research stage's metric_verify task checks it against the
organisers' self-evaluation notebook and reports any difference before decisions rely on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mu

SIZE = 2048
IOU_TP = 0.5
BOOTSTRAP = 2000


def rle(counts: str) -> dict:
    return {"size": [SIZE, SIZE], "counts": counts.encode() if isinstance(counts, str) else counts}


def encode(mask: np.ndarray) -> str:
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode()


def image_counts(gt: list[dict], pred: list[dict]) -> dict:
    """TP/FP/FN, the IoU sum over TPs, and the split/merge counts for one image (all masks as RLE dicts)."""
    if not gt or not pred:
        return {"tp": 0, "fp": len(pred), "fn": len(gt), "iou_sum": 0.0, "split": 0, "merge": 0}
    iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)))  # rows: predictions, cols: ground truth
    a_p, a_g = mu.area(pred).astype(float), mu.area(gt).astype(float)
    inter = iou * (a_p[:, None] + a_g[None, :]) / (1 + iou)
    used_p, used_g, iou_sum = set(), set(), 0.0
    for p, g in sorted(zip(*np.where(iou > IOU_TP)), key=lambda pg: -iou[pg]):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            iou_sum += iou[p, g]
    part_of = inter >= 0.5 * a_p[:, None]   # most of prediction p lies inside filament g
    covers = inter >= 0.5 * a_g[None, :]    # most of filament g lies inside prediction p
    return {"tp": len(used_p), "fp": len(pred) - len(used_p), "fn": len(gt) - len(used_g), "iou_sum": float(iou_sum),
            "split": int((part_of.sum(axis=0) > 1).sum()), "merge": int((covers.sum(axis=1) > 1).sum())}


def image_objects(stem: str, gt: list[dict], pred: list[dict]) -> list[dict]:
    """Per-filament rows for error analysis: every ground-truth and predicted mask with its area, best IoU against the
    other side, and whether that IoU makes a true positive."""
    a_g = mu.area(gt).astype(float) if gt else np.zeros(0)
    a_p = mu.area(pred).astype(float) if pred else np.zeros(0)
    iou = np.asarray(mu.iou(pred, gt, [0] * len(gt))) if gt and pred else np.zeros((len(pred), len(gt)))
    rows = [{"stem": stem, "side": "gt", "area": float(a_g[j]), "best_iou": float(iou[:, j].max()) if len(pred) else 0.0}
            for j in range(len(gt))]
    rows += [{"stem": stem, "side": "pred", "area": float(a_p[i]), "best_iou": float(iou[i].max()) if len(gt) else 0.0}
             for i in range(len(pred))]
    for r in rows:
        r["matched"] = r["best_iou"] > IOU_TP
    return rows


def pq(c: dict) -> float:
    den = c["tp"] + 0.5 * c["fp"] + 0.5 * c["fn"]
    return c["iou_sum"] / den if den else 1.0


def total(rows: list[dict]) -> dict:
    keys = ("tp", "fp", "fn", "iou_sum", "split", "merge")
    return {k: sum(r[k] for r in rows) for k in keys}


def predictions(csv) -> dict[str, list[dict]]:
    """Submission CSV (filament_id, segmentation_rle) -> {image stem: [RLE]}."""
    df = pd.read_csv(csv, dtype=str).fillna("")
    out: dict[str, list[dict]] = {}
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        if counts:
            out.setdefault(fid.rsplit("_", 1)[0], []).append(rle(counts))
    return out


def score(gt: dict[str, dict[str, list[dict]]], pred: dict[str, list[dict]]) -> dict:
    """gt: {image stem: {annotator image id: [RLE]}} for the validation stems. Each annotator's set is scored as its
    own image (the organisers' rule), against the one prediction set of that stem."""
    per_stem, objects = {}, []
    for stem, by_annotator in gt.items():
        per_stem[stem] = total([image_counts(g, pred.get(stem, [])) for g in by_annotator.values()])
        first = next(iter(by_annotator.values()), [])
        objects += image_objects(stem, first, pred.get(stem, []))
    agg = total(list(per_stem.values()))
    return {"pq": pq(agg), **agg, "images": len(per_stem), "per_stem": per_stem, "objects": objects,
            "mean_image_pq": float(np.mean([pq(c) for c in per_stem.values()])) if per_stem else 0.0}


def ci(res: dict, seed: int = 0, level: float = 0.9) -> tuple[float, float]:
    """Bootstrap interval of the aggregate PQ over image stems."""
    keys = ("tp", "fp", "fn", "iou_sum")
    m = np.array([[c[k] for k in keys] for c in res["per_stem"].values()], float)
    if not len(m):
        return 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, len(m), (BOOTSTRAP, len(m)))
    s = m[idx].sum(1)
    den = s[:, 0] + 0.5 * s[:, 1] + 0.5 * s[:, 2]
    v = np.where(den > 0, s[:, 3] / np.where(den > 0, den, 1), 1.0)
    return float(np.quantile(v, (1 - level) / 2)), float(np.quantile(v, 1 - (1 - level) / 2))


def parity_cases(gt: dict, seed: int = 0) -> list[tuple[str, dict]]:
    """Fixed prediction sets built from real ground truth: perfect, missing half, false extras, split, merged,
    dilated, empty. The official scorer and the referee must agree on all of them."""
    rng = np.random.default_rng(seed)
    first = {s: next(iter(v.values()), []) for s, v in gt.items()}

    def dec(r):
        return mu.decode(r).astype(bool)

    def enc(m):
        return mu.encode(np.asfortranarray(m.astype(np.uint8)))

    stems = sorted(first)
    cases = [("perfect", first), ("empty", {s: [] for s in first}),
             ("half_missing", {s: v[: len(v) // 2] for s, v in first.items()}),
             # uneven damage separates pooled from per-image aggregation and shows how empty images count
             ("uneven", {s: (first[s] if i % 2 else []) for i, s in enumerate(stems)}),
             ("one_image_only", {s: (first[s] if i == 0 else []) for i, s in enumerate(stems)})]
    extra, split, merged, dilated = {}, {}, {}, {}
    for s, v in first.items():
        extra[s] = v + [enc(np.roll(dec(r), 300, axis=0)) for r in v[:1]]
        sp = []
        for r in v:
            m = dec(r)
            ys, xs = np.nonzero(m)
            if len(xs) > 20:
                cut = int(np.median(xs))
                sp += [enc(m & (np.arange(m.shape[1])[None, :] < cut)), enc(m & (np.arange(m.shape[1])[None, :] >= cut))]
            else:
                sp.append(r)
        split[s] = sp
        merged[s] = [mu.merge(v)] if len(v) > 1 else v
        dilated[s] = [enc(np.maximum(dec(r), np.roll(dec(r), int(rng.integers(2, 6)), axis=1))) for r in v]
    return cases + [("false_extras", extra), ("split", split), ("merged", merged), ("dilated", dilated)]


def parity(official_score, gt: dict, tol: float = 1e-6) -> dict:
    """Compare the official scorer with this referee on the fixed cases -> {matches, cases: {name: (ours, theirs)}}."""
    out, ok = {}, True
    for name, pred in parity_cases(gt):
        ours, theirs = score(gt, pred)["pq"], float(official_score(gt, pred))
        out[name] = (round(ours, 6), round(theirs, 6))
        ok &= abs(ours - theirs) <= tol
    return {"matches": bool(ok), "cases": out}


def paired(child: dict, parent: dict, seed: int = 0) -> dict:
    """Bootstrap over shared stems of PQ(child) - PQ(parent) -> gain, its 90% interval, and p_better."""
    stems = sorted(set(child["per_stem"]) & set(parent["per_stem"]))
    if not stems:
        return {"gain": 0.0, "lo": 0.0, "hi": 0.0, "p_better": 0.0, "stems": 0}
    keys = ("tp", "fp", "fn", "iou_sum")
    c = np.array([[child["per_stem"][s][k] for k in keys] for s in stems], float)
    p = np.array([[parent["per_stem"][s][k] for k in keys] for s in stems], float)

    def pq_of(m):
        den = m[..., 0] + 0.5 * m[..., 1] + 0.5 * m[..., 2]
        return np.where(den > 0, m[..., 3] / np.where(den > 0, den, 1), 1.0)

    idx = np.random.default_rng(seed).integers(0, len(stems), (BOOTSTRAP, len(stems)))
    diff = pq_of(c[idx].sum(1)) - pq_of(p[idx].sum(1))
    return {"gain": float(pq_of(c.sum(0)) - pq_of(p.sum(0))), "lo": float(np.quantile(diff, 0.05)),
            "hi": float(np.quantile(diff, 0.95)), "p_better": float((diff > 0).mean()), "stems": len(stems)}
