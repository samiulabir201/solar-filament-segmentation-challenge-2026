"""Run script for Node 29 / Job 029: Production 3-Fold Hybrid Native 1:1 Co-DeepNet + Geodesic Hysteresis."""
from __future__ import annotations

import gc
import json
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

JOB_DIR = Path(__file__).resolve().parent
if str(JOB_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_DIR))

from codeepnet_native import predict_native_full_disk, train_native_codeepnet
from decode_and_eval import (
    HybridCandidate,
    TAU_HYST_GRID,
    count_overlapping_stems,
    crossfit_hybrid_select,
    decode_test_stem_rles,
    extract_stem_candidates_multitau,
    render_dark_ui_plots,
    save_metascorer_json,
    write_predictions_csv,
)
from physics_v2 import (
    IMG_SIZE,
    SLICE_SIZE,
    find_image_path,
    load_cached_physics_u8,
    preload_ram_physics,
    prepare_physics_cache,
)
import referee_metric
from stage1_detector import (
    DET_SIZE,
    cluster_multiview_proposals,
    predict_stage1_proposals,
    predict_stage1_raw_candidates,
    train_stage1_detector,
)
from targets_v10 import build_stem_targets_v10, build_train_bitmasks

ROOT_REL = "MAGFiLO_1.0_Kaggle_2026"
GT_REL = f"{ROOT_REL}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_REL = f"{ROOT_REL}/train/train_images"
TEST_REL = f"{ROOT_REL}/test/test_images"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _resolve_paths() -> tuple[Path, Path, Path]:
    local_root = JOB_DIR.parents[2]
    data_dir = Path(os.environ.get("DATA_DIR", str(local_root / "data")))
    out_dir = Path(os.environ.get("OUT_DIR", str(JOB_DIR / "out")))

    folds_env = os.environ.get("FOLDS_CSV", "")
    if folds_env and Path(folds_env).exists():
        folds_csv = Path(folds_env)
    elif (JOB_DIR / "folds.csv").exists():
        folds_csv = JOB_DIR / "folds.csv"
    elif (local_root / "harness" / "folds.csv").exists():
        folds_csv = local_root / "harness" / "folds.csv"
    else:
        raise FileNotFoundError("Could not locate folds.csv")
    return data_dir, out_dir, folds_csv


def _parse_val_folds(raw: str) -> list[int] | None:
    txt = raw.strip().lower()
    if txt == "none":
        return None
    out = [int(tok.strip()) for tok in txt.split(",") if tok.strip()]
    return out or [0, 1, 2]


def _compute_sq_rq(res: dict[str, Any]) -> tuple[float, float]:
    tp = int(res.get("tp", 0))
    fp = int(res.get("fp", 0))
    fn = int(res.get("fn", 0))
    iou_sum = float(res.get("iou_sum", 0.0))
    sq = iou_sum / tp if tp > 0 else 0.0
    den = tp + 0.5 * fp + 0.5 * fn
    rq = tp / den if den > 0 else 1.0
    return float(sq), float(rq)


def _write_error_analysis(
    out_dir: Path,
    metrics: dict[str, Any],
    best_params: dict[str, Any],
    raw_res: dict[str, Any],
    pure_v2_res: dict[str, Any] | None,
) -> None:
    lines = [
        "# 029 Production 3-Fold Hybrid Native 1:1 Co-DeepNet — OOF Validation Error Analysis",
        "",
        "## Headline Metrics (5-Subfold Solar-Rotation OOF)",
        f"- **pooled self_val_pq ({metrics['n_val_stems']} stems)**: `{metrics['self_val_pq']:.6f}` (`SQ={metrics['sq']:.4f}`, `RQ={metrics['rq']:.4f}`, `TP={metrics['tp']}`, `FP={metrics['fp']}`, `FN={metrics['fn']}`, `split={metrics['split']}`, `merge={metrics['merge']}`)",
        f"- **fold0_val_pq ({metrics.get('n_fold0_stems', 0)} stems)**: `{metrics.get('fold0_val_pq', 0.0):.6f}` (`CI90={metrics.get('fold0_ci90', [0.0, 0.0])}`)",
        f"- **zero_fit_pq (0-param baseline)**: `{metrics['zero_fit_pq']:.6f}`",
        f"- **consensus_pq**: `{metrics['consensus_pq']:.6f}` (`clean_SQ={metrics['clean_sq']:.4f}`, `clean_RQ={metrics['clean_rq']:.4f}`)",
        f"- **pure_votes2_consensus_pq**: `{metrics['pure_votes2_consensus_pq']:.6f}` (`SQ={metrics['pure_votes2_sq']:.4f}`, `RQ={metrics['pure_votes2_rq']:.4f}`)",
        f"- **As1 / As2 / As3 PQ**: `{metrics.get('as1_pq', 0.0):.4f} / {metrics.get('as2_pq', 0.0):.4f} / {metrics.get('as3_pq', 0.0):.4f}`",
        f"- **overlap_stems (val / test)**: `{metrics['stems_with_overlap_val']} / {metrics['stems_with_overlap_test']}`",
        f"- **vault_read_count**: `{metrics['vault_read_count']}`",
        f"- **runtime (min)**: total `{metrics['train_minutes']:.2f}` (Stage-1 `{metrics['stage1_minutes']:.2f}`, Native Co-DeepNet `{metrics['codeepnet_minutes']:.2f}`)",
        "",
        "## Selected Postprocess Parameters",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
        "",
    ]
    per_stem = raw_res.get("per_stem", {})
    if per_stem:
        ranked = sorted(per_stem.items(), key=lambda kv: (referee_metric.pq(kv[1]), -(kv[1]["fp"] + kv[1]["fn"])))
        lines.append("## Lowest-PQ Validation Stems (Raw GT)")
        lines.append("| Stem | PQ | TP | FP | FN | Split | Merge |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s, c in ranked[:15]:
            lines.append(f"| {s} | {referee_metric.pq(c):.4f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['split']} | {c['merge']} |")
        lines.append("")
    if pure_v2_res and pure_v2_res.get("per_stem"):
        ranked_v2 = sorted(pure_v2_res["per_stem"].items(), key=lambda kv: referee_metric.pq(kv[1]))
        lines.append(f"## Pure Votes>=2 Stems Summary (n={len(ranked_v2)})")
        lines.append("| Stem | PQ | TP | FP | FN | Split | Merge |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s, c in ranked_v2[:12]:
            lines.append(f"| {s} | {referee_metric.pq(c):.4f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['split']} | {c['merge']} |")
        lines.append("")
    (out_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    seed = int(os.environ.get("SEED", "42"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    val_folds = _parse_val_folds(os.environ.get("VAL_FOLDS", "0,1,2"))
    _seed_everything(seed)

    data_dir, out_dir, folds_csv = _resolve_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    w_dir = out_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = data_dir / TRAIN_REL
    test_img_dir = data_dir / TEST_REL
    gt_json_path = data_dir / GT_REL
    if not train_img_dir.exists() or not gt_json_path.exists():
        raise FileNotFoundError(f"Training data missing under DATA_DIR={data_dir}")

    folds_df = pd.read_csv(folds_csv)
    fold_map = {str(r["stem"]): int(r["fold"]) for _, r in folds_df.iterrows()}
    vault_stems = {s for s, f in fold_map.items() if f == -1}
    include_vault = val_folds is None or (-1 in val_folds)
    assert not include_vault, "Vault stems must remain strictly isolated!"
    active_train_pool = [s for s, f in sorted(fold_map.items()) if f != -1]

    test_paths = {}
    if test_img_dir.exists():
        for p in sorted(test_img_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                test_paths[p.stem] = p
    test_stems = sorted(test_paths.keys())

    vault_read_count = 0
    if val_folds is None:
        fold_splits = [(0, active_train_pool, [])]
    else:
        fold_splits = []
        for vf in val_folds:
            val_s = [s for s in active_train_pool if fold_map[s] == vf]
            tr_s = [s for s in active_train_pool if fold_map[s] != vf]
            fold_splits.append((vf, tr_s, val_s))

    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")) if os.name != "nt" else Path(tempfile.gettempdir())
    if smoke:
        vf, tr_s, vs = fold_splits[0]
        tr_s = [s for s in tr_s if s not in vault_stems][:4]
        vs = [s for s in vs if s not in vault_stems][:2]
        if not vs:
            vs = [s for s in active_train_pool if s not in vault_stems][:2]
        fold_splits = [(vf, tr_s, vs)]
        active_stems_for_cache = sorted(set(tr_s) | set(vs))
        test_stems = test_stems[:2]
        cache_dir = tmp_root / "solar_paradigm_v10_smoke_cache"
    else:
        active_stems_for_cache = sorted({s for _vf, tr_s, vs in fold_splits for s in (tr_s + vs)})
        cache_dir = Path(os.environ.get("CACHE_DIR", str(tmp_root / "solar_paradigm_v10_cache")))

    stem_to_path: dict[str, Path] = {}
    for s in active_stems_for_cache:
        if s in vault_stems:
            vault_read_count += 1
            continue
        stem_to_path[s] = find_image_path(train_img_dir, s)
    for s in test_stems:
        stem_to_path[s] = test_paths[s]

    print(
        f"[029] Starting run | smoke={smoke} | val_folds={val_folds} | folds_n={len(fold_splits)} | "
        f"train/val stems={len(active_stems_for_cache)} | test stems={len(test_stems)} | cache_dir={cache_dir}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom_map = prepare_physics_cache(stem_to_path, cache_dir, workers=(2 if smoke else 12))

    def _safe_load_physics(cdir: str | Path, stem: str) -> np.ndarray:
        nonlocal vault_read_count
        if stem in vault_stems:
            vault_read_count += 1
            raise RuntimeError(f"Vault violation attempted on stem {stem}")
        return load_cached_physics_u8(cdir, stem)

    eval_val_stems = sorted({s for _vf, _tr, vs in fold_splits for s in vs})
    targets_by_stem, raw_val_gt, cons_val_gt, pure_v2_val_gt, a1_val_gt, as_subsets_val_gt = build_stem_targets_v10(
        gt_json_path,
        allowed_stems=set(active_stems_for_cache),
        vault_stems=vault_stems,
        val_stems=set(eval_val_stems),
        cache_dir=cache_dir,
        load_physics_fn=_safe_load_physics,
        geom_map=geom_map,
    )

    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)) if device.type == "cuda" else 0.0
    is_big_gpu = vram_gb >= 35.0
    det_sz = 512 if smoke else (DET_SIZE if is_big_gpu else 1216)
    slice_sz = 512 if smoke else SLICE_SIZE
    s1_epochs = 1 if smoke else (12 if is_big_gpu else 10)
    s1_bs = 2 if smoke else (8 if is_big_gpu else 4)
    native_crop = 512 if smoke else (768 if is_big_gpu else 640)
    native_epochs = 2 if smoke else (20 if is_big_gpu else 16)
    native_bs = 2 if smoke else (4 if is_big_gpu else 2)
    native_accum = 1 if smoke else 4
    native_crops_per_stem = 1 if smoke else 3
    tta_views = (0,) if smoke else (0, 1, 2, 3)

    n_folds = max(1, len(fold_splits))
    inv_folds = 1.0 / float(n_folds)
    prob_accum_dir = tmp_root / ("test_prob_accum_029_smoke" if smoke else "test_prob_accum_029")
    shutil.rmtree(prob_accum_dir, ignore_errors=True)
    prob_accum_dir.mkdir(parents=True, exist_ok=True)

    all_val_cands_by_tau: dict[str, dict[str, list[HybridCandidate]]] = {}
    pooled_test_s1_raw: dict[str, list[dict[str, Any]]] = {s: [] for s in test_stems}
    train_log_rows: list[dict[str, Any]] = []
    last_s1_hist: list[dict[str, float]] = []
    last_native_hist: list[dict[str, float]] = []
    s1_minutes_total = 0.0
    native_minutes_total = 0.0

    for fold_idx, (vf, train_stems, fold_val_stems) in enumerate(fold_splits):
        fold_seed = seed + fold_idx * 101
        _seed_everything(fold_seed)
        fold_tag = f"v10_f{vf}_seed{seed}"
        s1_export = f"stage1_mrcnn_f{vf}_fp16.pt"
        native_export = f"codeepnet_native_f{vf}_fp16.pt"

        t_ram = time.time()
        ram_train = preload_ram_physics(train_stems, cache_dir, load_fn=_safe_load_physics, workers=(2 if smoke else 8))
        print(f"[029 f{vf}] Preloaded {len(ram_train)} train stems into RAM in {(time.time() - t_ram):.1f}s", flush=True)

        # ---- Stage-1: Complementary 1408px Mask R-CNN v2 ----
        ts1 = time.time()
        s1_model, s1_hist = train_stage1_detector(
            train_stems=train_stems,
            ram_physics=ram_train,
            cache_dir=cache_dir,
            geom_map=geom_map,
            targets_map=targets_by_stem,
            out_dir=out_dir,
            device=device,
            epochs=s1_epochs,
            batch_size=s1_bs,
            det_size=det_sz,
            smoke=smoke,
            fold_tag=fold_tag,
            export_name=s1_export,
        )
        s1_minutes_total += (time.time() - ts1) / 60.0
        last_s1_hist = s1_hist
        for row in s1_hist:
            train_log_rows.append({"fold": f"f{vf}", **row})

        props_val_cache = cache_dir / f"stage1_val_props_{fold_tag}.pkl"
        if not smoke and os.environ.get("REUSE_STAGE1_CACHE", "1") == "1" and props_val_cache.is_file():
            with props_val_cache.open("rb") as pf:
                s1_val_props = pickle.load(pf)
        else:
            s1_val_props = (
                predict_stage1_proposals(
                    s1_model, fold_val_stems, cache_dir=cache_dir, geom_map=geom_map,
                    device=device, det_size=det_sz, tta_views=tta_views, score_thresh=0.10,
                )
                if fold_val_stems else {}
            )
            if not smoke:
                tmp_p = props_val_cache.with_suffix(".pkl.tmp")
                with tmp_p.open("wb") as pf:
                    pickle.dump(s1_val_props, pf, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_p, props_val_cache)

        raw_test_cache = cache_dir / f"stage1_test_raw_{fold_tag}.pkl"
        if not smoke and os.environ.get("REUSE_STAGE1_CACHE", "1") == "1" and raw_test_cache.is_file():
            with raw_test_cache.open("rb") as pf:
                s1_test_raw = pickle.load(pf)
        else:
            s1_test_raw = (
                predict_stage1_raw_candidates(
                    s1_model, test_stems, cache_dir=cache_dir, geom_map=geom_map,
                    device=device, det_size=det_sz, tta_views=tta_views, score_thresh=0.10,
                    view_offset=fold_idx * len(tta_views),
                )
                if test_stems else {}
            )
            if not smoke:
                tmp_p = raw_test_cache.with_suffix(".pkl.tmp")
                with tmp_p.open("wb") as pf:
                    pickle.dump(s1_test_raw, pf, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_p, raw_test_cache)

        for s in test_stems:
            pooled_test_s1_raw[s].extend(s1_test_raw.get(s, []))

        del s1_model, s1_test_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Primary Stream: Native 1:1 Co-DeepNet (ConvNeXt-Small HiRes + StripBarbUNet) ----
        ts2 = time.time()
        bitmasks = build_train_bitmasks(train_stems, targets_by_stem, geom_map, workers=(2 if smoke else 6))
        native_model, native_hist = train_native_codeepnet(
            train_stems=train_stems,
            ram_physics=ram_train,
            bitmasks=bitmasks,
            geom_map=geom_map,
            targets_map=targets_by_stem,
            out_dir=out_dir,
            cache_dir=cache_dir,
            device=device,
            epochs=native_epochs,
            batch_size=native_bs,
            grad_accum=native_accum,
            crop_size=native_crop,
            crops_per_stem=native_crops_per_stem,
            smoke=smoke,
            seed=fold_seed,
            fold_tag=fold_tag,
            export_name=native_export,
        )
        native_minutes_total += (time.time() - ts2) / 60.0
        last_native_hist = native_hist
        for row in native_hist:
            train_log_rows.append({"fold": f"f{vf}", **row})

        del ram_train, bitmasks
        gc.collect()

        # Stream validation 1:1 inference -> multi-hysteresis candidates
        ti = time.time()
        for s in fold_val_stems:
            hw3 = _safe_load_physics(cache_dir, s)
            x0, y0, p_fg, p_core, p_spine, p_dis = predict_native_full_disk(
                native_model, hw3, geom_map[s], device=device, slice_size=slice_sz, tta_views=tta_views,
            )
            all_val_cands_by_tau[s] = extract_stem_candidates_multitau(
                stem=s, x0=x0, y0=y0, p_fg=p_fg, p_core=p_core, p_spine=p_spine, p_disagree=p_dis,
                hw3_2048_u8=hw3, geom=geom_map[s], s1_props=s1_val_props.get(s, []),
                tau_hyst_grid=TAU_HYST_GRID,
            )

        # Accumulate native full-disk probability maps on test stems across folds on disk (float16)
        for s in test_stems:
            hw3 = load_cached_physics_u8(cache_dir, s)
            _x0, _y0, p_fg, p_core, p_spine, p_dis = predict_native_full_disk(
                native_model, hw3, geom_map[s], device=device, slice_size=slice_sz, tta_views=tta_views,
            )
            cur = np.stack([p_fg, p_core, p_spine, p_dis], axis=0).astype(np.float32) * inv_folds
            acc_path = prob_accum_dir / f"{s}.npy"
            if acc_path.is_file():
                prev = np.load(acc_path).astype(np.float32)
                cur = prev + cur
            np.save(acc_path, cur.astype(np.float16))

        print(f"[029 f{vf}] Full-disk 1888x1888 TTA + accum finished in {(time.time() - ti)/60.0:.2f}m", flush=True)

        del native_model, s1_val_props
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if train_log_rows:
        pd.DataFrame(train_log_rows).to_csv(out_dir / "train_log.csv", index=False)

    best_params: dict[str, Any] = {
        "tau_key": "0.50_0.06",
        "tau_seed": 0.50,
        "delta_hyst": 0.06,
        "tau_fg": 0.50,
        "keep_thresh": 0.24,
        "min_area": 32,
        "zero_fit_pq": 0.0,
        "oof_pq": 0.0,
        "subfolds": [],
    }
    oof_pq = 0.0
    zero_fit_pq = 0.0
    sweep_rows: list[dict[str, Any]] = []
    oof_val_rles: dict[str, list[dict]] = {}
    fold_models: list[dict[str, np.ndarray]] = []
    raw_res = {"pq": 0.0, "tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "split": 0, "merge": 0, "per_stem": {}, "mean_image_pq": 0.0}
    cons_res = None
    pure_v2_res = None
    a1_res = None
    as_res: dict[int, dict[str, Any]] = {}
    fold0_res: dict[str, Any] | None = None

    if eval_val_stems:
        best_params, oof_pq, zero_fit_pq, sweep_rows, oof_val_rles, fold_models = crossfit_hybrid_select(
            val_stems=eval_val_stems,
            cands_by_stem_tau=all_val_cands_by_tau,
            val_gt=raw_val_gt,
            tau_hyst_grid=TAU_HYST_GRID,
        )
        save_metascorer_json(w_dir / "metascorer_v10.json", best_params, fold_models)
        write_predictions_csv(oof_val_rles, eval_val_stems, out_dir / "val_predictions.csv")
        pred_rles_val = referee_metric.predictions(out_dir / "val_predictions.csv")
        raw_res = referee_metric.score(raw_val_gt, pred_rles_val)
        cons_res = referee_metric.score(cons_val_gt, pred_rles_val) if cons_val_gt else None
        pure_v2_res = referee_metric.score(pure_v2_val_gt, pred_rles_val) if pure_v2_val_gt else None
        a1_res = referee_metric.score(a1_val_gt, pred_rles_val) if a1_val_gt else None
        for k_ann, sub_gt in as_subsets_val_gt.items():
            if sub_gt:
                as_res[k_ann] = referee_metric.score(sub_gt, {s: pred_rles_val.get(s, []) for s in sub_gt})

        f0_stems = [s for s in eval_val_stems if fold_map.get(s) == 0]
        if f0_stems:
            f0_gt = {s: raw_val_gt[s] for s in f0_stems if s in raw_val_gt}
            f0_pr = {s: pred_rles_val.get(s, []) for s in f0_gt}
            fold0_res = referee_metric.score(f0_gt, f0_pr)

    overlap_val = count_overlapping_stems(oof_val_rles)

    # Decode pooled 3-fold test predictions directly from averaged prob maps + pooled Stage-1 proposals
    sel_ts = float(best_params["tau_seed"])
    sel_dh = float(best_params["delta_hyst"])
    total_s1_views = max(1, n_folds * len(tta_views))
    test_rles: dict[str, list[dict]] = {}
    for s in test_stems:
        hw3 = load_cached_physics_u8(cache_dir, s)
        geom = geom_map[s]
        from physics_v2 import active_disk_slice_origin
        x0, y0 = active_disk_slice_origin(float(geom["cx"]), float(geom["cy"]), slice_sz=slice_sz, img_size=IMG_SIZE)
        stack4 = np.load(prob_accum_dir / f"{s}.npy").astype(np.float32)
        s1_props = cluster_multiview_proposals(pooled_test_s1_raw.get(s, []), iou_thr=0.45, n_views=total_s1_views)
        c_map = extract_stem_candidates_multitau(
            stem=s, x0=x0, y0=y0, p_fg=stack4[0], p_core=stack4[1], p_spine=stack4[2], p_disagree=stack4[3],
            hw3_2048_u8=hw3, geom=geom, s1_props=s1_props, tau_hyst_grid=((sel_ts, sel_dh),),
        )
        tkey = next(iter(c_map.keys()))
        test_rles[s] = decode_test_stem_rles(
            c_map[tkey],
            models=fold_models,
            keep_thresh=float(best_params["keep_thresh"]),
            min_area=int(best_params["min_area"]),
        )

    shutil.rmtree(prob_accum_dir, ignore_errors=True)
    overlap_test = count_overlapping_stems(test_rles)
    if test_stems:
        write_predictions_csv(test_rles, test_stems, out_dir / "test_predictions.csv")

    sq, rq = _compute_sq_rq(raw_res)
    clean_sq, clean_rq = _compute_sq_rq(cons_res) if cons_res else (0.0, 0.0)
    pv2_sq, pv2_rq = _compute_sq_rq(pure_v2_res) if pure_v2_res else (0.0, 0.0)
    a1_sq, a1_rq = _compute_sq_rq(a1_res) if a1_res else (0.0, 0.0)
    ci_lo, ci_hi = referee_metric.ci(raw_res) if eval_val_stems else (0.0, 0.0)
    c_lo, c_hi = referee_metric.ci(cons_res) if (eval_val_stems and cons_res) else (0.0, 0.0)
    f0_lo, f0_hi = referee_metric.ci(fold0_res) if fold0_res else (0.0, 0.0)
    total_minutes = (time.time() - t0) / 60.0

    (out_dir / "score.json").write_text(json.dumps(raw_res), encoding="utf-8")
    (out_dir / "postprocess.json").write_text(
        json.dumps({"params": best_params, "grid": sweep_rows}, indent=1),
        encoding="utf-8",
    )

    metrics = {
        "job": "029_prod_multifold_hybrid_codeepnet",
        "self_val_pq": float(raw_res["pq"]) if eval_val_stems else None,
        "oof_val_pq": float(oof_pq) if eval_val_stems else None,
        "zero_fit_pq": float(zero_fit_pq) if eval_val_stems else None,
        "ci90": [float(ci_lo), float(ci_hi)],
        "fold0_val_pq": float(fold0_res["pq"]) if fold0_res else None,
        "fold0_ci90": [float(f0_lo), float(f0_hi)],
        "n_fold0_stems": len(fold0_res["per_stem"]) if fold0_res else 0,
        "consensus_pq": float(cons_res["pq"]) if cons_res else 0.0,
        "consensus_ci90": [float(c_lo), float(c_hi)],
        "pure_votes2_consensus_pq": float(pure_v2_res["pq"]) if pure_v2_res else 0.0,
        "pure_votes2_sq": pv2_sq,
        "pure_votes2_rq": pv2_rq,
        "a1_pq": float(a1_res["pq"]) if a1_res else 0.0,
        "a1_sq": a1_sq,
        "a1_rq": a1_rq,
        "as1_pq": float(as_res[1]["pq"]) if 1 in as_res else 0.0,
        "as2_pq": float(as_res[2]["pq"]) if 2 in as_res else 0.0,
        "as3_pq": float(as_res[3]["pq"]) if 3 in as_res else 0.0,
        "sq": sq,
        "rq": rq,
        "clean_sq": clean_sq,
        "clean_rq": clean_rq,
        "tp": int(raw_res["tp"]),
        "fp": int(raw_res["fp"]),
        "fn": int(raw_res["fn"]),
        "split": int(raw_res["split"]),
        "merge": int(raw_res["merge"]),
        "mean_image_pq": float(raw_res.get("mean_image_pq", 0.0)),
        "stems_with_overlap_val": int(overlap_val),
        "stems_with_overlap_test": int(overlap_test),
        "n_val_stems": len(eval_val_stems),
        "n_test_stems": len(test_stems),
        "vault_read_count": int(vault_read_count),
        "train_minutes": float(total_minutes),
        "stage1_minutes": float(s1_minutes_total),
        "codeepnet_minutes": float(native_minutes_total),
        "epochs": int(len(train_log_rows)),
        "val_folds": val_folds,
        "seed": int(seed),
        "smoke": bool(smoke),
        "best_params": best_params,
        "weight_files": sorted(p.name for p in w_dir.iterdir() if p.is_file()),
        "notes": (
            "Paradigm v10 Production 3-Fold Hybrid Native 1:1 Co-DeepNet (ConvNeXt-Small + StripBarbUNet, bit-4 thin boost, "
            "Stage-1 1408px Mask R-CNN multi-fold proposal clustering, Geodesic Hysteresis binary_propagation, "
            "24-feature 5-subfold chronological Ridge+Logistic expected-PQ meta-scorer, fp16 portable weight export)."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_error_analysis(out_dir, metrics, best_params, raw_res, pure_v2_res)

    if eval_val_stems:
        render_dark_ui_plots(
            out_dir=out_dir,
            val_stems=eval_val_stems,
            cache_dir=cache_dir,
            geom_map=geom_map,
            s1_history=last_s1_hist,
            native_history=last_native_hist,
            cands_by_stem_tau=all_val_cands_by_tau,
            oof_rles=oof_val_rles,
            raw_val_gt=raw_val_gt,
            cons_val_gt=cons_val_gt,
            sweep_rows=sweep_rows,
            summary_metrics=metrics,
            raw_res=raw_res,
        )

    print(
        f"[029] DONE in {total_minutes:.2f}m | self_val_pq={metrics['self_val_pq']} | fold0_pq={metrics['fold0_val_pq']} | "
        f"zero_fit_pq={metrics['zero_fit_pq']} | consensus_pq={metrics['consensus_pq']:.4f} | "
        f"overlap=({overlap_val},{overlap_test}) | vault_reads={vault_read_count} | weights={metrics['weight_files']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
