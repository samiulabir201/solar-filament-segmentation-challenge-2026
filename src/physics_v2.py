"""Clean-sheet GONG H-alpha limb detector, 2D illumination-first physics, RAM preloader, and GPU 5-ch native barb tensor."""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 2048
SLICE_SIZE = 1888  # 59 * 32; encloses 100.0% of solar chromosphere (2 * R_sun <= 1844 < 1888) at 1:1 native lattice
DEFAULT_CX = 1021.5
DEFAULT_CY = 1022.2
DEFAULT_R = 906.8


def _algebraic_circle_irls(xs: np.ndarray, ys: np.ndarray, iters: int = 4, delta: float = 2.0) -> tuple[float, float, float, float]:
    """Fit circle (cx, cy, R) with Huber IRLS; returns (cx, cy, R, rmse)."""
    w = np.ones_like(xs, dtype=np.float64)
    A_mat = np.column_stack([xs, ys, np.ones_like(xs)])
    rhs = -(xs * xs + ys * ys)
    cx, cy, R = DEFAULT_CX, DEFAULT_CY, DEFAULT_R
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A_mat * sw, rhs * sw[:, 0], rcond=None)
        cx = -0.5 * float(coef[0])
        cy = -0.5 * float(coef[1])
        r2 = cx * cx + cy * cy - float(coef[2])
        if r2 <= 1.0:
            return DEFAULT_CX, DEFAULT_CY, DEFAULT_R, 99.0
        R = float(np.sqrt(r2))
        dists = np.hypot(xs - cx, ys - cy) - R
        abs_d = np.abs(dists)
        w = np.where(abs_d <= delta, 1.0, delta / np.maximum(abs_d, 1e-6))
    rmse = float(np.sqrt(np.mean((np.hypot(xs - cx, ys - cy) - R) ** 2)))
    return cx, cy, R, rmse


def fit_gong_limb(img_u8: np.ndarray) -> tuple[float, float, float, int]:
    """Locate the inner H-alpha solar limb (R in [892, 922] px), rejecting outer coronagraph edges."""
    ys_m, xs_m = np.nonzero(img_u8[::4, ::4] > 75)
    if len(xs_m) > 100:
        cx0 = float(np.clip(0.5 * (float(xs_m.mean()) * 4.0) + 0.5 * DEFAULT_CX, 995.0, 1050.0))
        cy0 = float(np.clip(0.5 * (float(ys_m.mean()) * 4.0) + 0.5 * DEFAULT_CY, 995.0, 1050.0))
    else:
        cx0, cy0 = DEFAULT_CX, DEFAULT_CY

    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False, dtype=np.float32)
    radii = np.arange(860.0, 946.0, 1.0, dtype=np.float32)
    cos_t = np.cos(thetas)[:, None]
    sin_t = np.sin(thetas)[:, None]
    img_f = img_u8.astype(np.float32)

    cx, cy, R, rmse = cx0, cy0, DEFAULT_R, 99.0
    n_inliers = 0
    for pass_idx in range(2):
        map_x = (cx + cos_t * radii[None, :]).astype(np.float32)
        map_y = (cy + sin_t * radii[None, :]).astype(np.float32)
        prof = cv2.remap(img_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        prof_s = prof.copy()
        prof_s[:, 1:-1] = 0.25 * prof[:, :-2] + 0.5 * prof[:, 1:-1] + 0.25 * prof[:, 2:]

        r_lo = int(882 - 860)
        r_hi = int(926 - 860)
        grad = np.zeros_like(prof_s)
        grad[:, 4:-4] = prof_s[:, :-8] - prof_s[:, 8:]
        sub = grad[:, r_lo:r_hi]
        best_idx = np.argmax(sub, axis=1) + r_lo
        best_r = radii[best_idx]
        best_g = grad[np.arange(len(thetas)), best_idx]
        inner_i = prof_s[np.arange(len(thetas)), np.maximum(0, best_idx - 14)]

        valid = (inner_i > 58.0) & (best_g > 9.0) & (best_r > 883.0) & (best_r < 925.0)
        if pass_idx == 1 and 890.0 <= R <= 924.0:
            valid &= np.abs(best_r - R) <= 8.0
        n_inliers = int(valid.sum())
        if n_inliers < 45:
            return DEFAULT_CX, DEFAULT_CY, DEFAULT_R, n_inliers

        px = cx + cos_t[valid, 0] * best_r[valid]
        py = cy + sin_t[valid, 0] * best_r[valid]
        cx_fit, cy_fit, R_fit, rmse_fit = _algebraic_circle_irls(px.astype(np.float64), py.astype(np.float64))
        if 892.0 <= R_fit <= 922.0 and abs(cx_fit - DEFAULT_CX) <= 50.0 and abs(cy_fit - DEFAULT_CY) <= 50.0:
            cx, cy, R, rmse = cx_fit, cy_fit, R_fit, rmse_fit
        else:
            return DEFAULT_CX, DEFAULT_CY, DEFAULT_R, n_inliers

    if not (892.0 <= R <= 922.0) or rmse > 8.0:
        return DEFAULT_CX, DEFAULT_CY, DEFAULT_R, n_inliers
    return float(cx), float(cy), float(R), n_inliers


def _clamp_exterior_512(img512: np.ndarray, cx: float, cy: float, R_sun: float) -> tuple[np.ndarray, np.ndarray]:
    cx5 = cx / 4.0
    cy5 = cy / 4.0
    r_lim = 0.982 * (R_sun / 4.0)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    dx = xx - cx5
    dy = yy - cy5
    rad = np.hypot(dx, dy)
    scale = np.where(rad > r_lim, r_lim / np.maximum(rad, 1e-6), 1.0).astype(np.float32)
    mx = (cx5 + dx * scale).astype(np.float32)
    my = (cy5 + dy * scale).astype(np.float32)
    clamped = cv2.remap(img512, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    on_disk512 = rad <= (0.990 * (R_sun / 4.0))
    return clamped, on_disk512


def compute_solar_geometry(cx: float, cy: float, R_sun: float, size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return (on_disk bool mask, mu cosine heliocentric angle float32) at native resolution."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rho = np.hypot(xx - np.float32(cx), yy - np.float32(cy)) / np.float32(R_sun)
    on_disk = rho <= np.float32(0.996)
    mu = np.sqrt(np.clip(1.0 - rho * rho, 0.0, 1.0)).astype(np.float32)
    return on_disk, mu


def _dark_ridge_tubularity_1024(d_multi_u8: np.ndarray, on_disk_1024: np.ndarray) -> np.ndarray:
    d1024 = cv2.resize(d_multi_u8, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)
    best = np.zeros((1024, 1024), dtype=np.float32)
    for sigma in (1.2, 2.4, 4.8):
        k = max(3, int(round(sigma * 6.0)) | 1)
        sm = cv2.GaussianBlur(d1024, (k, k), sigmaX=sigma, sigmaY=sigma)
        dxx = (np.roll(sm, -1, axis=1) - 2.0 * sm + np.roll(sm, 1, axis=1)) * (sigma * sigma)
        dyy = (np.roll(sm, -1, axis=0) - 2.0 * sm + np.roll(sm, 1, axis=0)) * (sigma * sigma)
        dxy = 0.25 * (
            np.roll(np.roll(sm, -1, axis=0), -1, axis=1)
            - np.roll(np.roll(sm, -1, axis=0), 1, axis=1)
            - np.roll(np.roll(sm, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(sm, 1, axis=0), 1, axis=1)
        ) * (sigma * sigma)
        tr = -(dxx + dyy)
        det_diff = np.sqrt(np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * dxy * dxy))
        lam_pos = 0.5 * (tr + det_diff)
        resp = np.maximum(0.0, lam_pos) * np.clip(d1024 * 4.0, 0.25, 1.0)
        best = np.maximum(best, resp)

    vals = best[on_disk_1024]
    scale = float(np.percentile(vals, 99.5)) if vals.size > 100 else 0.05
    scale = max(scale, 0.04)
    norm = np.clip(best / scale, 0.0, 1.0)
    norm[~on_disk_1024] = 0.0
    return (norm * 255.0).round().astype(np.uint8)


def build_physics_channels_u8(
    img_u8: np.ndarray,
    cx: float | None = None,
    cy: float | None = None,
    R_sun: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build 2048x2048x3 uint8 solar physics tensor [I_flat, D_multi, T_ribbon]."""
    if cx is None or cy is None or R_sun is None:
        cx, cy, R_sun, n_inl = fit_gong_limb(img_u8)
    else:
        n_inl = 360

    img_f32 = img_u8.astype(np.float32)
    img512 = cv2.resize(img_f32, (512, 512), interpolation=cv2.INTER_AREA)
    clamped512, _ = _clamp_exterior_512(img512, cx, cy, R_sun)

    m0 = cv2.GaussianBlur(clamped512, (0, 0), 18.0)
    no_plage = np.minimum(clamped512, m0 + 18.0)
    k21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    b_close = cv2.morphologyEx(no_plage, cv2.MORPH_CLOSE, k21)
    b_mid512 = cv2.GaussianBlur(b_close, (0, 0), 7.0)
    b_broad512 = cv2.GaussianBlur(b_close, (0, 0), 20.0)
    s_2d512 = np.maximum(cv2.GaussianBlur(np.abs(no_plage - b_mid512) * 1.4826, (0, 0), 16.0), 3.5)

    b_mid = cv2.resize(b_mid512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    b_broad = cv2.resize(b_broad512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    s_2d = cv2.resize(s_2d512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    on_disk, _ = compute_solar_geometry(cx, cy, R_sun, IMG_SIZE)
    on_disk_1024 = cv2.resize(on_disk.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST) > 0

    diff_mid = b_mid - img_f32
    diff_broad = b_broad - img_f32

    ch0_f = np.clip(185.0 - 28.0 * (diff_mid / s_2d), 0.0, 255.0)
    ch0_u8 = np.where(on_disk, ch0_f, 185.0).round().astype(np.uint8)

    d_mid = np.clip(diff_mid / (6.5 * s_2d), 0.0, 1.0)
    d_broad = np.clip(diff_broad / (7.5 * s_2d), 0.0, 1.0)
    d_multi_u8 = np.where(on_disk, (0.55 * d_mid + 0.45 * d_broad) * 255.0, 0.0).round().astype(np.uint8)

    t_1024 = _dark_ridge_tubularity_1024(d_multi_u8, on_disk_1024)
    ch2_u8 = cv2.resize(t_1024, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    ch2_u8[~on_disk] = 0

    hw3 = np.stack([ch0_u8, d_multi_u8, ch2_u8], axis=-1)
    geom = {
        "cx": float(cx),
        "cy": float(cy),
        "R_sun": float(R_sun),
        "n_inliers": int(n_inl),
        "s_2d_med": float(np.median(s_2d[on_disk])),
    }
    return hw3, geom


def find_image_path(folder: Path, stem: str) -> Path:
    for ext in (".jpeg", ".jpg", ".png"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    matches = sorted(folder.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Image for stem {stem} not found in {folder}")
    return matches[0]


def prepare_physics_cache(
    stem_to_path: dict[str, Path],
    cache_dir: Path,
    workers: int = 12,
) -> dict[str, dict[str, float]]:
    """Build or load cached (2048, 2048, 3) uint8 solar physics images + limb geometry JSON."""
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)
    geom_path = cache_dir / "limb_geometry.json"
    geom_map: dict[str, dict[str, float]] = {}
    if geom_path.is_file():
        try:
            geom_map = json.loads(geom_path.read_text(encoding="utf-8"))
        except Exception:
            geom_map = {}

    free_gb = shutil.disk_usage(str(cache_dir)).free / (1024**3)
    use_npy = free_gb > 20.0
    ext = ".npy" if use_npy else ".png"

    def _one(stem: str, img_path: Path) -> tuple[str, dict[str, float]]:
        out_p = cache_dir / f"{stem}{ext}"
        alt_p = cache_dir / f"{stem}{'.png' if use_npy else '.npy'}"
        if (out_p.is_file() or alt_p.is_file()) and stem in geom_map:
            return stem, geom_map[stem]
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"Failed to read {img_path}")
        hw3, geom = build_physics_channels_u8(raw)
        if use_npy:
            np.save(out_p, hw3)
        else:
            cv2.imwrite(str(out_p), hw3, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return stem, geom

    pending = [
        (s, p)
        for s, p in stem_to_path.items()
        if s not in geom_map or not ((cache_dir / f"{s}.npy").is_file() or (cache_dir / f"{s}.png").is_file())
    ]
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 4))) as ex:
            for stem, geom in ex.map(lambda sp: _one(sp[0], sp[1]), pending):
                geom_map[stem] = geom
        geom_path.write_text(json.dumps(geom_map, indent=1), encoding="utf-8")
    return {s: geom_map[s] for s in stem_to_path}


_MMAP_CACHE: dict[str, np.ndarray] = {}


def load_cached_physics_u8(cache_dir: Path, stem: str) -> np.ndarray:
    key = f"{cache_dir}:{stem}"
    cached = _MMAP_CACHE.get(key)
    if cached is not None:
        return cached
    npy_p = cache_dir / f"{stem}.npy"
    if npy_p.is_file():
        arr = np.load(npy_p, mmap_mode="r")
        if len(_MMAP_CACHE) >= 1024:
            _MMAP_CACHE.pop(next(iter(_MMAP_CACHE)))
        _MMAP_CACHE[key] = arr
        return arr
    png_p = cache_dir / f"{stem}.png"
    if png_p.is_file():
        arr = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Corrupted cache png: {png_p}")
        return arr
    raise FileNotFoundError(f"Missing cached physics for stem {stem} in {cache_dir}")


def preload_ram_physics(
    stems: list[str],
    cache_dir: Path,
    load_fn=load_cached_physics_u8,
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load all (2048, 2048, 3) uint8 arrays into RAM for zero-I/O training (~5.6 GB for 482 stems)."""
    out: dict[str, np.ndarray] = {}

    def _read_one(s: str) -> tuple[str, np.ndarray]:
        arr = np.array(load_fn(cache_dir, s), dtype=np.uint8, copy=True)
        return s, arr

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(stems) or 1))) as ex:
        for s, arr in ex.map(_read_one, stems):
            out[s] = arr
    return out


def active_disk_slice_origin(
    cx: float,
    cy: float,
    slice_sz: int = SLICE_SIZE,
    img_size: int = IMG_SIZE,
) -> tuple[int, int]:
    """Integer top-left (x0, y0) of the 1:1 native slice enclosing the full solar disk."""
    max_off = max(0, img_size - slice_sz)
    x0 = int(np.clip(round(float(cx) - 0.5 * slice_sz), 0, max_off))
    y0 = int(np.clip(round(float(cy) - 0.5 * slice_sz), 0, max_off))
    return x0, y0


def transform_local_center_d4(
    cx_loc: float,
    cy_loc: float,
    width: int,
    height: int,
    k_rot: int,
    hflip: bool,
) -> tuple[float, float]:
    """Transform crop-local solar center (cx_loc, cy_loc) identically to np.rot90 + hflip."""
    cx_t, cy_t = float(cx_loc), float(cy_loc)
    cur_h, cur_w = float(height), float(width)
    for _ in range(k_rot % 4):
        # np.rot90 ccw: (x, y) -> (y, W - 1 - x); dimensions swap
        new_cx = cy_t
        new_cy = (cur_w - 1.0) - cx_t
        cx_t, cy_t = new_cx, new_cy
        cur_h, cur_w = cur_w, cur_h
    if hflip:
        cx_t = (cur_w - 1.0) - cx_t
    return cx_t, cy_t


def build_native_5ch_gpu(
    hw3_bchw_u8: torch.Tensor,
    local_cxcy_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 5-channel ImageNet-polarity native physics tensor on GPU.

    Args:
      hw3_bchw_u8: (B, 3, H, W) uint8/float tensor containing [I_flat, D_multi, T_ribbon].
      local_cxcy_r: (B, 3) float32 tensor containing (cx_local, cy_local, R_sun) in crop coords.
    Returns:
      x_5ch: (B, 5, H, W) float32 in [0, 1]: [I_flat, 1 - D_multi, 1 - D_barb15, T_ribbon, mu_disk].
      on_disk: (B, H, W) float32 in {0.0, 1.0}.
    """
    dev = hw3_bchw_u8.device
    b, _, h, w = hw3_bchw_u8.shape
    f32 = hw3_bchw_u8.float() * (1.0 / 255.0)
    i_flat = f32[:, 0:1]
    d_multi = f32[:, 1:2]
    t_ribbon = f32[:, 2:3]

    gx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    gy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    cx = local_cxcy_r[:, 0].to(dtype=torch.float32, device=dev).view(b, 1, 1, 1)
    cy = local_cxcy_r[:, 1].to(dtype=torch.float32, device=dev).view(b, 1, 1, 1)
    r_sun = local_cxcy_r[:, 2].to(dtype=torch.float32, device=dev).view(b, 1, 1, 1).clamp(min=100.0)

    rho2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / (r_sun * r_sun)
    on_disk = (rho2 <= 0.996).to(torch.float32)
    mu_disk = torch.sqrt(torch.clamp(1.0 - rho2, min=0.0, max=1.0)) * on_disk

    # Native 1:1 15x15 fine barb local contrast enhancement (no downsampling!)
    bg_15 = F.avg_pool2d(i_flat, kernel_size=15, stride=1, padding=7, count_include_pad=False)
    d_barb = torch.clamp((bg_15 - i_flat) * 10.0, min=0.0, max=1.0) * on_disk

    # Polarity aligned with ImageNet on ch 0..2: bright quiet Sun, dark filaments
    x_5ch = torch.cat([i_flat, (1.0 - d_multi), (1.0 - d_barb), t_ribbon, mu_disk], dim=1)
    return x_5ch, on_disk[:, 0]
