"""Task 4 / Part A: non-learned enhancement baselines vs the learned 1M models.

All methods are evaluated with the SAME metric implementation used by the
frozen Stage-4/5 protocol (experiments/rcm_dce/metrics.py), so the numbers are
directly comparable with the published v0.7 results.

Methods
-------
input            : identity (no enhancement) - the honest reference
he               : global histogram equalisation on luma
clahe            : CLAHE on luma (clip=2.0, grid=8)
gamma            : fixed gamma 2.2
ssr              : single-scale Retinex (gaussian sigma 80)
msr              : multi-scale Retinex (15/80/250)
msrcr            : MSR with colour restoration
lime             : LIME (lighting map via L0-smooth refinement)
plain125         : learned baseline, 995,253 params (seed 20260830)
ms_refine_uncond : learned candidate, 1,006,265 params (seed 20260830)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
sys.path.insert(0, str(PROJECT / "experiments"))

from rcm_dce.model import RCMCurveNet  # noqa: E402

CKPT_ROOT = PROJECT / "experiments" / "results" / "stage4_clean_balanced_3seed" / "seed20260830"
CKPT = {
    "plain125": CKPT_ROOT / "plain125" / "plain_epoch002.pth",
    "ms_refine_uncond": CKPT_ROOT / "ms_refine_uncond" / "ms_refine_uncond_epoch003.pth",
}
LOL_EVAL15 = PROJECT / "experiments" / "datasets" / "LOL_v1" / "extracted" / "eval15"
DARK_NIGHT_VAL_IMG = PROJECT / "experiments" / "datasets" / "DarkDriving_det_release" / "Det_data" / "night" / "images" / "val"
DARK_NIGHT_VAL_LBL = PROJECT / "experiments" / "datasets" / "DarkDriving_det_release" / "Det_data" / "night" / "labels" / "val"


# ----------------------------------------------------------------------------
# classical (non-learned) operators - all deterministic, zero training
# ----------------------------------------------------------------------------
def m_input(rgb):
    return rgb


def m_he(rgb):
    import cv2
    u8 = (rgb * 255).round().astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return out.astype(np.float32) / 255.0


def m_clahe(rgb):
    import cv2
    u8 = (rgb * 255).round().astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


def m_gamma(rgb, g=2.2):
    return np.clip(rgb, 0, 1) ** (1.0 / g)


def _gauss(x, sigma):
    import cv2
    k = int(sigma * 6) | 1
    return cv2.GaussianBlur(x, (k, k), sigma)


def _ssr_channel(ch, sigma):
    ch = np.maximum(ch, 1e-4)
    return np.log10(ch) - np.log10(np.maximum(_gauss(ch, sigma), 1e-4))


def m_ssr(rgb, sigma=80.0):
    out = np.stack([_ssr_channel(rgb[:, :, c], sigma) for c in range(3)], axis=2)
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    return out.astype(np.float32)


def m_msr(rgb, sigmas=(15.0, 80.0, 250.0)):
    acc = np.zeros_like(rgb, dtype=np.float32)
    for s in sigmas:
        for c in range(3):
            acc[:, :, c] += _ssr_channel(rgb[:, :, c], s) / len(sigmas)
    out = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-6)
    return out.astype(np.float32)


def m_msrcr(rgb, sigmas=(15.0, 80.0, 250.0)):
    msr = m_msr(rgb, sigmas)
    eps = 1e-4
    denom = np.maximum(rgb.sum(axis=2, keepdims=True), eps)
    cr = np.log10(np.maximum(rgb * 3.0 / denom, eps)) * 1.0
    out = msr * (cr + 1.0)
    return np.clip((out - out.min()) / max(out.max() - out.min(), 1e-6), 0, 1).astype(np.float32)


def m_lime(rgb, gamma=0.8, rho=2.0, eps=1e-4, iters=3):
    """LIME: estimate illumination T, refine with L0-like smoothing, then I/T^gamma."""
    import cv2
    t = rgb.max(axis=2)
    t0 = t.copy()
    for _ in range(iters):
        t = cv2.GaussianBlur(t, (0, 0), 3.0)
        r = t - t0
        # approximate L0 by thresholding small gradients in the residual
        gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        w = (mag < 0.02).astype(np.float32)
        t = t * w + t0 * (1 - w)
    t = np.clip(t, eps, 1.0)[:, :, None]
    out = rgb / t ** gamma
    return np.clip(out, 0, 1).astype(np.float32)


CLASSICAL = {
    "input": m_input,
    "he": m_he,
    "clahe": m_clahe,
    "gamma": m_gamma,
    "ssr": m_ssr,
    "msr": m_msr,
    "msrcr": m_msrcr,
    "lime": m_lime,
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_pairs(root_low, root_high, limit=None, size=None):
    """Return list of (id, low_rgb_float32, high_rgb_float32)."""
    lows = sorted([p for p in Path(root_low).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if limit:
        lows = lows[:limit]
    out = []
    for lp in lows:
        hp = Path(root_high) / lp.name
        if not hp.exists():
            continue
        lo = np.asarray(Image.open(lp).convert("RGB"), dtype=np.float32) / 255.0
        hi = np.asarray(Image.open(hp).convert("RGB"), dtype=np.float32) / 255.0
        if lo.shape != hi.shape:
            continue
        if size:
            im_lo = Image.fromarray((lo * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
            im_hi = Image.fromarray((hi * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
            lo = np.asarray(im_lo, dtype=np.float32) / 255.0
            hi = np.asarray(im_hi, dtype=np.float32) / 255.0
        out.append((lp.stem, lo, hi))
    return out


def enhance_to_disk(method, src_img_dir, dst_img_dir, limit=0, skip_existing=False,
                    quality=95, enh_max=0, restore_size=True):
    """Stream a whole image directory through one classical operator.

    Used by Part B.  Enhancement never needs the ground-truth target, so this
    can process an unpaired directory at full resolution without loading all
    images into memory at once.

    ``enh_max`` caps the long side used *for the operator*, because the pixel-wise
    Retinex variants cost seconds per image at 2448x2048.  The enhanced result is
    then resized back to the original geometry, so the detector still sees the
    same frame shape as every other arm.  ``enh_max=0`` means native resolution.
    """
    src = Path(src_img_dir)
    dst = Path(dst_img_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if limit:
        files = files[:limit]
    fn = CLASSICAL.get(method)
    if fn is None:
        raise SystemExit("unknown classical method %r" % method)
    done = 0
    t0 = time.time()
    for p in files:
        op = dst / (p.stem + ".jpg")
        if skip_existing and op.exists():
            done += 1
            continue
        img = Image.open(p).convert("RGB")
        w, h = img.size
        work = img
        if enh_max and max(w, h) > enh_max:
            s = enh_max / max(w, h)
            work = img.resize((max(8, int(round(w * s))), max(8, int(round(h * s)))), Image.BILINEAR)
        rgb = np.asarray(work, dtype=np.float32) / 255.0
        out = np.clip(fn(rgb), 0, 1)
        res = Image.fromarray((out * 255).round().astype(np.uint8))
        if restore_size and res.size != (w, h):
            res = res.resize((w, h), Image.BILINEAR)
        res.save(op, quality=quality)
        done += 1
        if done % 100 == 0:
            print("   %s: %d/%d  (%.2f img/s)" % (method, done, len(files),
                                                  done / max(time.time() - t0, 1e-6)), flush=True)
    return done


# ----------------------------------------------------------------------------
# metrics (mirror metrics.py exactly)
# ----------------------------------------------------------------------------
def metrics_for(pred, target, source):
    from skimage.color import deltaE_ciede2000, rgb2lab
    from skimage.metrics import structural_similarity
    p = np.clip(pred, 0, 1).astype(np.float64)
    t = np.clip(target, 0, 1).astype(np.float64)
    z = np.clip(source, 0, 1).astype(np.float64)
    mse = float(np.mean((p - t) ** 2))
    psnr = float(-10.0 * np.log10(max(mse, 1e-10)))
    ssim = float(structural_similarity(p, t, channel_axis=2, data_range=1.0))
    de = float(deltaE_ciede2000(rgb2lab(p), rgb2lab(t)).mean())
    pl, tl, zl = p.mean(axis=2), t.mean(axis=2), z.mean(axis=2)
    dx_p, dx_t = p[:, 1:, :] - p[:, :-1, :], t[:, 1:, :] - t[:, :-1, :]
    dy_p, dy_t = p[1:, :, :] - p[:-1, :, :], t[1:, :, :] - t[:-1, :, :]
    grad = float(0.5 * (np.mean(np.abs(dx_p - dx_t)) + np.mean(np.abs(dy_p - dy_t))))
    return {
        "psnr_db": psnr, "ssim": ssim, "deltaE2000": de,
        "mae": float(np.mean(np.abs(p - t))),
        "gradient_l1": grad,
        "output_clip_fraction_luma_gt_0.98": float((pl > 0.98).mean()),
        "output_clip_fraction_channel_gt_0.995": float((p > 0.995).mean()),
        "input_mean_luma": float(zl.mean()),
        "output_mean_luma": float(pl.mean()),
        "target_mean_luma": float(tl.mean()),
    }


def aggregate(rows):
    keys = rows[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        agg[k] = float(np.mean(vals)) if vals else None
        agg[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


# ----------------------------------------------------------------------------
# learned models
# ----------------------------------------------------------------------------
def load_learned(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = payload["model_signature"]
    model = RCMCurveNet(sig["variant"], sig["width"], sig["conv_mode"],
                        sig["scale_factor"], sig["curve_steps"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    n = sum(p.numel() for p in model.parameters())
    return model, sig, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\33848\论文投稿\darkdriving\results_task4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=0, help="optional square resize for speed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="", help="comma list; default = classical only")
    ap.add_argument("--skip-learned", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (args.size, args.size) if args.size else None

    pairs = load_pairs(LOL_EVAL15 / "low", LOL_EVAL15 / "high", limit=args.limit or None, size=size)
    print("LOL-v1 eval15 pairs: %d" % len(pairs), flush=True)
    if not pairs:
        raise SystemExit("no pairs found at %s" % LOL_EVAL15)

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()] or list(CLASSICAL)
    results = {}

    for name in wanted:
        fn = CLASSICAL.get(name)
        if fn is None:
            print("skip unknown method %s" % name)
            continue
        rows = []
        t0 = time.time()
        for pid, lo, hi in pairs:
            rows.append(metrics_for(fn(lo), hi, lo))
        agg = aggregate(rows)
        agg["seconds_total"] = round(time.time() - t0, 3)
        agg["seconds_per_image"] = round((time.time() - t0) / max(1, len(pairs)), 4)
        agg["n_images"] = len(pairs)
        results[name] = agg
        print("%-18s PSNR %6.3f  SSIM %6.4f  dE00 %6.3f  clip_luma %.5f  t/img %.4fs"
              % (name, agg["psnr_db"], agg["ssim"], agg["deltaE2000"],
                 agg["output_clip_fraction_luma_gt_0.98"], agg["seconds_per_image"]), flush=True)

    if not args.skip_learned:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        for name, ck in CKPT.items():
            if not ck.exists():
                print("MISSING checkpoint for %s: %s" % (name, ck))
                continue
            try:
                model, sig, nparam = load_learned(ck, device)
            except Exception as exc:
                print("failed to load %s: %s: %s" % (name, type(exc).__name__, exc))
                continue
            rows = []
            t0 = time.time()
            with torch.no_grad():
                for pid, lo, hi in pairs:
                    x = torch.from_numpy(lo).permute(2, 0, 1).unsqueeze(0).to(device)
                    out = model(x)
                    pred = (out[0] if isinstance(out, tuple) else out)[0].permute(1, 2, 0).cpu().numpy()
                    rows.append(metrics_for(pred, hi, lo))
            agg = aggregate(rows)
            agg["parameters"] = nparam
            agg["seconds_per_image"] = round((time.time() - t0) / max(1, len(pairs)), 4)
            agg["n_images"] = len(pairs)
            agg["checkpoint"] = str(ck)
            results[name] = agg
            print("%-18s PSNR %6.3f  SSIM %6.4f  dE00 %6.3f  clip_luma %.5f  params %d"
                  % (name, agg["psnr_db"], agg["ssim"], agg["deltaE2000"],
                     agg["output_clip_fraction_luma_gt_0.98"], nparam), flush=True)

    outp = out_dir / "task4_partA_lol_eval15.json"
    outp.write_text(json.dumps({
        "protocol": "task4_nonlearned_vs_learned",
        "dataset": "LOL-v1 eval15",
        "n_images": len(pairs),
        "resize": list(size) if size else None,
        "metric_source": "mirrors experiments/rcm_dce/metrics.py",
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
