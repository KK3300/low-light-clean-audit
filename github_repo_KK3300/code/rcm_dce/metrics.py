"""Reference and region diagnostics used by the new protocol."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return float(-10.0 * np.log10(max(mse, 1e-10)))


def _masked_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 16:
        return None
    diff = pred - target
    mse = float(np.mean(diff[mask] ** 2))
    return float(-10.0 * np.log10(max(mse, 1e-10)))


def _box3(x: np.ndarray) -> np.ndarray:
    """Dependency-free 3x3 reflect-padded mean for a single-channel image."""
    h, w = x.shape
    p = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += p[dy : dy + h, dx : dx + w]
    return out / 9.0


def _gradient_l1(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute mismatch of first-order RGB gradients."""
    dx_p, dx_t = pred[:, 1:, :] - pred[:, :-1, :], target[:, 1:, :] - target[:, :-1, :]
    dy_p, dy_t = pred[1:, :, :] - pred[:-1, :, :], target[1:, :, :] - target[:-1, :, :]
    return float(0.5 * (np.mean(np.abs(dx_p - dx_t)) + np.mean(np.abs(dy_p - dy_t))))


def _saturation(rgb: np.ndarray) -> np.ndarray:
    vmax, vmin = rgb.max(axis=2), rgb.min(axis=2)
    return (vmax - vmin) / np.maximum(vmax, 1e-4)


def _masked_mean(values: np.ndarray, mask: np.ndarray, minimum: int = 16) -> float | None:
    if int(mask.sum()) < minimum:
        return None
    return float(values[mask].mean())


def _flat_dark_noise(pred_l: np.ndarray, target_l: np.ndarray, input_l: np.ndarray) -> dict:
    """High-frequency luma diagnostic on dark, locally flat input pixels.

    Flat-dark pixels have input luma below 0.2 and a 3x3 absolute high-pass
    magnitude below 0.02.  The statistic is descriptive: it measures local
    high-frequency energy, not calibrated sensor noise.
    """
    input_hf = input_l - _box3(input_l)
    mask = (input_l < 0.2) & (np.abs(input_hf) < 0.02)
    if int(mask.sum()) < 16:
        return {
            "flat_dark_fraction": float(mask.mean()),
            "flat_dark_output_hf_std": None,
            "flat_dark_target_hf_std": None,
            "flat_dark_noise_ratio_output_to_target": None,
            "flat_dark_noise_excess": None,
        }
    pred_hf = pred_l - _box3(pred_l)
    target_hf = target_l - _box3(target_l)
    pstd = float(np.std(pred_hf[mask]))
    tstd = float(np.std(target_hf[mask]))
    return {
        "flat_dark_fraction": float(mask.mean()),
        "flat_dark_output_hf_std": pstd,
        "flat_dark_target_hf_std": tstd,
        "flat_dark_noise_ratio_output_to_target": float(pstd / max(tstd, 1e-4)),
        "flat_dark_noise_excess": float(pstd - tstd),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device, dataset_name: str = "unknown") -> Dict[str, object]:
    model.eval()
    per_image = []
    pooled_sum_sq, pooled_count = 0.0, 0
    for x, y, ids in loader:
        output = model(x.to(device))
        pred = output[0] if isinstance(output, tuple) else output
        pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
        target = y.numpy().transpose(0, 2, 3, 1).clip(0, 1)
        inp = x.numpy().transpose(0, 2, 3, 1).clip(0, 1)
        if isinstance(ids, str):
            ids = [ids]
        for i, item_id in enumerate(ids):
            p, t, z = pred[i], target[i], inp[i]
            p_l = p.mean(axis=2)
            t_l = t.mean(axis=2)
            z_l = z.mean(axis=2)
            dark = z_l < 0.2
            bright = z_l > 0.8
            de = float(deltaE_ciede2000(rgb2lab(p), rgb2lab(t)).mean())
            ss = float(structural_similarity(p, t, channel_axis=2, data_range=1.0))
            sat_p, sat_t = _saturation(p), _saturation(t)
            diagnostics = _flat_dark_noise(p_l, t_l, z_l)
            row = {
                    "id": str(item_id),
                    "psnr_db": _psnr(p, t),
                    "ssim": ss,
                    "deltaE2000": de,
                    "mae": float(np.mean(np.abs(p - t))),
                    "dark_input_fraction": float(dark.mean()),
                    "bright_input_fraction": float(bright.mean()),
                    "dark_psnr_db": _masked_psnr(p, t, dark),
                    "bright_psnr_db": _masked_psnr(p, t, bright),
                    "output_clip_fraction_luma_gt_0.98": float((p_l > 0.98).mean()),
                    "output_clip_fraction_channel_gt_0.995": float((p > 0.995).mean()),
                    "input_mean_luma": float(z_l.mean()),
                    "output_mean_luma": float(p_l.mean()),
                    "target_mean_luma": float(t_l.mean()),
                    "gradient_l1": _gradient_l1(p, t),
                    "saturation_mae": float(np.mean(np.abs(sat_p - sat_t))),
                    "output_input_residual_l1": float(np.mean(np.abs(p - z))),
                    "dark_output_input_residual_l1": _masked_mean(np.mean(np.abs(p - z), axis=2), dark),
                    "bright_output_input_residual_l1": _masked_mean(np.mean(np.abs(p - z), axis=2), bright),
                    "dark_luma_mae": _masked_mean(np.abs(p_l - t_l), dark),
                    "bright_luma_mae": _masked_mean(np.abs(p_l - t_l), bright),
                }
            row.update(diagnostics)
            per_image.append(row)
        pooled_sum_sq += float(((pred - target) ** 2).sum())
        pooled_count += int(pred.size)
    if not per_image:
        raise ValueError("empty evaluator")
    keys = [
        "psnr_db", "ssim", "deltaE2000", "mae", "dark_psnr_db", "bright_psnr_db",
        "output_clip_fraction_luma_gt_0.98", "output_clip_fraction_channel_gt_0.995",
        "input_mean_luma", "output_mean_luma", "target_mean_luma", "gradient_l1",
        "saturation_mae", "output_input_residual_l1", "dark_output_input_residual_l1",
        "bright_output_input_residual_l1", "dark_luma_mae", "bright_luma_mae",
        "flat_dark_fraction", "flat_dark_output_hf_std", "flat_dark_target_hf_std",
        "flat_dark_noise_ratio_output_to_target", "flat_dark_noise_excess",
    ]
    macro = {}
    for key in keys:
        vals = [r[key] for r in per_image if r[key] is not None]
        macro[key] = float(np.mean(vals)) if vals else None
        macro[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
    # Pooled PSNR is computed over all pixels rather than averaging dB values.
    macro["pooled_psnr_db"] = float(-10.0 * np.log10(max(pooled_sum_sq / max(pooled_count, 1), 1e-10)))
    return {"dataset": dataset_name, "items": len(per_image), "macro": macro, "per_image": per_image}
