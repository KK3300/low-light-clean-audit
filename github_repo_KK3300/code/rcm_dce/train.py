"""Train a Stage 2/3 model with validation-only checkpoint observation.

The script accepts a validation manifest but never accepts a test manifest.
An independent evaluator must be called after the run is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HAM_DIR = ROOT / "ham_dce"
if str(HAM_DIR) not in sys.path:
    sys.path.insert(0, str(HAM_DIR))

from rcm_dce.data import LowLightFolder, LowLightManifest, PairedManifest, seed_worker
from rcm_dce.estimate import resource_signature
from rcm_dce.metrics import evaluate_loader
from rcm_dce.model import RCMCurveNet, count_parameters, model_signature


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


KNOWN_IDENTITY_CONTAMINATED_MANIFESTS = {
    "88e48d9194bbb38940047867e2da182d4f1afea8f474b22953ce9c36ab7edfb4": "combined LSD+LOL train (355 byte-identical LSD input/GT pairs)",
    "c53e6b46d1a355eb26ee8eaf51819be2c285d0b9253fc19c845dc50f7d593dfb": "combined LSD+LOL validation (18 byte-identical LSD input/GT pairs)",
    "0b55e6941c5005f9c55e5a7486c1ec3f47f0de8c27ddc803e2e0cfcb9852833a": "LSD 10K train (355 byte-identical pairs)",
    "2cf32eada9e1f7028e4989121c5b6b81abfbfc82733cea81387936a785ea4e9b": "LSD 10K validation (79 byte-identical pairs)",
}


@torch.no_grad()
def fast_validate(model, loader, device: torch.device) -> dict:
    """Torch-only validation used for per-epoch selection.

    It computes the same macro per-image PSNR used by the full evaluator, but
    avoids CPU Lab/SSIM conversion on every epoch. Full colour/region metrics
    are evaluated once on the frozen candidate checkpoint by ``evaluate.py``.
    """
    model.eval()
    psnr_vals, mae_vals = [], []
    group_psnr: dict[str, list[float]] = {}
    pooled_sum, pooled_count = 0.0, 0
    for x, y, ids in loader:
        output = model(x.to(device))
        pred = output[0] if isinstance(output, tuple) else output
        diff = pred.detach().cpu() - y
        mse = diff.square().flatten(1).mean(dim=1).clamp_min(1e-10)
        batch_psnr = (-10.0 * torch.log10(mse)).tolist()
        psnr_vals.extend(batch_psnr)
        mae_vals.extend(diff.abs().flatten(1).mean(dim=1).tolist())
        if isinstance(ids, str):
            ids = [ids]
        for item_id, score in zip(ids, batch_psnr):
            item_id = str(item_id)
            group = item_id.split(":", 1)[0] if ":" in item_id else "default"
            group_psnr.setdefault(group, []).append(float(score))
        pooled_sum += float(diff.square().sum())
        pooled_count += int(diff.numel())
    group_macro = {key: float(np.mean(values)) for key, values in sorted(group_psnr.items())}
    domain_macro = float(np.mean(list(group_macro.values()))) if group_macro else float(np.mean(psnr_vals))
    return {
        "psnr_db": float(np.mean(psnr_vals)),
        "psnr_db_sd": float(np.std(psnr_vals, ddof=1)) if len(psnr_vals) > 1 else 0.0,
        "mae": float(np.mean(mae_vals)),
        "pooled_psnr_db": float(-10.0 * np.log10(max(pooled_sum / max(pooled_count, 1), 1e-10))),
        "items": len(psnr_vals),
        "group_psnr_db": group_macro,
        "domain_macro_psnr_db": domain_macro,
        "selection_metrics": "torch_macro_psnr; full DeltaE/SSIM/region metrics deferred to frozen evaluator",
    }


def photometric_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 0.6,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Published PAL control from arXiv:2604.08172, Appendix D.

    The per-image 3x3 colour matrix and bias are solved in closed form from a
    4x4 design system and detached before the aligned L1 residual is formed.
    The solve is forced to FP32 even when the surrounding training uses AMP.
    PAL is prior art and is logged as an objective control, not a new module.
    """
    if weight <= 0:
        return pred.new_zeros(())
    with torch.amp.autocast("cuda", enabled=False):
        p = pred.float()
        t = target.float()
        b, _, h, w = p.shape
        pixels = p.permute(0, 2, 3, 1).reshape(b, -1, 3)
        targets = t.permute(0, 2, 3, 1).reshape(b, -1, 3)
        design = torch.cat([pixels, pixels.new_ones(b, pixels.shape[1], 1)], dim=-1)
        xtx = design.transpose(1, 2) @ design
        xtt = design.transpose(1, 2) @ targets
        eye = torch.eye(4, device=p.device, dtype=p.dtype).unsqueeze(0)
        transform = torch.linalg.solve(xtx + float(eps) * eye, xtt).transpose(1, 2).detach()
        flat = torch.cat([p.reshape(b, 3, -1), p.new_ones(b, 1, h * w)], dim=1)
        aligned = (transform @ flat).reshape(b, 3, h, w)
        return float(weight) * F.l1_loss(aligned, t)


def supervised_loss(
    target: torch.Tensor,
    enhanced: torch.Tensor,
    curve: torch.Tensor,
    pal_weight: float = 0.0,
    pal_eps: float = 1e-3,
    log_domain: bool = False,
    log_eps: float = 1e-3,
) -> torch.Tensor:
    """Small paired upper-bound objective for the same curve model.

    This branch is explicitly labelled supervised and is never mixed with the
    zero-reference main objective.  The gradient term discourages a purely
    pointwise brightness fit while the curve smoothness term keeps the output
    interpretable.

    ``log_domain`` adds an *additional* log-space L1 term alongside the linear
    one.  It is off by default so that every previously reported number is
    reproduced bit-for-bit; the Stage 6 ablation turns it on explicitly.
    The motivation is that a linear L1 treats an absolute error of 0.05 the same
    way at luma 0.10 and at luma 0.60, whereas the perceptual cost of dark-region
    error is much higher; a log-space term weights dark pixels relatively more.
    """
    l1 = F.smooth_l1_loss(enhanced, target, beta=0.02)
    gx_e, gx_t = enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_e, gy_t = enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :]
    grad = F.l1_loss(gx_e, gx_t) + F.l1_loss(gy_e, gy_t)
    tv = (curve[:, :, 1:, :] - curve[:, :, :-1, :]).abs().mean() + (curve[:, :, :, 1:] - curve[:, :, :, :-1]).abs().mean()
    pal = photometric_alignment_loss(enhanced, target, pal_weight, pal_eps)
    loss = l1 + 0.1 * grad + 0.01 * tv + pal
    if log_domain:
        log_e = torch.log(enhanced.clamp_min(log_eps))
        log_t = torch.log(target.clamp_min(log_eps))
        loss = loss + 0.1 * F.smooth_l1_loss(log_e, log_t, beta=0.02)
    return loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--train-manifest", default="")
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", choices=["plain", "gap_gate", "variance_gate", "bright_gate", "curve_residual", "rcm", "output_residual", "rcm_output", "budgeted_output", "ms_refine", "ms_refine_uncond", "ms_refine_dark", "ms_refine_reliability"], default="plain")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--conv-mode", choices=["standard", "depthwise"], default="standard")
    ap.add_argument("--scale-factor", type=int, default=12)
    ap.add_argument("--curve-steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-batch-size", type=int, default=1)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--full-val-every", type=int, default=0, help="run slow full validation every N epochs; 0 disables")
    ap.add_argument("--objective", choices=["zero_reference", "paired_l1", "paired_log_l1"], default="zero_reference")
    ap.add_argument("--domain-balanced-sampling", action="store_true", help="inverse-frequency sampling by source_dataset/id namespace")
    ap.add_argument("--domain-balance-exponent", type=float, default=1.0, help="sample weight=count^(-exponent); 1 is equal domain mass, 0 is raw sampling")
    ap.add_argument("--selection-metric", choices=["macro_psnr", "domain_macro_psnr"], default="macro_psnr")
    ap.add_argument("--pal-weight", type=float, default=0.0, help="published PAL objective-control weight; 0 disables")
    ap.add_argument("--pal-eps", type=float, default=1e-3)
    ap.add_argument("--allow-contaminated-identity-pairs", action="store_true", help="legacy pilot only; final training must leave this disabled")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_manifest = Path(args.val_manifest).resolve()
    train_root = Path(args.train_root).resolve()
    train_manifest = Path(args.train_manifest).resolve() if args.train_manifest else None
    manifest_hashes = {
        "train": sha256_file(train_manifest) if train_manifest else None,
        "validation": sha256_file(val_manifest),
    }
    contaminated = {
        role: KNOWN_IDENTITY_CONTAMINATED_MANIFESTS[digest]
        for role, digest in manifest_hashes.items()
        if digest in KNOWN_IDENTITY_CONTAMINATED_MANIFESTS
    }
    if contaminated and not args.allow_contaminated_identity_pairs:
        raise ValueError(
            "Refusing known identity-contaminated manifests: "
            + json.dumps(contaminated, ensure_ascii=False)
            + ". Use clean_v1 manifests; override is legacy-pilot only."
        )
    if args.objective in ("paired_l1", "paired_log_l1") and train_manifest:
        train_ds = PairedManifest(train_manifest, crop_size=args.crop_size, augment=True)
    else:
        train_ds = LowLightManifest(train_manifest, crop_size=args.crop_size, augment=True) if train_manifest else LowLightFolder(train_root, crop_size=args.crop_size, augment=True)
    val_ds = PairedManifest(val_manifest, crop_size=0)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = None
    sampler_report = None
    if args.domain_balanced_sampling:
        if not isinstance(train_ds, PairedManifest):
            raise ValueError("domain-balanced sampling requires a paired train manifest")
        counts: dict[str, int] = {}
        for group in train_ds.groups:
            counts[group] = counts.get(group, 0) + 1
        if len(counts) < 2:
            raise ValueError(f"domain-balanced sampling needs >=2 groups, got {counts}")
        if not (0.0 <= args.domain_balance_exponent <= 1.0):
            raise ValueError("domain-balance-exponent must be in [0,1]")
        exponent = float(args.domain_balance_exponent)
        weights = torch.tensor([counts[group] ** (-exponent) for group in train_ds.groups], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True, generator=generator)
        group_mass = {group: counts[group] ** (1.0 - exponent) for group in counts}
        mass_total = sum(group_mass.values())
        sampler_report = {
            "mode": "tempered_inverse_frequency_replacement",
            "exponent": exponent,
            "groups": counts,
            "expected_group_mass": {group: group_mass[group] / mass_total for group in counts},
        }
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    model = RCMCurveNet(args.variant, args.width, args.conv_mode, args.scale_factor, args.curve_steps).to(device)
    if args.conv_mode == "standard" and args.width == 128:
        expected_backbone = 1043331
        actual_backbone = count_parameters(RCMCurveNet("plain", 128, "standard", args.scale_factor, args.curve_steps))
        if actual_backbone != expected_backbone:
            raise AssertionError(f"plain-1M parameter regression: {actual_backbone} != {expected_backbone}")
    from ham_dce import ZeroReferenceLoss  # device-safe modern objective, no legacy CUDA constants
    objective = ZeroReferenceLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "protocol": "stage2_val_only_v1",
        "model": model_signature(model),
        "data": {"train_root": str(train_root), "train_manifest": str(train_manifest) if train_manifest else None, "val_manifest": str(val_manifest), "train_manifest_sha256": manifest_hashes["train"], "val_manifest_sha256": manifest_hashes["validation"], "identity_contamination_override": bool(args.allow_contaminated_identity_pairs), "crop_size": args.crop_size, "batch_size": args.batch_size, "val_batch_size": args.val_batch_size, "num_workers": args.num_workers, "value_range": "[0,1] RGB", "sampler": sampler_report},
        "optimization": {"objective": args.objective, "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "epochs": args.epochs, "seed": args.seed, "amp": use_amp, "pal_control": {"enabled": args.pal_weight > 0, "weight": args.pal_weight, "eps": args.pal_eps, "source": "arXiv:2604.08172 Appendix D"}},
        "selection": {"mode": args.selection_metric, "tie_break": "earlier_epoch", "test_read_during_training": False},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": str(device), "platform": platform.platform()},
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    start = time.perf_counter()
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break
            if args.objective in ("paired_l1", "paired_log_l1"):
                inp, target, _ = batch
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            else:
                inp = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                enhanced, curve = model(inp)
                if args.objective in ("paired_l1", "paired_log_l1"):
                    total_loss = supervised_loss(target, enhanced, curve, args.pal_weight, args.pal_eps,
                                     log_domain=(args.objective == "paired_log_l1"))
                else:
                    parts = objective(inp, enhanced, curve)
                    total_loss = parts["total"]
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(total_loss.detach().cpu()))
        fast_val = fast_validate(model, val_loader, device)
        full_val = None
        if args.full_val_every and epoch % args.full_val_every == 0:
            full_val = evaluate_loader(model, val_loader, device, dataset_name="scene_disjoint_val")["macro"]
        rec = {"epoch": epoch, "train_steps": len(losses), "train_loss_mean": float(np.mean(losses)) if losses else None, "validation_fast": fast_val, "validation_full": full_val, "checkpoint": str((out / f"{args.variant}_epoch{epoch:03d}.pth").resolve())}
        payload = {"state_dict": model.state_dict(), "model_signature": model_signature(model), "config": config, "epoch": epoch}
        torch.save(payload, out / f"{args.variant}_epoch{epoch:03d}.pth")
        records.append(rec)
        selection_value = fast_val["domain_macro_psnr_db"] if args.selection_metric == "domain_macro_psnr" else fast_val["psnr_db"]
        rec["selection_value"] = float(selection_value)
        rec["selection_metric"] = args.selection_metric
        key = (-float(selection_value), epoch)
        if best is None or key < best[0]:
            best = (key, epoch)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - start
    summary = {"protocol": "stage2_val_only_v1", "variant": args.variant, "seed": args.seed, "device": str(device), "objective": args.objective, "model_signature": model_signature(model), "resources": resource_signature(model), "elapsed_seconds": elapsed, "records": records, "selected_validation_epoch": best[1] if best else None, "selection_rule": f"highest {args.selection_metric}; full colour metrics evaluated after freeze", "test_manifest_read": False, "train_root_sha256": None, "train_manifest_sha256": sha256_file(train_manifest) if train_manifest else None, "val_manifest_sha256": sha256_file(val_manifest)}
    (out / "epoch_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
