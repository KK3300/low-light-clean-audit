"""Protocol-matched comparison on LOL-v1 eval15 (the manuscript's held-out test set).

Every entry is scored with a single metric implementation so the table is
internally consistent.  Learned baselines use the *official* published
checkpoints fetched from the authors' repositories:

    Zero-DCE     Li-Chongyi/Zero-DCE          snapshots/Epoch99.pth          (79,416 params)
    Zero-DCE++   Li-Chongyi/Zero-DCE_extension snapshots_Zero_DCE++/Epoch99.pth (10,561 params)

Both checkpoints load into the reimplementations in `zero_dce.py` with zero
missing and zero unexpected keys, which is the evidence that the architectures
match the published ones.

Reporting boundary
------------------
Zero-DCE and Zero-DCE++ are zero-reference methods trained by their authors on
their own unpaired data.  Our models are trained supervised on the project's
clean LSD+LOL split.  The table is therefore an inference-protocol-matched
comparison (same test pairs, same metrics, same resolution), *not* a
training-protocol-matched one, and the manuscript must say so.
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

ROOT = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8")
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT.parent / "darkdriving"))

from zero_dce import DCENet, DCENetPlusPlus, count_params  # noqa: E402
from task4_nonlearned_vs_learned import CLASSICAL, CKPT, load_learned, metrics_for  # noqa: E402

PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
EVAL15 = PROJECT / "experiments" / "datasets" / "LOL_v1" / "extracted" / "eval15"
REF = ROOT / "third_party" / "official_ref"
LPIPS_CKPT = PROJECT / "experiments" / "rcm_dce" / "lpips_alex.pth"


def load_pairs(limit=0):
    lows = sorted(EVAL15.joinpath("low").glob("*.png"))
    if limit:
        lows = lows[:limit]
    out = []
    for lp in lows:
        hp = EVAL15 / "high" / lp.name
        if not hp.exists():
            continue
        lo = np.asarray(Image.open(lp).convert("RGB"), dtype=np.float32) / 255.0
        hi = np.asarray(Image.open(hp).convert("RGB"), dtype=np.float32) / 255.0
        if lo.shape == hi.shape:
            out.append((lp.stem, lo, hi))
    return out


def aggregate(rows):
    keys = rows[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        if not vals:
            continue
        agg[k] = float(np.mean(vals))
        agg[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


def try_lpips(device):
    """LPIPS is optional: only used when the AlexNet backbone weights exist."""
    for cand in [LPIPS_CKPT, PROJECT / "experiments" / "rcm_dce" / "lpips" / "alex.pth"]:
        if cand.exists():
            try:
                import lpips  # noqa
                net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                return net, str(cand)
            except Exception as e:
                return None, "lpips unavailable: %s" % e
    return None, "no local alex LPIPS checkpoint"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "baselines" / "comparison_lol_eval15.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.limit)
    print("LOL-v1 eval15 pairs: %d  device=%s" % (len(pairs), device), flush=True)
    if not pairs:
        return 1

    results: dict[str, dict] = {}

    # ---------------- classical / non-learned (reused from Task 4 Part A)
    partA = ROOT.parent / "darkdriving" / "results_task4" / "task4_partA_lol_eval15.json"
    if partA.exists():
        pa = json.loads(partA.read_text(encoding="utf-8"))["results"]
        for name in ("input", "he", "clahe", "gamma", "ssr", "msr", "msrcr", "lime"):
            if name in pa:
                row = {k: v for k, v in pa[name].items() if not k.endswith("_sd")}
                row["family"] = "non-learned"
                row["note"] = "scored with the same metric implementation (Task 4 Part A)"
                results[name] = row
        print("imported %d non-learned baselines from Task 4 Part A" % len(results), flush=True)

    # ---------------- official pretrained zero-reference models
    official = [
        ("Zero-DCE (official ckpt)", DCENet(32, 8), REF / "ZeroDCE_Epoch99.pth", 79416),
        ("Zero-DCE++ (official ckpt)", DCENetPlusPlus(32, 8, 12), REF / "ZeroDCEpp_Epoch99.pth", 10561),
    ]
    for name, model, ckpt, target in official:
        if not ckpt.exists():
            print("  SKIP %s: checkpoint missing (%s)" % (name, ckpt))
            continue
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print("  SKIP %s: state_dict mismatch (missing=%d unexpected=%d)"
                  % (name, len(missing), len(unexpected)))
            continue
        model = model.to(device).eval()
        n = count_params(model)
        rows = []
        t0 = time.time()
        with torch.no_grad():
            for pid, lo, hi in pairs:
                x = torch.from_numpy(lo).permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(x)
                pred = (out[0] if isinstance(out, tuple) else out)[0].permute(1, 2, 0).cpu().numpy()
                rows.append(metrics_for(pred, hi, lo))
        agg = aggregate(rows)
        agg.update({
            "family": "learned (zero-reference, official checkpoint)",
            "parameters": n,
            "published_parameters": target,
            "param_match": n == target,
            "checkpoint": ckpt.name,
            "seconds_per_image": round((time.time() - t0) / max(1, len(pairs)), 4),
            "n_images": len(pairs),
            "training_protocol": "author-trained, zero-reference, unpaired data",
        })
        results[name] = agg
        print("  %-28s PSNR %6.3f  SSIM %6.4f  dE00 %7.3f  params %d"
              % (name, agg["psnr_db"], agg["ssim"], agg["deltaE2000"], n), flush=True)

    # ---------------- our own models (same protocol, supervised on clean split)
    for name, ck in CKPT.items():
        if not ck.exists():
            print("  SKIP %s: %s" % (name, ck))
            continue
        model, sig, n = load_learned(ck, device)
        rows = []
        t0 = time.time()
        with torch.no_grad():
            for pid, lo, hi in pairs:
                x = torch.from_numpy(lo).permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(x)
                pred = (out[0] if isinstance(out, tuple) else out)[0].permute(1, 2, 0).cpu().numpy()
                rows.append(metrics_for(pred, hi, lo))
        agg = aggregate(rows)
        agg.update({
            "family": "learned (supervised on clean LSD+LOL split)",
            "parameters": n,
            "checkpoint": str(ck),
            "seconds_per_image": round((time.time() - t0) / max(1, len(pairs)), 4),
            "n_images": len(pairs),
            "training_protocol": "supervised paired training on the project's clean split",
        })
        results[name] = agg
        print("  %-28s PSNR %6.3f  SSIM %6.4f  dE00 %7.3f  params %d"
              % (name, agg["psnr_db"], agg["ssim"], agg["deltaE2000"], n), flush=True)

    payload = {
        "protocol": "protocol_matched_comparison_lol_eval15",
        "test_set": "LOL-v1 official eval15 (held-out; never used for training)",
        "n_pairs": len(pairs),
        "metric_implementation": "same code path as the manuscript's reported numbers",
        "boundary": ("inference-protocol matched (same pairs, metrics, resolution); "
                     "training protocols differ and are recorded per row"),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", args.out)

    # ---------------- print the paper table
    print("\n" + "=" * 96)
    print("%-30s %9s %8s %9s %11s %10s" % ("method", "PSNR", "SSIM", "CIEDE2000", "clip_luma", "params"))
    print("-" * 96)
    order = sorted(results.items(), key=lambda kv: -kv[1].get("psnr_db", 0))
    for name, r in order:
        print("%-30s %9.3f %8.4f %9.3f %11.5f %10s"
              % (name, r.get("psnr_db", float("nan")), r.get("ssim", float("nan")),
                 r.get("deltaE2000", float("nan")),
                 r.get("output_clip_fraction_luma_gt_0.98", float("nan")),
                 r.get("parameters", "-")))
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
