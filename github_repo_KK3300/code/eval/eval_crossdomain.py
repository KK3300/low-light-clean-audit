"""Evaluate the cross-domain (LSD-trained) checkpoints on LOL-v1 eval15.

This is the measurement that turns the manuscript's "cross-domain" claim from a
protocol statement into a domain statement: the model is trained on LSD only
and never sees LOL, then scored on the LOL-v1 held-out pairs.

For context the script also scores:
  * the in-domain control (trained and selected on LOL only)
  * the main clean models (trained on the mixed LSD+LOL clean manifest)
so the table shows the size of the domain gap rather than a single number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8")
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT.parent / "darkdriving"))

from compare_lol_eval15 import load_pairs, aggregate  # noqa: E402
from task4_nonlearned_vs_learned import load_learned, metrics_for, CKPT  # noqa: E402

CD_RESULTS = ROOT / "crossdomain" / "results"


def find_checkpoint(run_dir: Path) -> Path | None:
    """Pick the epoch the training run selected (highest validation macro PSNR)."""
    if not run_dir.exists():
        return None
    meta = run_dir / "metrics.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            sel = m.get("selected_validation_epoch") or m.get("selected_epoch")
            for cand in run_dir.glob("*epoch%03d*.pth" % int(sel)) if sel else []:
                return cand
        except Exception:
            pass
    cks = sorted(run_dir.glob("*.pth"))
    return cks[-1] if cks else None


def evaluate(ckpt: Path, pairs, device):
    model, sig, n = load_learned(ckpt, device)
    rows = []
    with torch.no_grad():
        for pid, lo, hi in pairs:
            x = torch.from_numpy(lo).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(x)
            pred = (out[0] if isinstance(out, tuple) else out)[0].permute(1, 2, 0).cpu().numpy()
            rows.append(metrics_for(pred, hi, lo))
    agg = aggregate(rows)
    agg["parameters"] = n
    agg["checkpoint"] = str(ckpt)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default=str(CD_RESULTS / "crossdomain_eval_lol_eval15.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs()
    print("LOL-v1 eval15 pairs: %d  device=%s" % (len(pairs), device), flush=True)
    if not pairs:
        return 1

    groups = {
        "cross_domain_lsd_to_lol": {
            "plain125": CD_RESULTS / "lsd_to_lol" / ("seed%d" % args.seed) / "plain125",
            "ms_refine_uncond": CD_RESULTS / "lsd_to_lol" / ("seed%d" % args.seed) / "ms_refine_uncond125",
        },
        "in_domain_lol_only_control": {
            "plain125": CD_RESULTS / "lol_only" / ("seed%d" % args.seed) / "plain125",
        },
        "log_domain_l1_ablation": {
            "plain125": CD_RESULTS / "log_l1" / ("seed%d" % args.seed) / "plain125",
            "ms_refine_uncond": CD_RESULTS / "log_l1" / ("seed%d" % args.seed) / "ms_refine_uncond125",
        },
        "main_clean_mixed": {
            "plain125": CKPT["plain125"],
            "ms_refine_uncond": CKPT["ms_refine_uncond"],
        },
    }

    out: dict[str, dict] = {}
    for gname, entries in groups.items():
        out[gname] = {}
        for label, path in entries.items():
            if not path.exists():
                print("  SKIP %s / %s: %s" % (gname, label, path), flush=True)
                continue
            ck = path if path.suffix == ".pth" else find_checkpoint(path)
            if ck is None or not ck.exists():
                print("  SKIP %s / %s: no checkpoint in %s" % (gname, label, path), flush=True)
                continue
            agg = evaluate(ck, pairs, device)
            out[gname][label] = agg
            print("  %-28s %-18s PSNR %6.3f  SSIM %6.4f  dE00 %7.3f  params %d"
                  % (gname, label, agg["psnr_db"], agg["ssim"], agg["deltaE2000"],
                     agg["parameters"]), flush=True)

    payload = {
        "protocol": "cross_domain_evaluation_lol_eval15",
        "test_set": "LOL-v1 official eval15 (never used for training in any arm)",
        "domain_note": ("cross_domain_lsd_to_lol is trained on LSD pairs only and selected on "
                        "the LOL validation split; it never sees LOL training data"),
        "results": out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
