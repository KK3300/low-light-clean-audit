"""E4: true cross-domain experiment (train on LSD only, select and test on LOL).

Why this experiment exists
--------------------------
The manuscript's "cross-domain" claim currently rests on two evaluation
protocols, one of which (NTIRE val24) is LSD-derived. That is a protocol claim,
not a domain claim, and a reviewer is entitled to say the model has never been
measured on a family it did not train on. This script closes that gap.

Protocol
--------
Identical to the frozen Stage-4 recipe in every respect except the data source:

    --train-manifest  crossdomain/train_lsd_only.csv     (7,652 LSD pairs)
    --val-manifest    crossdomain/val_lol_only.csv       (97 LOL pairs)
    --domain-balanced-sampling is REMOVED (only one source domain exists)

Everything else is byte-identical to run_stage4_clean_balanced_3seed.ps1:
width 125, standard convs, scale factor 12, 8 curve steps, 3 epochs x 100 steps,
batch 32, crop 256, Adam lr 1e-4, wd 1e-4, grad clip 0.1, AMP,
objective paired_l1, selection metric domain-macro PSNR.

An in-domain control (train on the 388 LOL training pairs only) is included so
the LSD->LOL drop can be attributed to the domain shift rather than to model
capacity or optimisation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
PY = PROJECT / "experiments" / ".venv" / "Scripts" / "python.exe"
TRAIN = PROJECT / "experiments" / "rcm_dce" / "train.py"
CD = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8\crossdomain")
RESULTS = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8\crossdomain\results")


def run(variant: str, seed: int, train_manifest: Path, val_manifest: Path,
        out_dir: Path, balanced: bool, log_domain: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), str(TRAIN),
        "--train-root", str(PROJECT / "experiments" / "datasets" / "LSD_DLL_subset_10000" / "input"),
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--width", "125", "--conv-mode", "standard", "--scale-factor", "12", "--curve-steps", "8",
        "--epochs", "3", "--batch-size", "32", "--val-batch-size", "1", "--crop-size", "256",
        "--seed", str(seed), "--lr", "1e-4", "--weight-decay", "1e-4", "--grad-clip", "0.1",
        "--num-workers", "0", "--max-steps-per-epoch", "100", "--device", "auto", "--amp",
        "--objective", "paired_log_l1" if log_domain else "paired_l1",
        "--selection-metric", "domain_macro_psnr",
        "--variant", variant,
        "--output-dir", str(out_dir),
    ]
    if balanced:
        cmd += ["--domain-balanced-sampling", "--domain-balance-exponent", "1.0"]

    t0 = time.time()
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = (r.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print("   " + line, flush=True)
    if r.returncode != 0:
        print("   FAILED (rc=%d)" % r.returncode, flush=True)
        print("   stderr:", (r.stderr or "")[-800:], flush=True)
    return {"variant": variant, "out_dir": str(out_dir), "returncode": r.returncode,
            "seconds": round(dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--phase", default="cross", choices=["cross", "in_domain", "log_domain", "all"])
    args = ap.parse_args()

    if not TRAIN.exists():
        print("training entry not found:", TRAIN)
        return 1

    jobs = []
    if args.phase in ("cross", "all"):
        # the headline cross-domain run: LSD -> LOL
        for v in ("plain", "ms_refine_uncond"):
            jobs.append((v, CD / "train_lsd_only.csv", CD / "val_lol_only.csv",
                         RESULTS / "lsd_to_lol" / ("seed%d" % args.seed) / (v + "125"),
                         False, False))
    if args.phase in ("in_domain", "all"):
        # control: same architecture trained and selected on LOL only
        jobs.append(("plain", CD / "train_lol_only.csv", CD / "val_lol_only.csv",
                     RESULTS / "lol_only" / ("seed%d" % args.seed) / "plain125",
                     False, False))
    if args.phase in ("log_domain", "all"):
        # E5: log-domain L1 objective on the standard clean mixed manifest
        for v in ("plain", "ms_refine_uncond"):
            jobs.append((v,
                         PROJECT / "experiments" / "datasets" / "combined_lsd_lol_clean_v1" / "train_pairs.csv",
                         PROJECT / "experiments" / "datasets" / "combined_lsd_lol_clean_v1" / "val_pairs.csv",
                         RESULTS / "log_l1" / ("seed%d" % args.seed) / (v + "125"),
                         True, True))

    summary = []
    for variant, tr, va, out, balanced, logd in jobs:
        summary.append(run(variant, args.seed, tr, va, out, balanced, logd))

    outp = RESULTS / ("run_summary_%s_seed%d.json" % (args.phase, args.seed))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", outp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
