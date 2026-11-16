"""Multi-seed downstream evaluation: 3 detectors x 7 front-ends.

Reuses the front-end images already materialised under `work_final/`, so no
enhancement is recomputed.  Reports the per-seed table, the mean and sample
standard deviation across detector seeds, and -- most importantly -- the
seed variance of the refinement-branch gain (candidate minus backbone), which
is the quantity the manuscript currently reports as a single point estimate.

Each detector was fine-tuned on the same bright split with only the training
seed changed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

DARK = Path(r"C:\Users\33848\论文投稿\darkdriving")
IMG_ROOT = DARK / "work_final"
SEED_ROOT = DARK / "work_seeds" / "runs"
OUT = DARK / "results_multiseed"

FRONTENDS = ["none", "gamma", "clahe", "lime", "msrcr", "plain125", "ms_refine_uncond"]

# seed -> detector weights
DETECTORS = {
    20260830: DARK / "work_partB_finetune" / "runs" / "night_ft" / "weights" / "best.pt",
    20260831: SEED_ROOT / "seed20260831" / "weights" / "best.pt",
    20260832: SEED_ROOT / "seed20260832" / "weights" / "best.pt",
}

LABELS_DIR = Path(r"C:\Users\33848\Documents\ChatGPT\new\experiments\datasets"
                  r"\DarkDriving_det_release\Det_data\night\labels\val")


def make_view(frontend: str, work: Path) -> Path:
    """Point a YOLO dataset yaml at the pre-enhanced images for one front-end."""
    root = work / ("eval_" + frontend)
    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for f in LABELS_DIR.glob("*.txt"):
        t = lbl_dir / f.name
        if not t.exists():
            t.hardlink_to(f)
    cfg = {"path": str(root), "train": "images/val", "val": "images/val",
           "nc": 1, "names": {0: "car"}}
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # the enhanced images live in the Task-4 workspace; symlink the directory once
    src = IMG_ROOT / ("night_val__" + frontend) / "images" / "val"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        try:
            img_dir.rmdir()
        except OSError:
            pass
        try:
            img_dir.hardlink_to(src)
        except Exception:
            import shutil
            shutil.copytree(src, img_dir, dirs_exist_ok=True)
    return yml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frontends", default=",".join(FRONTENDS))
    ap.add_argument("--seeds", default="", help="default: all available detectors")
    args = ap.parse_args()

    from ultralytics import YOLO

    work = DARK / "work_multiseed"
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(DETECTORS))
    frontends = [f.strip() for f in args.frontends.split(",") if f.strip()]

    results: dict[str, dict] = {}
    for seed in seeds:
        w = DETECTORS.get(seed)
        if not w or not w.exists():
            print("SKIP seed %d: weights missing (%s)" % (seed, w), flush=True)
            continue
        model = YOLO(str(w))
        results[str(seed)] = {}
        for fe in frontends:
            src = IMG_ROOT / ("night_val__" + fe) / "images" / "val"
            if not src.exists():
                print("  SKIP %s: no images at %s" % (fe, src), flush=True)
                continue
            yml = make_view(fe, work / ("seed%d" % seed))
            t0 = time.time()
            m = model.val(data=str(yml), imgsz=args.imgsz, device="0", split="val",
                          verbose=False, plots=False, project=str(work / "runs"),
                          name="s%d_%s" % (seed, fe))
            results[str(seed)][fe] = {
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                "P": float(m.box.mp), "R": float(m.box.mr),
            }
            print("seed %d  %-18s mAP50 %.4f  mAP50-95 %.4f  (%.0fs)"
                  % (seed, fe, m.box.map50, m.box.map, time.time() - t0), flush=True)
            (OUT / "multiseed_raw.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- aggregate
    agg = {}
    for fe in frontends:
        vals_50 = [results[s][fe]["mAP50"] for s in results if fe in results[s]]
        vals_95 = [results[s][fe]["mAP50_95"] for s in results if fe in results[s]]
        pr = [results[s][fe]["P"] for s in results if fe in results[s]]
        rc = [results[s][fe]["R"] for s in results if fe in results[s]]
        if not vals_50:
            continue
        agg[fe] = {
            "n_seeds": len(vals_50),
            "mAP50_mean": statistics.mean(vals_50),
            "mAP50_sd": statistics.stdev(vals_50) if len(vals_50) > 1 else 0.0,
            "mAP50_per_seed": vals_50,
            "mAP50_95_mean": statistics.mean(vals_95),
            "mAP50_95_sd": statistics.stdev(vals_95) if len(vals_95) > 1 else 0.0,
            "P_mean": statistics.mean(pr), "R_mean": statistics.mean(rc),
        }

    # ---- branch gain per seed (candidate - backbone), paired within a seed
    branch = None
    if "ms_refine_uncond" in agg and "plain125" in agg:
        per = []
        for s in results:
            if "ms_refine_uncond" in results[s] and "plain125" in results[s]:
                per.append(results[s]["ms_refine_uncond"]["mAP50"]
                           - results[s]["plain125"]["mAP50"])
        branch = {
            "definition": "mAP50(ms_refine_uncond) - mAP50(plain125), paired within detector seed",
            "per_seed": per,
            "mean": statistics.mean(per),
            "sd": statistics.stdev(per) if len(per) > 1 else 0.0,
            "n_seeds": len(per),
        }

    # ---- relative to no-enhancement baseline, per seed then averaged
    rel = {}
    if "none" in agg:
        for fe in frontends:
            if fe == "none" or fe not in agg:
                continue
            ds = []
            for s in results:
                if fe in results[s] and "none" in results[s]:
                    b = results[s]["none"]["mAP50"]
                    if b > 0:
                        ds.append(100.0 * (results[s][fe]["mAP50"] - b) / b)
            if ds:
                rel[fe] = {"mean_pct": statistics.mean(ds),
                           "sd_pct": statistics.stdev(ds) if len(ds) > 1 else 0.0,
                           "per_seed_pct": ds}

    payload = {
        "protocol": "multiseed_downstream_detection",
        "note": ("three detectors fine-tuned on the same bright split with only the "
                 "training seed changed; front-end images reused from Task 4"),
        "imgsz": args.imgsz,
        "seeds": list(results.keys()),
        "per_seed": results,
        "aggregate": agg,
        "branch_gain": branch,
        "relative_to_none": rel,
    }
    outp = OUT / "multiseed_summary.json"
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 100)
    print("%-20s %18s %18s %14s" % ("front-end", "mAP50 mean", "mAP50-95 mean", "rel. to none"))
    print("-" * 100)
    for fe in sorted(agg, key=lambda k: -agg[k]["mAP50_mean"]):
        a = agg[fe]
        r = rel.get(fe)
        rs = ("%+.1f%% +- %.1f" % (r["mean_pct"], r["sd_pct"])) if r else "baseline"
        print("%-20s %10.4f +- %-6.4f %10.4f +- %-6.4f %14s"
              % (fe, a["mAP50_mean"], a["mAP50_sd"],
                 a["mAP50_95_mean"], a["mAP50_95_sd"], rs))
    print("=" * 100)
    if branch:
        print("\nbranch gain (candidate - backbone), paired per seed:")
        print("  per seed: %s" % ", ".join("%+.4f" % v for v in branch["per_seed"]))
        print("  mean %.4f +- %.4f mAP50 over %d seeds"
              % (branch["mean"], branch["sd"], branch["n_seeds"]))
    print("\nwrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
