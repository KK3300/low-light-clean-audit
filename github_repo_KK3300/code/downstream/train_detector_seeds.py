"""Train additional detector seeds on the DarkDriving bright split.

The manuscript's downstream result currently rests on a single fine-tuned
detector.  This script produces two more detectors with different training
seeds so the front-end comparison can be reported with a seed standard
deviation instead of a point estimate.

Only the detector seed changes: the image view, epoch count, resolution and
batch size are identical to the original run, so the three detectors differ in
initialisation and batch order alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

DATA = Path(r"C:\Users\33848\Documents\ChatGPT\new\experiments\datasets\DarkDriving_det_release\Det_data")
WORK = Path(r"C:\Users\33848\论文投稿\darkdriving\work_seeds")


def build_view(root: Path, val_n: int) -> Path:
    tr_i, tr_l = root / "images" / "train", root / "labels" / "train"
    va_i, va_l = root / "images" / "val", root / "labels" / "val"
    for d in (tr_i, tr_l, va_i, va_l):
        d.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path, n: int = 0):
        files = sorted(src.glob("*"))
        if n:
            files = files[:n]
        c = 0
        for p in files:
            t = dst / p.name
            if t.exists():
                c += 1
                continue
            if p.suffix.lower() == ".jpg":
                try:
                    t.hardlink_to(p)
                except Exception:
                    shutil.copy2(p, t)
            else:
                shutil.copy2(p, t)
            c += 1
        return c

    ntr = link(DATA / "night" / "images" / "train", tr_i)
    link(DATA / "night" / "labels" / "train", tr_l)
    nva = link(DATA / "night" / "images" / "val", va_i, val_n)
    link(DATA / "night" / "labels" / "val", va_l, val_n)

    cfg = {"path": str(root), "train": "images/train", "val": "images/val",
           "nc": 1, "names": {0: "car"}}
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print("view: %d train / %d val -> %s" % (ntr, nva, yml), flush=True)
    return yml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\33848\论文投稿\yolo11x.pt")
    ap.add_argument("--seeds", default="20260831,20260832")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-n", type=int, default=200)
    args = ap.parse_args()

    from ultralytics import YOLO

    WORK.mkdir(parents=True, exist_ok=True)
    yml = build_view(WORK / "night_single_class", args.val_n)

    summary = {}
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        name = "seed%d" % s
        print("\n=== training detector %s ===" % name, flush=True)
        model = YOLO(args.base)
        t0 = time.time()
        model.train(
            data=str(yml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device="0", workers=4, seed=s, deterministic=True,
            project=str(WORK / "runs"), name=name, exist_ok=True,
            val=True, plots=False, verbose=True, cache=False,
        )
        dt = time.time() - t0
        best = WORK / "runs" / name / "weights" / "best.pt"
        rec = {"seed": s, "seconds": round(dt, 1), "best": str(best), "exists": best.exists()}
        if best.exists():
            m = YOLO(str(best))
            vm = m.val(data=str(yml), imgsz=args.imgsz, device="0", split="val",
                       verbose=False, plots=False, project=str(WORK / "runs"),
                       name=name + "_final")
            rec.update({"mAP50": float(vm.box.map50), "mAP50_95": float(vm.box.map),
                        "P": float(vm.box.mp), "R": float(vm.box.mr)})
            print("%s: mAP50 %.4f  mAP50-95 %.4f  P %.4f  R %.4f  (%.0fs)"
                  % (name, vm.box.map50, vm.box.map, vm.box.mp, vm.box.mr, dt), flush=True)
        summary[name] = rec

    outp = WORK / "seed_training_summary.json"
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
