"""Task 4 Part B (final): fine-tune YOLO11 on DarkDriving night/train, then test
whether low-light enhancement helps that adapted detector on night/val.

Why fine-tune instead of zero-shot
---------------------------------
Zero-shot COCO YOLO11x scored mAP50 = 0.0076 (640) and 0.0042 (1280) on
night/val: the detector does not recognise this domain at all, so it cannot
discriminate between enhancement front-ends.  Fine-tuning on the release's own
training split makes the detector a valid measuring instrument.

What is (and is not) claimed
----------------------------
* night/train is the BRIGHT half of each capture session (mean luma 131.3) and
  night/val is the DARK half (mean luma 47.4).  This is therefore a
  bright-train -> dark-test transfer setting, and the results are reported as
  such, NOT as "low-light detection performance".
* The enhancement front-ends are applied only at test time; the detector is
  trained once on the unenhanced training split, so all arms share one detector.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import yaml

DATA = Path(r"C:\Users\33848\Documents\ChatGPT\new\experiments\datasets\DarkDriving_det_release\Det_data")
WORK = Path(r"C:\Users\33848\论文投稿\darkdriving\work_partB_finetune")


def build_train_view(root: Path, train_n: int = 0, val_n: int = 0) -> Path:
    """Single-class car dataset: train = night/train images, val = night/val images."""
    tr_i = root / "images" / "train"
    tr_l = root / "labels" / "train"
    va_i = root / "images" / "val"
    va_l = root / "labels" / "val"
    for d in (tr_i, tr_l, va_i, va_l):
        d.mkdir(parents=True, exist_ok=True)

    def link(src_dir: Path, dst_dir: Path, n: int):
        files = sorted(src_dir.glob("*"))
        if n:
            files = files[:n]
        c = 0
        for p in files:
            tgt = dst_dir / p.name
            if tgt.exists():
                c += 1
                continue
            if p.suffix.lower() == ".jpg":
                try:
                    tgt.hardlink_to(p)
                except Exception:
                    shutil.copy2(p, tgt)
            else:
                shutil.copy2(p, tgt)
            c += 1
        return c

    ntr = link(DATA / "night" / "images" / "train", tr_i, train_n)
    link(DATA / "night" / "labels" / "train", tr_l, train_n)
    nva = link(DATA / "night" / "images" / "val", va_i, val_n)
    link(DATA / "night" / "labels" / "val", va_l, val_n)

    cfg = {"path": str(root), "train": "images/train", "val": "images/val",
           "nc": 1, "names": {0: "car"}}
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print("train view: %d train imgs, %d val imgs -> %s" % (ntr, nva, yml), flush=True)
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\33848\论文投稿\yolo11x.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=0, help="0 = all 2822")
    ap.add_argument("--val-n", type=int, default=200, help="subset used for periodic val")
    ap.add_argument("--out", default=r"C:\Users\33848\论文投稿\darkdriving\results_task4_finetune")
    args = ap.parse_args()

    from ultralytics import YOLO

    WORK.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    yml = build_train_view(WORK / "night_single_class", args.train_n, args.val_n)

    model = YOLO(args.base)
    t0 = time.time()
    results = model.train(
        data=str(yml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="0",
        workers=4,
        project=str(WORK / "runs"),
        name="night_ft",
        exist_ok=True,
        val=True,
        plots=False,
        verbose=True,
        cache=False,
    )
    dt = time.time() - t0

    best = WORK / "runs" / "night_ft" / "weights" / "best.pt"
    summary = {
        "protocol": "task4_partB_finetune_night_train",
        "base_weights": args.base,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_images": args.train_n or 2822,
        "val_images": args.val_n,
        "train_seconds": round(dt, 1),
        "best_weights": str(best),
        "best_exists": best.exists(),
        "note": ("night/train is the BRIGHT half (mean luma 131.3); night/val is the DARK "
                 "half (mean luma 47.4). This is bright->dark transfer, not low-light training."),
    }
    if best.exists():
        # final check on the same subset used for validation during training
        m = YOLO(str(best))
        vm = m.val(data=str(yml), imgsz=args.imgsz, device="0", split="val",
                   verbose=False, plots=False, project=str(WORK / "runs"), name="ft_final")
        summary["fine_tuned_val"] = {
            "mAP50": float(vm.box.map50), "mAP50_95": float(vm.box.map),
            "P": float(vm.box.mp), "R": float(vm.box.mr),
        }
        print("FINE-TUNED val (unenhanced): mAP50 %.4f mAP50-95 %.4f P %.4f R %.4f"
              % (vm.box.map50, vm.box.map, vm.box.mp, vm.box.mr), flush=True)

        # baseline for comparison: same base model, zero-shot, same subset
        zb = YOLO(args.base)
        zm = zb.val(data=str(yml), imgsz=args.imgsz, device="0", split="val",
                    verbose=False, plots=False, project=str(WORK / "runs"), name="zeroshot_same")
        summary["zero_shot_val"] = {
            "mAP50": float(zm.box.map50), "mAP50_95": float(zm.box.map),
            "P": float(zm.box.mp), "R": float(zm.box.mr),
        }
        print("ZERO-SHOT   val (unenhanced): mAP50 %.4f mAP50-95 %.4f P %.4f R %.4f"
              % (zm.box.map50, zm.box.map, zm.box.mp, zm.box.mr), flush=True)

    (out_dir / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", out_dir / "finetune_summary.json")


if __name__ == "__main__":
    main()
