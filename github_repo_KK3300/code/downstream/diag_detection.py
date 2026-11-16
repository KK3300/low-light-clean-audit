"""Diagnostic: why is frozen YOLO11x mAP ~0 on DarkDriving night/val?

Two hypotheses are tested cheaply on a 40-image slice:
  H1 "domain gap"    -> the detector does not recognise this scene at all
  H2 "resolution"    -> the detector sees it, but boxes are too small at 640

Test A: sweep imgsz at NATIVE resolution, no rescaling (640 / 960 / 1280 / 1600).
Test B: feed an upscaled image (x1.5) so boxes get more pixels at the same imgsz.
Both are inference-only, so they can run in minutes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

DATA = Path(r"C:\Users\33848\Documents\ChatGPT\new\experiments\datasets\DarkDriving_det_release\Det_data")
IMG = DATA / "night" / "images" / "val"
LBL = DATA / "night" / "labels" / "val"
WORK = Path(r"C:\Users\33848\论文投稿\darkdriving\work_diag")
LIMIT = 40

from ultralytics import YOLO

WEIGHTS = Path(r"C:\Users\33848\论文投稿\yolo11x.pt")


def prep(name: str, scale: float = 1.0) -> Path:
    root = WORK / name
    imgs = root / "images" / "val"
    lbls = root / "labels" / "val"
    imgs.mkdir(parents=True, exist_ok=True)
    lbls.mkdir(parents=True, exist_ok=True)
    files = sorted(IMG.glob("*.jpg"))[:LIMIT]
    for p in files:
        tgt = imgs / p.name
        if tgt.exists():
            continue
        if scale == 1.0:
            try:
                tgt.hardlink_to(p)
                continue
            except Exception:
                shutil.copy2(p, tgt)
        else:
            im = Image.open(p).convert("RGB")
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BICUBIC)
            im.save(tgt, quality=95)
    for f in sorted(LBL.glob("*.txt"))[:LIMIT]:
        dst = lbls / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    cfg = {"path": str(root), "train": "images/val", "val": "images/val",
           "nc": 1, "names": {0: "car"}}
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return yml


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    out = {}

    print("=== TEST A: imgsz sweep at native 2448x2048 (no rescale) ===", flush=True)
    yml_native = prep("native", 1.0)
    for imgsz in (640, 960, 1280, 1600):
        try:
            m = model.val(data=str(yml_native), imgsz=imgsz, device="0", split="val",
                          verbose=False, plots=False, project=str(WORK / "runs"),
                          name="native_%d" % imgsz)
            out["native_imgsz_%d" % imgsz] = {
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                "P": float(m.box.mp), "R": float(m.box.mr)}
            print("  native imgsz=%-5d mAP50 %.4f  mAP50-95 %.4f  P %.4f  R %.4f"
                  % (imgsz, m.box.map50, m.box.map, m.box.mp, m.box.mr), flush=True)
        except Exception as exc:
            print("  imgsz=%d failed: %s: %s" % (imgsz, type(exc).__name__, exc), flush=True)

    print("\n=== TEST B: upscale x1.5 then imgsz sweep ===", flush=True)
    yml_up = prep("up15", 1.5)
    for imgsz in (640, 1280):
        try:
            m = model.val(data=str(yml_up), imgsz=imgsz, device="0", split="val",
                          verbose=False, plots=False, project=str(WORK / "runs"),
                          name="up15_%d" % imgsz)
            out["up15_imgsz_%d" % imgsz] = {
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                "P": float(m.box.mp), "R": float(m.box.mr)}
            print("  up1.5 imgsz=%-5d mAP50 %.4f  mAP50-95 %.4f  P %.4f  R %.4f"
                  % (imgsz, m.box.map50, m.box.map, m.box.mp, m.box.mr), flush=True)
        except Exception as exc:
            print("  imgsz=%d failed: %s: %s" % (imgsz, type(exc).__name__, exc), flush=True)

    outp = Path(r"C:\Users\33848\论文投稿\darkdriving\results_task4\diagnostic_detection.json")
    outp.write_text(json.dumps({
        "protocol": "task4_partB_diagnostic",
        "detector": str(WEIGHTS), "n_images": LIMIT, "domain": "night/val",
        "results": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
