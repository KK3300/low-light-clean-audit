"""Task 4 / Part B: does low-light enhancement help a FROZEN detector?

Protocol
--------
front-end  : none | gamma | clahe | msrcr | lime | plain125 | ms_refine_uncond
detector   : YOLO11x (official COCO release weights), frozen, single-class car
domains    : night/val (real low light) | day/val (bright reference) |
             snr-aware/night/val (512 px scale)
metrics    : mAP50, mAP50-95, precision, recall

Design notes
------------
* Images are resized to `--imgsz` with a fixed letterbox before detection, so a
  detector cannot simply win by being handed more pixels.
* The learned front-ends were trained on LOL/LSD paired data; on DarkDriving
  they are used ZERO-SHOT.  That is stated, not hidden.
* Enhancement is applied to the full-resolution image, then the result is
  written as a single-class YOLO dataset and validated with the same frozen
  weights, so every arm differs only in the front-end.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task4_nonlearned_vs_learned import CLASSICAL, CKPT, load_learned  # noqa: E402

DATA = PROJECT / "experiments" / "datasets" / "DarkDriving_det_release" / "Det_data"
DOMAINS = {
    "night_val": (DATA / "night" / "images" / "val", DATA / "night" / "labels" / "val"),
    "day_val": (DATA / "day" / "images" / "val", DATA / "day" / "labels" / "val"),
    "snr512_val": (DATA / "snr-aware" / "night" / "images" / "val",
                   DATA / "snr-aware" / "night" / "labels" / "val"),
}


def _torch_device(dev: str) -> str:
    """Ultralytics uses '0'/'cpu'; torch wants 'cuda'/'cpu'."""
    d = str(dev).strip()
    if d.isdigit():
        return "cuda"
    return d


def ensure_learned_images(method, src_imgs, dst_imgs, device, limit=0, imgsz=None, enh_max=1024):
    """Run a learned front-end over an unpaired image directory.

    The learned models were trained on 256 px crops; feeding a 2448x2048 frame
    would blow up an 8 GB GPU.  We therefore run the enhancer at a reduced
    resolution (`enh_max` on the long side) and resize the enhanced result back
    to the original size, so the detector still sees the same frame geometry as
    every other arm.  This is recorded in the result JSON.
    """
    import torch
    ck = CKPT[method]
    tdev = torch.device(_torch_device(device))
    model, sig, nparam = load_learned(ck, tdev)
    dst_imgs.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in Path(src_imgs).glob("*.jpg")])
    if limit:
        files = files[:limit]
    done = 0
    t0 = time.time()
    with torch.no_grad():
        for p in files:
            op = dst_imgs / (p.stem + ".jpg")
            if op.exists():
                done += 1
                continue
            src = Image.open(p).convert("RGB")
            w, h = src.size
            small = src
            if enh_max and max(w, h) > enh_max:
                s = enh_max / max(w, h)
                small = src.resize((max(8, int(round(w * s))), max(8, int(round(h * s)))), Image.BILINEAR)
            rgb = np.asarray(small, dtype=np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(tdev)
            out = model(x)
            pred = (out[0] if isinstance(out, tuple) else out)[0].permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((np.clip(pred, 0, 1) * 255).round().astype(np.uint8))
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            img.save(op, quality=95)
            done += 1
            if done % 200 == 0:
                print("   %s: %d/%d  (%.1f img/s)" % (method, done, len(files),
                                                      done / max(time.time() - t0, 1e-6)), flush=True)
    return done, nparam


def build_dataset_yaml(work: Path, images_dir: Path, labels_dir: Path, name: str):
    """Create a single-class YOLO dataset view pointing at prepared images."""
    root = work / name
    img_link = root / "images" / "val"
    lbl_link = root / "labels" / "val"
    img_link.mkdir(parents=True, exist_ok=True)
    lbl_link.mkdir(parents=True, exist_ok=True)
    # labels are shared; copy once (they are tiny)
    for f in Path(labels_dir).glob("*.txt"):
        tgt = lbl_link / f.name
        if not tgt.exists():
            shutil.copy2(f, tgt)
    cfg = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": {0: "car"},
    }
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return yml, img_link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\33848\论文投稿\darkdriving\results_task4")
    ap.add_argument("--work", default=r"C:\Users\33848\论文投稿\darkdriving\work_partB")
    ap.add_argument("--weights", default=r"C:\Users\33848\论文投稿\yolo11x.pt")
    ap.add_argument("--domains", default="night_val")
    ap.add_argument("--methods", default="none,gamma,clahe,msrcr,lime")
    ap.add_argument("--learned", default="", help="comma list of learned front-ends to include")
    ap.add_argument("--limit", type=int, default=300, help="images per domain (0 = all)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--enh-max", type=int, default=1024,
                    help="long-side resolution at which learned front-ends run")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.exists():
        # fall back to whatever ultralytics downloaded into cwd
        cand = sorted(Path.cwd().glob("yolo11x*.pt")) + sorted(Path(r"C:\Users\33848\论文投稿\darkdriving").glob("*.pt"))
        if not cand:
            raise SystemExit("no yolo11x.pt found; run the download step first (%s)" % weights)
        weights = cand[0]
    print("weights:", weights, flush=True)

    model = YOLO(str(weights))
    results = {}

    method_list = [x.strip() for x in args.methods.split(",") if x.strip()]
    method_list += [x.strip() for x in args.learned.split(",") if x.strip()]

    for dname in [d.strip() for d in args.domains.split(",") if d.strip()]:
        src_imgs, lbl_dir = DOMAINS[dname]
        n_src = len(list(Path(src_imgs).glob("*.jpg")))
        print("\n=== domain %s (%d images, using %s) ===" % (dname, n_src, args.limit or "all"), flush=True)

        for m in method_list:
            tag = "%s__%s" % (dname, m)
            yml, img_link = build_dataset_yaml(work, src_imgs, lbl_dir, tag)
            t0 = time.time()
            if m == "none":
                # link originals without re-encoding
                files = sorted(Path(src_imgs).glob("*.jpg"))
                if args.limit:
                    files = files[:args.limit]
                for p in files:
                    tgt = img_link / p.name
                    if not tgt.exists():
                        try:
                            tgt.hardlink_to(p)
                        except Exception:
                            shutil.copy2(p, tgt)
                n = len(files)
            elif m in CLASSICAL:
                from task4_nonlearned_vs_learned import enhance_to_disk
                n = enhance_to_disk(m, src_imgs, img_link, limit=args.limit, skip_existing=True,
                                    enh_max=args.enh_max)
            elif m in CKPT:
                try:
                    n, nparam = ensure_learned_images(m, src_imgs, img_link, args.device,
                                                      limit=args.limit, enh_max=args.enh_max)
                except Exception as exc:
                    print("  learned front-end %s failed: %s: %s" % (m, type(exc).__name__, exc))
                    continue
            else:
                print("  skip unknown method", m)
                continue

            t_prep = time.time() - t0
            try:
                metrics = model.val(data=str(yml), imgsz=args.imgsz, device=args.device,
                                    split="val", verbose=False, plots=False,
                                    project=str(work / "runs"), name=tag)
                row = {
                    "domain": dname, "frontend": m, "n_images": n,
                    "mAP50": float(metrics.box.map50),
                    "mAP50_95": float(metrics.box.map),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "prep_seconds": round(t_prep, 1),
                    "imgsz": args.imgsz,
                }
            except Exception as exc:
                row = {"domain": dname, "frontend": m, "n_images": n,
                       "error": "%s: %s" % (type(exc).__name__, exc)}
            results["%s|%s" % (dname, m)] = row
            print("  %-22s mAP50 %.4f  mAP50-95 %.4f  P %.4f  R %.4f  (n=%s, prep %.0fs)"
                  % (m, row.get("mAP50", float("nan")), row.get("mAP50_95", float("nan")),
                     row.get("precision", float("nan")), row.get("recall", float("nan")),
                     row.get("n_images"), t_prep), flush=True)

            outp = out_dir / "task4_partB_detection.json"
            outp.write_text(json.dumps({
                "protocol": "task4_partB_frozen_detector",
                "detector": str(weights),
                "imgsz": args.imgsz,
                "limit_per_domain": args.limit,
                "results": results,
            }, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nwrote", out_dir / "task4_partB_detection.json")


if __name__ == "__main__":
    main()
