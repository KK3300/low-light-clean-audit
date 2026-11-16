"""Deterministically split a paired manifest by scene key."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def bucket(scene: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()
    src = Path(args.input).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty input manifest")
    scenes = {r.get("scene_key") or Path(r["id"]).stem.split("_", 1)[0] for r in rows}
    ordered = sorted(scenes, key=lambda s: (bucket(s, args.seed), s))
    val_n = max(1, round(len(ordered) * args.val_fraction))
    val_scenes = set(ordered[:val_n])
    train, val = [], []
    for r in rows:
        s = r.get("scene_key") or Path(r["id"]).stem.split("_", 1)[0]
        r = dict(r)
        r["scene_key"] = s
        r["split"] = "scene_disjoint_val" if s in val_scenes else "scene_disjoint_train"
        (val if s in val_scenes else train).append(r)
    fields = ["input", "target", "id", "scene_key", "split"]
    for name, subset in [("train_pairs.csv", train), ("val_pairs.csv", val), ("train_low_manifest.csv", train)]:
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)
    report = {"source": str(src), "seed": args.seed, "val_fraction": args.val_fraction, "rows": len(rows), "train_rows": len(train), "val_rows": len(val), "scene_count": len(scenes), "train_scene_count": len(scenes - val_scenes), "val_scene_count": len(val_scenes), "scene_overlap": sorted((scenes - val_scenes) & val_scenes), "train_pairs": str((out / "train_pairs.csv").resolve()), "val_pairs": str((out / "val_pairs.csv").resolve()), "train_low_manifest": str((out / "train_low_manifest.csv").resolve())}
    import json
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
