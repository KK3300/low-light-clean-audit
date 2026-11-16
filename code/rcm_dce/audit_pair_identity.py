"""Audit paired manifests for byte-identical input/target leakage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_sha(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    manifest_reports = []

    for manifest_arg in args.manifests:
        manifest = Path(manifest_arg).resolve()
        with manifest.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        identical = []
        size_equal = 0
        prefixes = Counter()
        for index, row in enumerate(rows, 1):
            inp, tgt = Path(row["input"]), Path(row["target"])
            a, b = inp.stat(), tgt.stat()
            same = False
            digest = None
            if a.st_size == b.st_size:
                size_equal += 1
                digest_a = sha256_file(inp)
                digest_b = sha256_file(tgt)
                same = digest_a == digest_b
                digest = digest_a if same else None
            if same:
                name = inp.name
                if "Test240213" in name:
                    prefix = "Test240213"
                elif "240119" in name:
                    prefix = "240119"
                else:
                    prefix = name.split("_", 1)[-1] if "_" in name else "other"
                prefixes[prefix] += 1
                record = {
                    "manifest": str(manifest),
                    "row_index": index,
                    "id": row.get("id"),
                    "scene_key": row.get("scene_key"),
                    "source_dataset": row.get("source_dataset"),
                    "input": str(inp),
                    "target": str(tgt),
                    "bytes": a.st_size,
                    "sha256": digest,
                    "filename_group": prefix,
                }
                identical.append(record)
                all_records.append(record)
            if index % 1000 == 0:
                print(f"{manifest.name}: {index}/{len(rows)}", flush=True)
        manifest_reports.append(
            {
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha(manifest),
                "rows": len(rows),
                "size_equal_pairs": size_equal,
                "byte_identical_pairs": len(identical),
                "byte_identical_fraction": len(identical) / max(len(rows), 1),
                "unique_identical_scene_keys": len({row["scene_key"] for row in identical}),
                "filename_groups": dict(prefixes),
            }
        )

    report = {
        "protocol": "stage4_pair_identity_audit_v1",
        "definition": "byte-identical input and target files confirmed by equal size and SHA-256",
        "manifests": manifest_reports,
        "total_rows": sum(row["rows"] for row in manifest_reports),
        "total_byte_identical_pairs": len(all_records),
        "unique_identical_scene_keys": len({row["scene_key"] for row in all_records}),
        "implication": "Such pairs contain no enhancement mapping and inflate identity-output metrics. Existing checkpoints are retained as contaminated-pilot evidence only until clean-manifest retraining.",
    }
    (out_dir / "pair_identity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(all_records[0]) if all_records else ["manifest", "id", "scene_key", "input", "target"]
    with (out_dir / "byte_identical_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
