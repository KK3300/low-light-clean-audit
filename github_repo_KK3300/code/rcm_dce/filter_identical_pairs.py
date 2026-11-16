"""Create an explicit clean paired manifest by excluding audited identity pairs."""
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


def norm(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--identity-csv", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    src = Path(args.input).resolve()
    output = Path(args.output).resolve()
    audited_pairs = set()
    audit_sources = []
    for audit_arg in args.identity_csv:
        audit = Path(audit_arg).resolve()
        audit_sources.append({"path": str(audit), "sha256": sha256_file(audit)})
        with audit.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("input") and row.get("target"):
                    audited_pairs.add((norm(row["input"]), norm(row["target"])))
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    kept, removed = [], []
    for row in rows:
        key = (norm(row["input"]), norm(row["target"]))
        (removed if key in audited_pairs else kept).append(row)
    if not kept:
        raise ValueError("filter removed every row")
    ids = [row.get("id") for row in kept]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs remain after filtering")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    groups_before = Counter(row.get("source_dataset") or "default" for row in rows)
    groups_after = Counter(row.get("source_dataset") or "default" for row in kept)
    report = {
        "protocol": "stage4_identity_filtered_manifest_v1",
        "source": str(src),
        "source_sha256": sha256_file(src),
        "identity_audits": audit_sources,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "removed_byte_identical_pairs": len(removed),
        "removed_fraction": len(removed) / len(rows),
        "groups_before": dict(groups_before),
        "groups_after": dict(groups_after),
        "scene_keys_after": len({row.get("scene_key") for row in kept}),
        "removed_ids": [row.get("id") for row in removed],
    }
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
