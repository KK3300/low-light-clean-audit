"""Create project-root-relative copies of absolute-path pair manifests.

The original manifests are never overwritten. Relative copies are intended for
portability and must be used with the repository root as the working directory.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def normalize_path(raw: str, project_root: Path, manifest_path: Path) -> str:
    path = Path(raw)
    if not path.is_absolute():
        candidates = [manifest_path.parent / path, project_root / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    path = path.resolve()
    try:
        rel = os.path.relpath(path, project_root)
    except ValueError as exc:
        raise ValueError(f"Cannot make {path} relative to {project_root}") from exc
    return Path(rel).as_posix()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--project-root", required=True, type=Path)
    args = ap.parse_args()

    source = args.input.resolve()
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "input" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError(f"Manifest lacks input/target columns: {source}")
        rows = []
        for row in reader:
            row["input"] = normalize_path(row["input"], project_root, source)
            row["target"] = normalize_path(row["target"], project_root, source)
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)}")


if __name__ == "__main__":
    main()
