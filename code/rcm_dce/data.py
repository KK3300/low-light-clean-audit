"""Deterministic data utilities for the Stage 2/3 RCM-DCE protocol."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_worker(worker_id: int) -> None:
    # torch.initial_seed is derived from the DataLoader generator.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def pil_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class LowLightFolder(Dataset):
    def __init__(self, root: str | Path, crop_size: int = 256, augment: bool = True):
        self.root = Path(root)
        self.paths = list_images(self.root)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.crop_size = int(crop_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = pil_to_tensor(self.paths[index])
        if self.crop_size:
            h, w = x.shape[-2:]
            if h < self.crop_size or w < self.crop_size:
                x = F.interpolate(
                    x[None],
                    size=(max(h, self.crop_size), max(w, self.crop_size)),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                h, w = x.shape[-2:]
            if self.augment:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top = max(0, (h - self.crop_size) // 2)
                left = max(0, (w - self.crop_size) // 2)
            x = x[:, top : top + self.crop_size, left : left + self.crop_size]
        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x


class LowLightManifest(Dataset):
    """Low-only view of an explicit paired/scene manifest.

    Target columns are intentionally ignored during training; they remain in
    the manifest only for auditability and later paired validation.
    """

    def __init__(self, manifest: str | Path, crop_size: int = 256, augment: bool = True):
        self.manifest = Path(manifest)
        self.rows = []
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for line_no, row in enumerate(csv.DictReader(f), 2):
                if not row.get("input"):
                    raise ValueError(f"Missing input at {manifest}:{line_no}")
                p = Path(row["input"])
                if not p.exists():
                    raise FileNotFoundError(f"Missing input at {manifest}:{line_no}: {p}")
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty low-light manifest: {manifest}")
        self.crop_size = int(crop_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = pil_to_tensor(Path(self.rows[index]["input"]))
        if self.crop_size:
            h, w = x.shape[-2:]
            if h < self.crop_size or w < self.crop_size:
                x = F.interpolate(x[None], size=(max(h, self.crop_size), max(w, self.crop_size)), mode="bilinear", align_corners=False)[0]
                h, w = x.shape[-2:]
            if self.augment:
                top, left = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            else:
                top, left = max(0, (h - self.crop_size) // 2), max(0, (w - self.crop_size) // 2)
            x = x[:, top : top + self.crop_size, left : left + self.crop_size]
        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x


class PairedManifest(Dataset):
    """Explicit paired evaluator; fails loudly on missing or malformed rows."""

    def __init__(self, manifest: str | Path, crop_size: int = 0, augment: bool = False):
        self.manifest = Path(manifest)
        self.rows: List[Tuple[Path, Path, str]] = []
        self.groups: List[str] = []
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for line_no, row in enumerate(csv.DictReader(f), 2):
                if not row.get("input") or not row.get("target"):
                    raise ValueError(f"Missing input/target at {manifest}:{line_no}")
                inp, tgt = Path(row["input"]), Path(row["target"])
                if not inp.exists() or not tgt.exists():
                    raise FileNotFoundError(f"Missing pair at {manifest}:{line_no}: {inp} / {tgt}")
                item_id = row.get("id") or str(len(self.rows))
                self.rows.append((inp, tgt, item_id))
                # Optional manifest namespace used only for domain-balanced
                # sampling/selection.  It never changes the returned target.
                group = row.get("source_dataset") or (item_id.split(":", 1)[0] if ":" in item_id else "default")
                self.groups.append(str(group))
        if not self.rows:
            raise ValueError(f"Empty paired manifest: {manifest}")
        ids = [r[2] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate ids in {manifest}")
        self.crop_size = int(crop_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        inp, tgt, item_id = self.rows[index]
        x, y = pil_to_tensor(inp), pil_to_tensor(tgt)
        if x.shape[-2:] != y.shape[-2:]:
            raise ValueError(f"shape mismatch for {item_id}: {x.shape} vs {y.shape}")
        if self.crop_size:
            h, w = x.shape[-2:]
            h = min(h, self.crop_size)
            w = min(w, self.crop_size)
            if self.augment:
                top = random.randint(0, max(0, x.shape[-2] - h))
                left = random.randint(0, max(0, x.shape[-1] - w))
            else:
                top, left = max(0, (x.shape[-2] - h) // 2), max(0, (x.shape[-1] - w) // 2)
            x, y = x[:, top : top + h, left : left + w], y[:, top : top + h, left : left + w]
        if self.augment and random.random() < 0.5:
            x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
        return x, y, item_id
