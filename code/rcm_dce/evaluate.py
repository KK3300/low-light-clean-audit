"""Independent paired evaluator for a frozen Stage 2/3 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcm_dce.data import PairedManifest
from rcm_dce.metrics import evaluate_loader
from rcm_dce.model import RCMCurveNet, model_signature


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


KNOWN_IDENTITY_CONTAMINATED_MANIFESTS = {
    "88e48d9194bbb38940047867e2da182d4f1afea8f474b22953ce9c36ab7edfb4",
    "c53e6b46d1a355eb26ee8eaf51819be2c285d0b9253fc19c845dc50f7d593dfb",
    "0b55e6941c5005f9c55e5a7486c1ec3f47f0de8c27ddc803e2e0cfcb9852833a",
    "2cf32eada9e1f7028e4989121c5b6b81abfbfc82733cea81387936a785ea4e9b",
}


def checkpoint_training_manifest_hash(payload: dict) -> str | None:
    data = payload.get("config", {}).get("data", {})
    digest = data.get("train_manifest_sha256")
    if digest:
        return str(digest)
    value = data.get("train_manifest")
    if value:
        path = Path(value)
        if path.exists():
            return sha256_file(path)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset-name", default="frozen_test")
    ap.add_argument("--crop-size", type=int, default=0, help="optional center crop for screening; 0 keeps native size")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--allow-contaminated-pilot", action="store_true", help="legacy diagnostic only; final evidence must use clean checkpoints")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    ck_path = Path(args.checkpoint).resolve()
    # These checkpoints are produced locally by the trusted Stage 2 trainer;
    # the payload includes a model/config metadata object that PyTorch's
    # weights-only default cannot unpickle.
    payload = torch.load(ck_path, map_location="cpu", weights_only=False)
    if "model_signature" not in payload or "state_dict" not in payload:
        raise ValueError("checkpoint lacks Stage 2 model_signature/state_dict")
    training_manifest_sha256 = checkpoint_training_manifest_hash(payload)
    if training_manifest_sha256 in KNOWN_IDENTITY_CONTAMINATED_MANIFESTS and not args.allow_contaminated_pilot:
        raise ValueError(
            "Refusing checkpoint trained on a known identity-contaminated manifest. "
            "Use a clean_v1 checkpoint; --allow-contaminated-pilot is diagnostic-only."
        )
    sig = payload["model_signature"]
    model = RCMCurveNet(sig["variant"], sig["width"], sig["conv_mode"], sig["scale_factor"], sig["curve_steps"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_signature(model)["parameters"] != sig["parameters"]:
        raise AssertionError("checkpoint parameter signature mismatch")
    model.to(device)
    manifest = Path(args.manifest).resolve()
    ds = PairedManifest(manifest, crop_size=args.crop_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device, dataset_name=args.dataset_name)
    out = {"protocol": "stage2_independent_frozen_test_v1", "checkpoint": str(ck_path), "checkpoint_sha256": sha256_file(ck_path), "checkpoint_train_manifest_sha256": training_manifest_sha256, "identity_contamination_override": bool(args.allow_contaminated_pilot), "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "epoch": payload.get("epoch"), "model_signature": sig, "metrics": metrics}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
