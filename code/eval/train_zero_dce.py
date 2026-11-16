"""Train the Zero-DCE / Zero-DCE++ baselines on our clean low-light inputs.

Zero-DCE is a *zero-reference* method: it never sees a target image.  To keep the
comparison honest we therefore train it on exactly the low-light inputs of our
own clean split -- the same images our supervised models were trained on -- and
then evaluate on the held-out LOL-v1 eval15 pairs.  The ground truth is used only
for scoring, never for training.

Losses follow the published formulation:
  L_spa  spatial consistency  (4 neighbourhoods, TV of the mean-shifted pairs)
  L_exp  exposure control     (mean patch level to a target exposure E)
  L_col  colour constancy     (sum over channel-pair mean differences)
  L_tvA  illumination smoothness (TV on the curve parameter maps)
Total: L = L_spa + L_exp + 5*L_col + 200*L_tvA   (published weights)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_dce import DCENet, DCENetPlusPlus, count_params  # noqa: E402

PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
CLEAN_TRAIN_CSV = PROJECT / "experiments" / "datasets" / "combined_lsd_lol_clean_v1" / "train_pairs.csv"


# ---------------------------------------------------------------- losses
def loss_spa(out, x, patch=4):
    """Spatial consistency: keep relative contrast between local patches."""
    k = torch.ones(1, 1, patch, patch, device=x.device) / (patch * patch)
    kernel = k.repeat(3, 1, 1, 1)

    out_mean = F.conv2d(out, kernel, stride=patch, groups=3)
    x_mean = F.conv2d(x, kernel, stride=patch, groups=3)

    d_out = (out_mean[:, :, 1:, :] - out_mean[:, :, :-1, :]).abs().mean() + \
            (out_mean[:, :, :, 1:] - out_mean[:, :, :, :-1]).abs().mean()
    d_in = (x_mean[:, :, 1:, :] - x_mean[:, :, :-1, :]).abs().mean() + \
           (x_mean[:, :, :, 1:] - x_mean[:, :, :, :-1]).abs().mean()
    return (d_out - d_in).pow(2)


def loss_exp(out, mean_val=0.6, patch=16):
    """Exposure control: pull every local patch mean toward the target exposure."""
    k = torch.ones(1, 1, patch, patch, device=out.device) / (patch * patch)
    kernel = k.repeat(3, 1, 1, 1)
    mean = F.conv2d(out, kernel, stride=patch, groups=3)
    return (mean - mean_val).pow(2).mean()


def loss_col(out):
    """Colour constancy: penalise channel-mean divergence."""
    mr, mg, mb = out[:, 0].mean(), out[:, 1].mean(), out[:, 2].mean()
    d1 = (mr - mg).pow(2) + (mr - mb).pow(2)
    d2 = (mg - mr).pow(2) + (mg - mb).pow(2)
    d3 = (mb - mr).pow(2) + (mb - mg).pow(2)
    return d1 + d2 + d3


def loss_tv(layer_out):
    """Total variation on the curve parameter maps."""
    diff_h = (layer_out[:, :, 1:, :] - layer_out[:, :, :-1, :]).abs().mean()
    diff_w = (layer_out[:, :, :, 1:] - layer_out[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


# ---------------------------------------------------------------- data
class LowOnly(Dataset):
    """Returns only the low-light input; Zero-DCE is trained without targets."""

    def __init__(self, csv_path: Path, crop: int = 256, limit: int = 0, seed: int = 0):
        rows = []
        with csv_path.open(encoding="utf-8") as f:
            header = f.readline()
            cols = [c.strip() for c in header.split(",")]
            li = 0
            for cand in ("low_path", "input_path", "low", "input", "src"):
                if cand in cols:
                    li = cols.index(cand)
                    break
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) > li:
                    rows.append(parts[li].strip().strip('"'))
        if limit:
            rows = rows[:limit]
        self.rows = [r for r in rows if Path(r).exists()]
        self.crop = crop
        self.rng = random.Random(seed)
        if not self.rows:
            raise SystemExit("no usable low-light images found in %s" % csv_path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p = self.rows[i]
        im = Image.open(p).convert("RGB")
        w, h = im.size
        c = self.crop
        if w < c or h < c:
            im = im.resize((max(c, w), max(c, h)), Image.BILINEAR)
            w, h = im.size
        x = self.rng.randint(0, w - c)
        y = self.rng.randint(0, h - c)
        im = im.crop((x, y, x + c, y + c))
        if self.rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["zero_dce", "zero_dce_pp"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=3000, help="0 = all training low images")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default=r"C:\Users\33848\论文投稿\STAGE6_v0.8\baselines\runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DCENet(32, 8) if args.variant == "zero_dce" else DCENetPlusPlus(32, 8, 12)
    model = model.to(device)
    nparam = count_params(model)
    print("variant=%s  params=%d  device=%s" % (args.variant, nparam, device), flush=True)

    ds = LowOnly(CLEAN_TRAIN_CSV, crop=args.crop, limit=args.limit, seed=args.seed)
    print("training low-light images: %d (targets never used)" % len(ds), flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out) / args.variant / ("seed%d" % args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"spa": 0.0, "exp": 0.0, "col": 0.0, "tv": 0.0, "tot": 0.0}
        nb = 0
        for x in dl:
            x = x.to(device, non_blocking=True)
            out, curve_maps = model(x, return_maps=True)

            l_spa = loss_spa(out, x)
            l_exp = loss_exp(out)
            l_col = loss_col(out)
            # published L_tvA acts on the illumination/curve parameter map A,
            # not on the enhanced image
            l_tv = loss_tv(curve_maps)

            loss = l_spa + l_exp + 5.0 * l_col + 200.0 * l_tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()

            agg["spa"] += l_spa.detach().item()
            agg["exp"] += l_exp.detach().item()
            agg["col"] += l_col.detach().item()
            agg["tv"] += l_tv.detach().item()
            agg["tot"] += loss.detach().item()
            nb += 1
            step += 1
            if step % args.steps_per_epoch == 0:
                break
        sched.step()
        rec = {k: v / max(nb, 1) for k, v in agg.items()}
        rec["epoch"] = ep
        history.append(rec)
        print("epoch %2d/%d  spa %.4f exp %.5f col %.2e tv %.5f  total %.4f  (%.0fs)"
              % (ep, args.epochs, rec["spa"], rec["exp"], rec["col"], rec["tv"],
                 rec["tot"], time.time() - t0), flush=True)

    ckpt = out_dir / "best.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "parameters": nparam,
        "epochs": args.epochs,
        "steps": step,
        "crop": args.crop,
        "train_low_images": len(ds),
        "train_csv": str(CLEAN_TRAIN_CSV),
        "seed": args.seed,
        "note": "zero-reference training; ground truth never used",
    }, ckpt)
    (out_dir / "history.json").write_text(
        json.dumps({"variant": args.variant, "params": nparam, "history": history},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", ckpt)
    print("total time %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
