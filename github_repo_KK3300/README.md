# low-light-clean-audit

Artifacts for the manuscript:

> **A Parameter-Matched Cross-Domain Study of Multi-Scale Refinement in Single-Pass Low-Light Curve Enhancement**
> Xiang Liu, Harbin Engineering University
> Manuscript prepared for *Electronics* (MDPI), 2026.

This repository contains the clean manifests, the byte-identical pair audit, the two clean checkpoints, and the reference evaluation code used in the paper. It does **not** redistribute any image data; the four datasets keep their original licenses.

---

## What is here

| Path | Contents |
|---|---|
| `manifests/train_pairs.csv` | Clean combined training manifest, **8,040 pairs** (LSD 7,652 + LOL 388) |
| `manifests/val_pairs.csv` | Clean combined validation manifest, **479 pairs** (LSD 382 + LOL 97) |
| `audit/identity_pairs.csv` | **434** byte-identical input–target pairs in the 10,000-pair LSD-DLL mirror |
| `audit/content_leak_pairs.csv` | Additional content-leak pairs found during the audit |
| `checkpoints/plain_epoch002.pth` | Plain125 curve backbone, **995,253** parameters |
| `checkpoints/ms_refine_uncond_epoch003.pth` | MS-Refine (backbone + quarter-resolution branch), **1,006,265** parameters |
| `code/rcm_dce/` | Model, data, metrics, training and manifest utilities |
| `code/eval/` | Protocol-matched fidelity evaluation and cross-domain evaluation |
| `code/downstream/` | Downstream YOLO11x detection scripts |
| `SHA256SUMS.txt` | SHA-256 of every file in this repository |

The checkpoints match Appendix A of the manuscript exactly:

```text
plain_epoch002.pth              fcc050473f5d4d55d6fbd8b9f524456124420b4b79bdd4b11bcff35346a39b24
ms_refine_uncond_epoch003.pth   e49a138f7359272922b69ae7f11279684d11e969b7ae314eb63408516de60e6b
```

---

## Key numbers

| Quantity | Value |
|---|---|
| Degenerate pairs in the 10,000-pair mirror | 434 (4.34%) |
| Clean training / validation manifests | 8,040 / 479 pairs |
| Parameter difference between the two models | +11,012 (+1.11%) |
| LOL-v1 eval15, ΔPSNR (refinement − backbone) | +0.239 ± 1.136 dB |
| NTIRE 2026 center-1024, ΔPSNR | +0.421 ± 0.639 dB |
| Downstream DarkDriving, branch vs backbone | +10.2% mAP50 (one frozen detector) |

The fidelity gain is positive on average, but the pre-specified colour/artefact gate fails on NTIRE; the downstream gain is reported as relative evidence from a weak, single-run detector. See the manuscript for the full boundary conditions.

---

## Quick start

```bash
git clone https://github.com/KK3300/low-light-clean-audit.git
cd low-light-clean-audit

# 1. verify file integrity
sha256sum -c SHA256SUMS.txt

# 2. inspect manifest sizes
wc -l manifests/train_pairs.csv manifests/val_pairs.csv
# 8041 and 480 lines including the header

# 3. inspect a checkpoint (PyTorch)
python - <<'PY'
import torch
ckpt = torch.load("checkpoints/plain_epoch002.pth", map_location="cpu")
print(type(ckpt))
print(len(ckpt) if hasattr(ckpt, "__len__") else "")
PY
```

The evaluation scripts are **reference implementations preserved as run**. They contain absolute path constants at the top (`PROJECT`, `ROOT`, `EVAL15`, `LPIPS_CKPT`) that must be edited for your environment. See `REPRODUCE.md` for the expected directory layout and commands.

---

## Datasets

No images are included. The four corpora are:

- **LSD-DLL** 10,000-pair mirror (LSD / NTIRE 2026 Track-1 training family), pinned at commit `e078e0d3c8de043b1214ca5943ed4abd320aeb66` in the manuscript.
- **LOL-v1** with the standard `eval15` split.
- **NTIRE 2026** public validation package (24 pairs, center-1024 protocol).
- **DarkDriving** detection release (bright-half training, dark validation).

Please obtain each dataset from its original authors and follow its license. The manifests use relative paths of the form:

```text
experiments/datasets/LSD_DLL_subset_10000/input/<name>.png
experiments/datasets/LSD_DLL_subset_10000/gt/<name>.png
```

so a checkout that places the datasets under `experiments/datasets/` can use the manifests without rewriting them.

---

## Code notes

- `code/rcm_dce/` is the model/training package used to produce the checkpoints.
- `code/eval/compare_lol_eval15.py` scores the protocol-matched LOL-v1 table.
- `code/eval/eval_crossdomain.py` scores the LSD-only / LOL-only / mixed arms.
- `code/eval/task4_nonlearned_vs_learned.py` contains the metric and baseline helpers.
- `code/downstream/` contains the YOLO11x fine-tuning and evaluation scripts; they require `ultralytics` and the DarkDriving detection release.

The scripts are research code, not a turnkey benchmark. Paths, checkpoint selection and dataset roots are explicit constants so that the exact protocol used in the paper is auditable. If you adapt them, keep the metric implementation (`code/rcm_dce/metrics.py`) unchanged when comparing with the manuscript.

---

## License

Code and checkpoints: **MIT** (see `LICENSE`).
Manifests and audit lists: **CC BY 4.0**.
Datasets: original licenses; no image data is redistributed here.

## Citation

If you use these artifacts, please cite the manuscript (under review) and this repository. A `CITATION.cff` is included.

## Contact

Xiang Liu — `liuxiang19846190250@hrbeu.edu.cn`
ORCID: [0009-0004-5419-1698](https://orcid.org/0009-0004-5419-1698)
