# Reproducing the manuscript numbers

This is a **reference** reproduction guide, not a one-command benchmark. The code is preserved as it was run; the absolute path constants at the top of each script must be edited for your checkout.

---

## 1. Environment

Tested with:

- Python 3.10+ (3.12 used locally)
- PyTorch 2.x with CUDA
- `numpy`, `Pillow`, `scipy`
- `lpips` (LPIPS-Alex weights)
- `ultralytics` (downstream YOLO11x)
- `pandas` (manifest handling)

The fidelity metrics used in the paper are implemented in `code/rcm_dce/metrics.py`; do not substitute a different PSNR/SSIM/CIEDE2000 implementation when comparing with the manuscript.

---

## 2. Expected directory layout

The scripts assume a project root (called `PROJECT` in most files) containing:

```text
experiments/
├── datasets/
│   ├── LSD_DLL_subset_10000/
│   │   ├── input/
│   │   └── gt/
│   ├── LOL_v1/
│   │   └── extracted/
│   │       └── eval15/
│   │           ├── low/
│   │           └── high/
│   ├── NTIRE2026/              # public validation package, center-1024 protocol
│   └── DarkDriving_det_release/
│       └── Det_data/
│           └── night/
│               ├── images/{train,val}/
│               └── labels/{train,val}/
└── results/
    └── stage4_clean_balanced_3seed/
        └── seed20260830/
            ├── plain125/
            └── ms_refine_uncond/
```

The manifests in `manifests/` use relative paths such as:

```text
experiments/datasets/LSD_DLL_subset_10000/input/1000_s22Ultra_dll.png
experiments/datasets/LSD_DLL_subset_10000/gt/1000_s22Ultra_dll.png
```

so a project root with the datasets at those paths can consume them unchanged.

No image data is redistributed in this repository. Obtain each dataset from its original authors and follow its license.

---

## 3. Verify the artifacts

```bash
# counts: 8040 and 479 rows excluding the header
python - <<'PY'
import csv
for f in ["manifests/train_pairs.csv", "manifests/val_pairs.csv"]:
    rows = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    print(f, len(rows))
PY

# audit list: 434 rows
python - <<'PY'
import csv
rows = list(csv.DictReader(open("audit/identity_pairs.csv", newline="", encoding="utf-8")))
print("identity pairs:", len(rows))
PY

# checksums
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell:

```powershell
Get-FileHash .\checkpoints\*.pth -Algorithm SHA256
```

Expected:

```text
plain_epoch002.pth              fcc050473f5d4d55d6fbd8b9f524456124420b4b79bdd4b11bcff35346a39b24
ms_refine_uncond_epoch003.pth   e49a138f7359272922b69ae7f11279684d11e969b7ae314eb63408516de60e6b
```

---

## 4. Edit the path constants

Each script has explicit constants at the top. Edit at least:

| File | Constants |
|---|---|
| `code/eval/compare_lol_eval15.py` | `ROOT`, `PROJECT`, `EVAL15`, `LPIPS_CKPT` |
| `code/eval/task4_nonlearned_vs_learned.py` | `PROJECT`, `CKPT_ROOT`, `LOL_EVAL15` |
| `code/eval/eval_crossdomain.py` | `ROOT` (and the paths imported from the two files above) |
| `code/rcm_dce/train.py` | `PROJECT` / dataset roots, output directory |
| `code/rcm_dce/evaluate.py` | checkpoint and manifest roots |

Then put the code on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/code:$PWD/code/eval:$PWD/code/rcm_dce"
```

```powershell
$env:PYTHONPATH = "$PWD\code;$PWD\code\eval;$PWD\code\rcm_dce"
```

---

## 5. Fidelity evaluation (LOL-v1 eval15)

```bash
python code/eval/compare_lol_eval15.py
```

This produces the protocol-matched table (Table 7 of the manuscript): all methods scored on the same 15 held-out LOL-v1 pairs with the same metric implementation. The learned baselines use the official Zero-DCE / Zero-DCE++ checkpoints; the two project models use the checkpoints in `checkpoints/`.

The seed-level comparison in Table 5 (the +0.239 ± 1.136 dB result) requires the three clean training seeds, not a single checkpoint. The seed checkpoints are not all included here; the two released files are the seed-20260830 models used in the qualitative figure and the downstream sweep.

---

## 6. Cross-domain evaluation

```bash
python code/eval/eval_crossdomain.py
```

This scores the LSD-only, LOL-only and mixed arms on LOL-v1 eval15 and reproduces the domain-gap numbers in Section 4.6 / Table 8. The in-domain arm exists for the backbone only, as stated in the manuscript.

---

## 7. Downstream detection (DarkDriving)

```bash
python code/downstream/task4_partB_finetune.py
python code/downstream/task4_partB_detection.py
```

The detector is a YOLO11x initialised from COCO weights and fine-tuned for ten epochs on the bright-half training split; the enhancement front-ends are applied to the dark validation split and scored by the Ultralytics validation routine at 640 px. The manuscript reports one frozen detector (single run); the fine-tuning curve does not converge, so the numbers are relative comparisons, not absolute detection performance.

---

## 8. What the paper reports

| Quantity | Value |
|---|---|
| Degenerate pairs / corpus | 434 / 10,000 (4.34%) |
| Clean training / validation | 8,040 / 479 |
| Parameters, Plain125 / MS-Refine | 995,253 / 1,006,265 |
| Parameter difference | +11,012 (+1.11%) |
| ΔPSNR, LOL-v1 eval15 | +0.239 ± 1.136 dB |
| ΔPSNR, NTIRE 2026 center-1024 | +0.421 ± 0.639 dB |
| Downstream, MS-Refine vs Plain125 | +10.2% mAP50 |

The fidelity gain is positive on average, but the pre-specified colour/artefact gate fails on NTIRE; the downstream gain comes from one weak, single-run detector. These boundaries are part of the result, not caveats to be removed.

---

## 9. Known limitations of this reference code

- Paths are absolute in the original runs; they are not auto-detected.
- The seed-level training sweep (three seeds) is not a single command in this repository; the released checkpoints are the seed-20260830 artifacts.
- The downstream detector is intentionally weak and single-run; it supports relative front-end comparisons only.
- The NTIRE hidden-test ground truth is not used or redistributed.
