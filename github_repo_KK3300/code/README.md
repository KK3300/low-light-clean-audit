# Code

Reference implementation of the evaluation protocol. These files are preserved as run; path constants at the top must be edited for your checkout.

## Layout

```text
code/
├── rcm_dce/
│   ├── model.py                  # RCMCurveNet: curve backbone + optional refinement branch
│   ├── data.py                   # dataset/manifest loading
│   ├── metrics.py                # PSNR, SSIM, CIEDE2000, LPIPS helpers
│   ├── train.py                  # training loop (three-seed recipe)
│   ├── evaluate.py               # evaluation entry point
│   ├── make_lol_manifests.py     # LOL-v1 eval15 manifest construction
│   ├── filter_identical_pairs.py # byte-identical pair filter
│   ├── audit_pair_identity.py    # SHA-256 pair audit
│   ├── normalize_manifest_paths.py
│   ├── split_manifest_by_scene.py
│   ├── combine_manifests.py
│   └── subset_manifest.py
├── eval/
│   ├── compare_lol_eval15.py     # protocol-matched LOL-v1 eval15 table
│   ├── eval_crossdomain.py       # LSD-only / LOL-only / mixed arms
│   ├── task4_nonlearned_vs_learned.py  # metric + baseline helpers
│   ├── zero_dce.py               # Zero-DCE / Zero-DCE++ reimplementations
│   ├── train_zero_dce.py
│   ├── run_crossdomain.py
│   └── run_batch.py
└── downstream/
    ├── task4_partB_finetune.py   # YOLO11x fine-tuning
    ├── task4_partB_detection.py  # front-end sweep on the dark validation split
    ├── multiseed_downstream.py
    ├── train_detector_seeds.py
    └── diag_detection.py
```

## Path constants

Most scripts start with constants such as:

```python
ROOT = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8")
PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new")
EVAL15 = PROJECT / "experiments" / "datasets" / "LOL_v1" / "extracted" / "eval15"
LPIPS_CKPT = PROJECT / "experiments" / "rcm_dce" / "lpips_alex.pth"
```

Edit them to your checkout before running. The dataset layout expected by the manifests is documented in `../REPRODUCE.md`.

## Dependencies

- `torch`, `numpy`, `Pillow`, `scipy`
- `lpips` (LPIPS-Alex)
- `ultralytics` (downstream only)
- `pandas` (manifest utilities)

Put the code on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/code:$PWD/code/eval:$PWD/code/rcm_dce"
```

```powershell
$env:PYTHONPATH = "$PWD\code;$PWD\code\eval;$PWD\code\rcm_dce"
```

## Caveats

- This is research code, not a packaged benchmark. There is no single `make all` command.
- The metric implementation (`rcm_dce/metrics.py`) is the one used for all numbers in the manuscript; keep it fixed when comparing.
- The downstream detector is deliberately weak and single-run; it supports relative front-end comparisons only.
- The scripts may import helpers from sibling directories; `PYTHONPATH` must include both `code/eval` and `code/rcm_dce`.
