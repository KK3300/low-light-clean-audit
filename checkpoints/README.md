# Checkpoints

Two clean seed-20260830 checkpoints, matching Appendix A of the manuscript.

| File | Model | Parameters | SHA-256 |
|---|---|---|---|
| `plain_epoch002.pth` | Plain125 — single-pass curve backbone | 995,253 | `fcc050473f5d4d55d6fbd8b9f524456124420b4b79bdd4b11bcff35346a39b24` |
| `ms_refine_uncond_epoch003.pth` | MS-Refine — backbone + quarter-resolution RGB branch | 1,006,265 | `e49a138f7359272922b69ae7f11279684d11e969b7ae314eb63408516de60e6b` |

Parameter difference: +11,012 (+1.11%).

## Loading

The model definitions are in `../code/rcm_dce/model.py` (`RCMCurveNet`). Example:

```python
import sys
import torch

sys.path.insert(0, "code")
from rcm_dce.model import RCMCurveNet  # check the constructor signature in model.py

state = torch.load("checkpoints/plain_epoch002.pth", map_location="cpu")
print(type(state), len(state) if hasattr(state, "__len__") else "")
```

Check `model.py` for the exact constructor arguments used by the paper (width `n = 125`, eight curve applications, refinement branch enabled for MS-Refine).

## Notes

- These are the clean models trained on `manifests/train_pairs.csv`; the contaminated pilot checkpoints are not included and are not used in any primary comparison.
- The checkpoint files are 3.8 MB each.
- Code and checkpoints are MIT-licensed; the datasets are not included.
