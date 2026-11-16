"""Queue the Stage 6 experiment batch so it starts only after the GPU is free.

Runs, in order:
  1. E4 cross-domain: train plain + ms_refine_uncond on LSD only, select on LOL
  2. E4 control:      train plain on LOL only (in-domain reference)
  3. E4 eval:         score all of the above on LOL-v1 eval15
  4. E5 log-domain L1 ablation: train both variants with the extra log-space term
  5. E5 eval:         score the ablation checkpoints

The script waits for the detector fine-tuning job to finish first, because that
job owns the GPU and two concurrent trainings would slow both down and could
push an 8 GB card into an out-of-memory failure.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\33848\论文投稿\STAGE6_v0.8")
CD = ROOT / "crossdomain"
PY_PROJECT = Path(r"C:\Users\33848\Documents\ChatGPT\new\experiments\.venv\Scripts\python.exe")
PY_SYS = Path(sys.executable)

# a file the detector training writes near the end; used as a "GPU free" signal
DETECTOR_MARKERS = [
    Path(r"C:\Users\33848\论文投稿\darkdriving\work_seeds\seed_training_summary.json"),
]


def wait_for_gpu_free(max_wait_s: int = 7200, poll: int = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if all(m.exists() for m in DETECTOR_MARKERS):
            print("detector training finished; GPU queue is clear", flush=True)
            return
        done = [m.exists() for m in DETECTOR_MARKERS]
        print("  waiting for detector training... (%ds, markers=%s)"
              % (int(time.time() - t0), done), flush=True)
        time.sleep(poll)
    print("WARNING: wait timed out; proceeding anyway", flush=True)


def run(cmd: list[str], label: str) -> bool:
    print("\n" + "=" * 88, flush=True)
    print(">>> %s" % label, flush=True)
    print(">>> " + " ".join(str(c) for c in cmd), flush=True)
    print("=" * 88, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (r.stdout or "").strip().splitlines()[-25:]:
        print("   " + line, flush=True)
    if r.returncode != 0:
        print("   FAILED rc=%d" % r.returncode, flush=True)
        print("   stderr tail:", (r.stderr or "")[-1200:], flush=True)
        return False
    return True


def main() -> int:
    wait_for_gpu_free()

    steps = [
        ([PY_PROJECT, CD / "run_crossdomain.py", "--phase", "cross", "--seed", "20260830"],
         "E4 cross-domain training (LSD -> LOL): plain + ms_refine_uncond"),
        ([PY_PROJECT, CD / "run_crossdomain.py", "--phase", "in_domain", "--seed", "20260830"],
         "E4 in-domain control (LOL only)"),
        ([PY_PROJECT, CD / "eval_crossdomain.py", "--seed", "20260830"],
         "E4 evaluation on LOL-v1 eval15 (includes the main clean models for context)"),
        ([PY_PROJECT, CD / "run_crossdomain.py", "--phase", "log_domain", "--seed", "20260830"],
         "E5 log-domain L1 ablation training"),
        ([PY_PROJECT, CD / "eval_crossdomain.py", "--seed", "20260830"],
         "E5 evaluation"),
    ]

    ok = 0
    for cmd, label in steps:
        if run(cmd, label):
            ok += 1

    print("\n" + "=" * 88)
    print("batch finished: %d/%d steps succeeded" % (ok, len(steps)))
    return 0 if ok == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
