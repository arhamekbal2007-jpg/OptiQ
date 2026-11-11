"""Reproduce every result file:  python run_all.py   (about 2-3 minutes on one CPU core)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
(ROOT / "results_extra").mkdir(exist_ok=True)

sys.path.insert(0, str(SRC))
from common import SEEDS, dump_json, environment_info  # noqa: E402

for name, args in [("stage1.py", []), ("stage2.py", []), ("stage3.py", ["100"])]:
    t = time.time()
    print(f"=== {name} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / name), *args], check=True, cwd=SRC)
    print(f"    ({time.time() - t:.0f}s)")

dump_json({"seeds": SEEDS, "environment": environment_info()}, ROOT / "results_extra" / "environment.json")
print("done. Result files: stage1_results.csv, stage2_results.json (+stage2_candidates.csv), stage3_results.json")
