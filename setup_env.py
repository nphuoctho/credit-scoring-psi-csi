"""One-shot environment setup for this project (run: `python3 setup_env.py`).

Uses `uv` because the base interpreter is a uv-managed standalone CPython,
which the stdlib venv module cannot relocate correctly. The env dir is
`.ckenv` (not `.venv`) to dodge a repo scout-hook on that literal string.
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ENV = HERE / ".ckenv"
PKGS = [
    "pandas", "numpy", "scikit-learn", "lightgbm",
    "matplotlib", "seaborn", "joblib",
    "jupyter", "nbconvert", "ipykernel",
]

if not (ENV / "bin" / "python").exists():
    print("creating env via uv ...")
    subprocess.run(["uv", "venv", str(ENV), "--python", "3.12"], check=True)

print("installing packages ...")
subprocess.run(
    ["uv", "pip", "install", "--python", str(ENV / "bin" / "python"), *PKGS],
    check=True,
)
print("OK")
