"""
MuleGuard AI — environment verifier.

Run after installing requirements.txt:
    python verify_env.py

Confirms the Python version is sane and every required library imports,
printing each one's version. Exits non-zero if anything is missing.
"""

import importlib
import sys

# (import name, friendly label)
REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("imblearn", "imbalanced-learn"),
    ("shap", "shap"),
    ("networkx", "networkx"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("joblib", "joblib"),
    ("tqdm", "tqdm"),
    ("openpyxl", "openpyxl"),
    ("pyarrow", "pyarrow"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
]


def main() -> int:
    print("=" * 52)
    print(" MuleGuard AI — Environment Check")
    print("=" * 52)

    # --- Python version guidance ---
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if v.major == 3 and v.minor in (11, 12):
        print("  -> Python version OK (3.11/3.12 recommended).")
    else:
        print("  -> WARNING: 3.11 or 3.12 recommended. ML wheels may be")
        print("     unavailable on this version (e.g. 3.14). See SETUP.md.")
    print("-" * 52)

    # --- Library imports ---
    failures = []
    for mod, label in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            print(f"  OK    {label:<20} {ver}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  FAIL  {label:<20} {exc}")

    print("=" * 52)
    if failures:
        print(f"MISSING/BROKEN: {', '.join(failures)}")
        print("Fix: activate the venv and re-run:")
        print("     python -m pip install -r requirements.txt")
        print("(LightGBM/XGBoost failures on macOS usually mean: brew install libomp)")
        return 1

    print("ALL IMPORTS OK")
    print("Next: place DataSet.csv in ./data/ (or the project root), then run")
    print("      .\\run.ps1   on Windows,  or  ./run.sh   on macOS/Linux.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
