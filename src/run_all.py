"""Run the whole pipeline:  python src/run_all.py   (RAW_DIR / OUT_DIR env vars are inherited)."""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["00_dataset_audit.py", "01_clean.py", "02_build_features.py", "04_validate.py",
         "07_validate_onset_targets.py", "03_eda.py", "08_rule_based_onset_baseline.py",
         "09_logistic_regression_onset.py", "10_random_forest_onset.py",
         "12_xgboost_onset.py", "11_event_level_evaluation.py", "13_fixed_budget_thresholds.py",
         "14_feature_ablation.py", "15_experiment_d_multi_scenario.py",
         "16_bootstrap_confidence_intervals.py"]

for s in STEPS:
    print(f"\n=== {s} ===")
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / s)])
    if r.returncode != 0:
        sys.exit(f"{s} failed with exit code {r.returncode}")
    print(f"--- {s} finished in {time.time() - t:.0f}s")
