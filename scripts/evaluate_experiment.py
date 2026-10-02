"""CUPED-adjusted experiment evaluation pipeline for DPE simulation data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as ``python scripts/evaluate_experiment.py``
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.data.dpe_connector import describe_switchback, load_dpe_data
from src.data.synthetic import generate_uplift_dataset
from src.experiments.power_analysis import cuped_adjust, sample_size_calculator


def evaluate_experiment(
    source: str = "synthetic",
    dpe_db_path: str | Path | None = None,
    covariate_col: str = "past_trips",
    target_col: str = "accepted",
) -> dict:
    """Evaluate experiment with CUPED variance reduction and hypothesis testing.

    Args:
        source: ``synthetic`` evaluates the randomized surcharge. ``dpe`` describes the switchback and does not test rows.
        dpe_db_path: Path to DPE SQLite database.
        covariate_col: Pre-experiment covariate for CUPED adjustment.
        target_col: Experiment outcome variable name.

    Returns:
        Dict with metrics, confidence intervals, p-values, and formatted markdown report.
    """
    cfg = load_config()

    if source.lower() == "dpe":
        frame = load_dpe_data(db_path=dpe_db_path)
        design = describe_switchback(frame)
        return {
            "source": "dpe",
            "identified_ite": False,
            "design": design,
            "markdown_report": design["report"],
        }

    df = generate_uplift_dataset(n=cfg.data.n_samples, random_state=cfg.data.random_state)

    if covariate_col not in df.columns:
        raise ValueError(f"Covariate '{covariate_col}' not found in dataframe columns.")
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in dataframe columns.")

    # Control vs Treatment masks
    mask_c = df["treatment"] == 0
    mask_t = df["treatment"] == 1

    n_c = int(mask_c.sum())
    n_t = int(mask_t.sum())

    if n_c == 0 or n_t == 0:
        raise ValueError("Dataset must contain both control (0) and treatment (1) units.")

    y_raw = df[target_col].to_numpy(dtype=float)
    y_c_raw = y_raw[mask_c]
    y_t_raw = y_raw[mask_t]

    mean_c_raw = float(np.mean(y_c_raw))
    mean_t_raw = float(np.mean(y_t_raw))
    raw_delta = mean_t_raw - mean_c_raw
    var_raw = float(np.var(y_raw, ddof=1))

    # Apply CUPED adjustment
    y_cuped = cuped_adjust(df, target_col=target_col, covariate_col=covariate_col).to_numpy(dtype=float)
    y_c_cuped = y_cuped[mask_c]
    y_t_cuped = y_cuped[mask_t]

    mean_c_cuped = float(np.mean(y_c_cuped))
    mean_t_cuped = float(np.mean(y_t_cuped))
    cuped_delta = mean_t_cuped - mean_c_cuped
    var_cuped = float(np.var(y_cuped, ddof=1))

    variance_reduction_pct = (
        (var_raw - var_cuped) / var_raw * 100.0 if var_raw > 0 else 0.0
    )

    # Hypothesis testing on CUPED-adjusted outcome (Welch's t-test)
    t_stat, p_val = stats.ttest_ind(y_t_cuped, y_c_cuped, equal_var=False)
    p_value = float(p_val)

    # 95% Confidence Interval for CUPED delta
    se_diff = np.sqrt(
        np.var(y_t_cuped, ddof=1) / n_t + np.var(y_c_cuped, ddof=1) / n_c
    )
    z_crit = stats.norm.ppf(1.0 - cfg.experiment.alpha / 2.0)
    ci_lower = float(cuped_delta - z_crit * se_diff)
    ci_upper = float(cuped_delta + z_crit * se_diff)

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((n_t - 1) * np.var(y_t_cuped, ddof=1) + (n_c - 1) * np.var(y_c_cuped, ddof=1))
        / (n_t + n_c - 2)
    )
    cohens_d = float(cuped_delta / pooled_std) if pooled_std > 0 else 0.0

    # Required sample size per group
    std_dev_for_calc = float(np.std(y_cuped, ddof=1))
    req_n_per_group = sample_size_calculator(
        std_dev=std_dev_for_calc if std_dev_for_calc > 0 else 1.0,
        mde=cfg.experiment.mde,
        alpha=cfg.experiment.alpha,
        power=cfg.experiment.power,
    )

    actual_min_n = min(n_c, n_t)
    is_underpowered = actual_min_n < req_n_per_group
    is_significant = p_value < cfg.experiment.alpha

    # Format Markdown Report
    report = f"""
## Experiment Evaluation Report

| Metric                          | Value                    |
|:--------------------------------|:-------------------------|
| Data Source                     | `{source}`               |
| Control Sample Size (n_c)       | {n_c}                    |
| Treatment Sample Size (n_t)     | {n_t}                    |
| Control acceptance rate         | {mean_c_raw:.4f}         |
| Treated acceptance rate         | {mean_t_raw:.4f}         |
| Raw Delta                       | {raw_delta:+.4f}         |
| CUPED Delta ({covariate_col})    | {cuped_delta:+.4f}        |
| p-value (CUPED t-test)          | {p_value:.4f}            |
| 95% Confidence Interval         | [{ci_lower:+.4f}, {ci_upper:+.4f}] |
| Cohen's d Effect Size           | {cohens_d:.4f}           |
| Variance Reduction (CUPED)      | {variance_reduction_pct:.2f}% |
| Verdict                         | {'[+] SIGNIFICANT' if is_significant else '[-] NOT SIGNIFICANT'} |

### Sample Size & Power Check
- Required sample size per group (alpha={cfg.experiment.alpha}, power={cfg.experiment.power}, mde={cfg.experiment.mde}): **{req_n_per_group}**
- Current smallest group sample size: **{actual_min_n}**
"""

    if is_underpowered:
        diff_n = req_n_per_group - actual_min_n
        report += f"\n[!] Warning: Experiment is underpowered. Need ~{diff_n} more observations per group to reliably detect MDE={cfg.experiment.mde}.\n"
    else:
        report += "\n[+] Power Check Passed: Current sample size is sufficient to detect specified MDE.\n"

    res = {
        "source": source,
        "n_c": n_c,
        "n_t": n_t,
        "mean_c_raw": mean_c_raw,
        "mean_t_raw": mean_t_raw,
        "raw_delta": raw_delta,
        "cuped_delta": cuped_delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
        "variance_reduction_pct": variance_reduction_pct,
        "req_n_per_group": req_n_per_group,
        "is_underpowered": is_underpowered,
        "is_significant": is_significant,
        "markdown_report": report,
    }
    return res


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate experiment with CUPED variance reduction.")
    parser.add_argument(
        "--source",
        choices=["dpe", "synthetic"],
        default="synthetic",
        help="synthetic (default) is the randomized surcharge. dpe prints the switchback design and does not run a row-level test.",
    )
    parser.add_argument(
        "--dpe-db-path",
        type=str,
        default=None,
        help="Optional custom path to DPE SQLite database",
    )
    parser.add_argument(
        "--covariate",
        type=str,
        default="past_trips",
        help="Covariate column for CUPED adjustment (default: past_trips)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional filepath to save markdown evaluation report",
    )
    args = parser.parse_args()

    results = evaluate_experiment(
        source=args.source,
        dpe_db_path=args.dpe_db_path,
        covariate_col=args.covariate,
    )
    print(results["markdown_report"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(results["markdown_report"], encoding="utf-8")
        print(f"\n[+] Evaluation report written to {out_path}")



if __name__ == "__main__":
    main()
