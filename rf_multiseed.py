"""
benchmarks/rf_multiseed/run.py
==============================
Multi-seed benchmark: YO Hybrid vs default hyperparameters on Random Forest
regression (synthetic make_regression dataset).

Runs N independent experiments with different random seeds to demonstrate
consistent improvement over baseline.

Usage
-----
    python benchmarks/rf_multiseed/run.py
"""

import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from yo_hybrid import YOHybridOptimizer

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val):
    """Return a callable that maps a param-dict → validation R²."""

    def objective(params: dict) -> float:
        try:
            model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=float(params["max_features"]),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            return r2_score(y_val, model.predict(X_val))
        except Exception:
            return -1.0

    return objective


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(seed: int, verbose: bool = False) -> dict:
    """Run one independent experiment and return metric dict."""

    # ---- Data ----
    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=15, noise=10.0, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    def eval_on_test(params=None):
        model = (
            RandomForestRegressor(random_state=42, n_jobs=-1)
            if params is None
            else RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=float(params["max_features"]),
                random_state=42,
                n_jobs=-1,
            )
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds), np.sqrt(mean_squared_error(y_test, preds))

    # ---- Baseline ----
    r2_base, rmse_base = eval_on_test()

    # ---- YO Hybrid ----
    param_space = {
        "n_estimators":      (50, 200, "int"),
        "max_depth":         (5, 30, "int"),
        "min_samples_split": (2, 20, "int"),
        "min_samples_leaf":  (1, 10, "int"),
        "max_features":      (0.3, 1.0, "float"),
    }
    objective = make_objective(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt)
    opt = YOHybridOptimizer(
        param_space,
        n_iterations=60,
        n_chains=3,
        initial_temp=5.0,
        cooling_rate=0.95,
        burnin_ratio=0.25,
        stagnation_threshold=8,
        reheating_factor=2.5,
        seed=seed,
    )
    best_params, best_val = opt.optimize_multichain(objective, verbose=verbose)
    r2_opt, rmse_opt = eval_on_test(best_params)

    if verbose:
        print(f"  Seed {seed}: baseline R²={r2_base:.4f} → optimised R²={r2_opt:.4f}")

    return {
        "seed":          seed,
        "r2_baseline":   r2_base,
        "rmse_baseline": rmse_base,
        "r2_optimized":  r2_opt,
        "rmse_optimized":rmse_opt,
        "best_params":   best_params,
        "best_val_score":best_val,
    }


# ---------------------------------------------------------------------------
# Multi-seed run
# ---------------------------------------------------------------------------

def run_multi_seed(seeds=(42, 142, 242, 342, 442), verbose: bool = True) -> list:
    print("\n" + "=" * 70)
    print(f"MULTI-SEED BENCHMARK  ({len(seeds)} seeds)")
    print("=" * 70)
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n>>> Run {i+1}/{len(seeds)}  (seed={seed})")
        res = run_single(seed, verbose=verbose)
        results.append(res)
        delta = res["r2_optimized"] - res["r2_baseline"]
        print(
            f"    Baseline : R²={res['r2_baseline']:.4f}  RMSE={res['rmse_baseline']:.4f}"
        )
        print(
            f"    YO-Opt   : R²={res['r2_optimized']:.4f}  RMSE={res['rmse_optimized']:.4f}"
        )
        print(f"    ΔR²      : {delta:+.4f}")
    return results


# ---------------------------------------------------------------------------
# Table + plot
# ---------------------------------------------------------------------------

def print_table(results: list):
    df = pd.DataFrame(results)
    display = pd.DataFrame(
        {
            "Run":          range(1, len(results) + 1),
            "Seed":         df["seed"],
            "Baseline R²":  df["r2_baseline"].round(4),
            "Baseline RMSE":df["rmse_baseline"].round(4),
            "YO-Opt R²":    df["r2_optimized"].round(4),
            "YO-Opt RMSE":  df["rmse_optimized"].round(4),
            "ΔR²":          (df["r2_optimized"] - df["r2_baseline"]).round(4),
        }
    )
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(display.to_string(index=False))

    avg_base = df["r2_baseline"].mean()
    avg_opt  = df["r2_optimized"].mean()
    wins     = (df["r2_optimized"] > df["r2_baseline"]).sum()
    print(f"\nAvg Baseline R²   : {avg_base:.4f} ± {df['r2_baseline'].std():.4f}")
    print(f"Avg YO-Opt R²     : {avg_opt:.4f}  ± {df['r2_optimized'].std():.4f}")
    print(f"Avg Improvement   : {avg_opt - avg_base:+.4f}")
    print(f"Win rate          : {wins}/{len(results)}")


def plot_results(results: list):
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x  = np.arange(len(results))
    w  = 0.35

    # R² comparison
    ax = axes[0, 0]
    ax.bar(x - w / 2, df["r2_baseline"], w, label="Baseline", color="#3498db", alpha=0.8)
    ax.bar(x + w / 2, df["r2_optimized"], w, label="YO-Opt", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(results))])
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score: Baseline vs YO-Optimised")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # RMSE comparison
    ax = axes[0, 1]
    ax.bar(x - w / 2, df["rmse_baseline"], w, label="Baseline", color="#3498db", alpha=0.8)
    ax.bar(x + w / 2, df["rmse_optimized"], w, label="YO-Opt", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(results))])
    ax.set_ylabel("RMSE  (lower is better)")
    ax.set_title("RMSE: Baseline vs YO-Optimised")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Improvement trend
    ax = axes[1, 0]
    improvement = df["r2_optimized"] - df["r2_baseline"]
    ax.plot(range(1, len(results) + 1), improvement, marker="o", linewidth=2, color="#27ae60")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(improvement.mean(), color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {improvement.mean():.4f}")
    ax.set_xlabel("Run")
    ax.set_ylabel("ΔR² (optimised − baseline)")
    ax.set_title("R² Improvement per Run")
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plots
    ax = axes[1, 1]
    bp = ax.boxplot(
        [df["r2_baseline"], df["r2_optimized"]],
        labels=["Baseline", "YO-Optimised"],
        patch_artist=True,
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("rf_multiseed_results.png", dpi=150)
    plt.show()
    print("Saved: rf_multiseed_results.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEEDS = [42, 142, 242, 342, 442]
    results = run_multi_seed(seeds=SEEDS, verbose=True)
    print_table(results)
    plot_results(results)


if __name__ == "__main__":
    main()
