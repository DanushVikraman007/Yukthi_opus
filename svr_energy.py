"""
benchmarks/svr_energy/run.py
============================
YO Hybrid vs Random Search on the UCI Energy Efficiency dataset
(heating load prediction via SVR hyperparameter tuning).

Dataset
-------
UCI Energy Efficiency (ENB2012):
  https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx
  Features: 8 building characteristics
  Target:   Heating Load (column Y1)

Usage
-----
    python benchmarks/svr_energy/run.py
"""

import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/00242/ENB2012_data.xlsx"
    )
    df = pd.read_excel(url)
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8].values          # Heating Load
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    return X_tr, X_val, y_tr, y_val


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def evaluate_svr(C, gamma, epsilon, X_train, X_val, y_train, y_val) -> float:
    clf = SVR(C=C, gamma=gamma, epsilon=epsilon)
    clf.fit(X_train, y_train)
    return r2_score(y_val, clf.predict(X_val))


# ---------------------------------------------------------------------------
# Baseline: Random Search
# ---------------------------------------------------------------------------

def random_search(n_evals: int, X_train, X_val, y_train, y_val, seed: int = 42):
    rng = random.Random(seed)
    best_score, best_params, scores = -float("inf"), None, []
    t0 = time.time()

    for _ in range(n_evals):
        C       = 10 ** rng.uniform(-1, 2)
        gamma   = 10 ** rng.uniform(-4, -1)
        epsilon = 10 ** rng.uniform(-4, 0)
        score   = evaluate_svr(C, gamma, epsilon, X_train, X_val, y_train, y_val)
        if score > best_score:
            best_score  = score
            best_params = {"C": C, "gamma": gamma, "epsilon": epsilon}
        scores.append(best_score)

    return best_score, best_params, scores, time.time() - t0


# ---------------------------------------------------------------------------
# YO Hybrid Search
# ---------------------------------------------------------------------------

def yo_hybrid_search(
    n_evals: int,
    X_train, X_val, y_train, y_val,
    neighborhood_size: int = 12,
    seed: int = 42,
    initial_T: float = 1.0,
    cooling: float = 0.95,
    W: int = 10,
    alpha: float = 0.05,
    r_reheat: float = 2.0,
):
    rng = random.Random(seed)
    BOUNDS = {"logC": (-1, 2), "logG": (-4, -1), "logE": (-4, 0)}

    def clip(pt):
        return tuple(
            min(max(pt[i], lo), hi)
            for i, (lo, hi) in enumerate(BOUNDS.values())
        )

    def to_params(pt):
        return 10 ** pt[0], 10 ** pt[1], 10 ** pt[2]

    def eval_pt(pt):
        C, gamma, epsilon = to_params(pt)
        return evaluate_svr(C, gamma, epsilon, X_train, X_val, y_train, y_val)

    current_pt = clip(tuple(rng.uniform(lo, hi) for lo, hi in BOUNDS.values()))
    current_score = eval_pt(current_pt)
    best_global, best_pt = current_score, current_pt
    evals_done, scores_trace = 1, [best_global]
    T = initial_T
    accept_history, best_since = [], 0

    while evals_done < n_evals:
        proposal = clip((
            current_pt[0] + rng.gauss(0, 0.3),
            current_pt[1] + rng.gauss(0, 0.3),
            current_pt[2] + rng.gauss(0, 0.3),
        ))

        # Greedy neighbourhood
        best_local_score, best_local_pt = -float("inf"), None
        for _ in range(neighborhood_size):
            if evals_done >= n_evals:
                break
            pt = clip((
                proposal[0] + rng.gauss(0, 0.1),
                proposal[1] + rng.gauss(0, 0.1),
                proposal[2] + rng.gauss(0, 0.1),
            ))
            score = eval_pt(pt)
            evals_done += 1
            if score > best_local_score:
                best_local_score, best_local_pt = score, pt

        if best_local_pt is None:
            break

        # SA acceptance
        delta = best_local_score - current_score
        accepted = delta >= 0 or (T > 0 and rng.random() < math.exp(delta / T))
        if accepted:
            current_pt, current_score = best_local_pt, best_local_score
            accept_history.append(1)
        else:
            accept_history.append(0)

        if current_score > best_global:
            best_global, best_pt = current_score, current_pt
            best_since = 0
        else:
            best_since += 1

        T *= cooling

        # Adaptive reheating
        if len(accept_history) >= W:
            if best_since >= W and sum(accept_history[-W:]) / W < alpha:
                T = T * r_reheat if T * r_reheat > T else T * r_reheat
                current_pt = clip((
                    current_pt[0] + rng.gauss(0, 1),
                    current_pt[1] + rng.gauss(0, 1),
                    current_pt[2] + rng.gauss(0, 1),
                ))
                current_score = eval_pt(current_pt)
                evals_done += 1
                best_since = 0

        scores_trace.append(best_global)

    return best_global, to_params(best_pt), scores_trace, evals_done


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(baseline_scores, yo_scores):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(baseline_scores) + 1), baseline_scores,
             label="Random Search", marker="o", markersize=3)
    plt.plot(range(1, len(yo_scores) + 1), yo_scores,
             label="YO Hybrid", marker="s", markersize=3)
    plt.xlabel("Function evaluations")
    plt.ylabel("Best R² so far")
    plt.title("YO Hybrid vs Random Search — UCI Energy Efficiency (SVR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("svr_energy_comparison.png", dpi=150)
    plt.show()
    print("Saved: svr_energy_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading UCI Energy Efficiency dataset...")
    X_train, X_val, y_train, y_val = load_data()

    BUDGET = 100

    print(f"\nRunning Random Search  (budget={BUDGET})...")
    rs_score, rs_params, rs_trace, rs_time = random_search(
        BUDGET, X_train, X_val, y_train, y_val
    )
    print(f"  Best R² = {rs_score:.4f}  params = {rs_params}  time = {rs_time:.2f}s")

    print(f"\nRunning YO Hybrid      (budget≈{BUDGET})...")
    yo_score, yo_params, yo_trace, yo_evals = yo_hybrid_search(
        BUDGET, X_train, X_val, y_train, y_val
    )
    print(f"  Best R² = {yo_score:.4f}  params = {yo_params}  evals used = {yo_evals}")

    print("\nGenerating plot...")
    plot_comparison(rs_trace, yo_trace)


if __name__ == "__main__":
    main()
