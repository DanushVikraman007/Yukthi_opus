"""
benchmarks/multimodal/run.py
============================
Rigorous benchmark: YO Hybrid vs Random Search on a computationally expensive
multi-modal test function (Rastrigin + Rosenbrock + Sphere + sinexp).

Produces 4 publication-ready plots and a CSV summary.

Usage
-----
    python benchmarks/multimodal/run.py
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------

def complex_benchmark_function(x: np.ndarray, delay: float = 0.01) -> float:
    """
    Multi-modal test function combining Rastrigin, Rosenbrock, Sphere, sinexp.
    delay: artificial sleep to simulate expensive evaluation.
    """
    time.sleep(delay)
    d = len(x)
    rastrigin  = 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    rosenbrock = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    sphere     = np.sum(x**2)
    sinexp     = np.sum(np.sin(x) * np.exp(-(x**2) / 10))
    return 0.3 * rastrigin + 0.3 * rosenbrock + 0.2 * sphere + 0.2 * sinexp


# ---------------------------------------------------------------------------
# YO Hybrid Optimizer (continuous, minimisation)
# ---------------------------------------------------------------------------

class YOHybridOptimizer:
    """
    3-layer hybrid: MCMC + greedy local search + SA with reheating.
    Minimises a continuous function over a box constraint.
    """

    def __init__(
        self,
        func: Callable,
        bounds: Tuple[float, float],
        dim: int,
        n_chains: int = 3,
        blacklist_threshold: float = None,
    ):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.n_chains = n_chains
        self.blacklist_threshold = blacklist_threshold
        self.blacklist: list = []
        self.history: List[dict] = []
        self.best_x: np.ndarray = None
        self.best_f: float = np.inf
        self.n_evals: int = 0

    def _in_blacklist(self, x: np.ndarray, tol: float = 0.5) -> bool:
        return any(np.linalg.norm(x - b) < tol for b in self.blacklist)

    def _evaluate(self, x: np.ndarray) -> float:
        if self._in_blacklist(x):
            return np.inf
        f = self.func(x)
        self.n_evals += 1
        self.history.append({"x": x.copy(), "f": f, "eval": self.n_evals})
        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()
        if self.blacklist_threshold and f > self.blacklist_threshold:
            self.blacklist.append(x.copy())
        return f

    def _mcmc_chain(self, x0, n_samples, temperature):
        samples, x, f = [], x0.copy(), self._evaluate(x0)
        for _ in range(n_samples):
            xp = np.clip(x + np.random.randn(self.dim) * 0.5, *self.bounds)
            fp = self._evaluate(xp)
            if fp < f or np.random.rand() < np.exp(-(fp - f) / temperature):
                x, f = xp, fp
            samples.append({"x": x.copy(), "f": f})
        return samples

    def _greedy_search(self, x0, n_steps, step_size=0.3):
        x, f = x0.copy(), self._evaluate(x0)
        for _ in range(n_steps):
            improved = False
            for i in range(self.dim):
                for delta in [step_size, -step_size]:
                    xp = x.copy()
                    xp[i] = np.clip(xp[i] + delta, *self.bounds)
                    fp = self._evaluate(xp)
                    if fp < f:
                        x, f, improved = xp, fp, True
                        break
                if improved:
                    break
            if not improved:
                step_size *= 0.5
                if step_size < 1e-3:
                    break

    def _sa(self, x0, n_iter, T0=10.0):
        x, f = x0.copy(), self._evaluate(x0)
        T, no_improve = T0, 0
        for i in range(n_iter):
            xp = np.clip(x + np.random.randn(self.dim) * np.sqrt(T), *self.bounds)
            fp = self._evaluate(xp)
            if fp < f or np.random.rand() < np.exp(-(fp - f) / T):
                no_improve = 0 if fp >= f else 0
                if fp >= f:
                    no_improve += 1
                else:
                    no_improve = 0
                x, f = xp, fp
            if no_improve > 10:
                T = T0 * 0.5
                no_improve = 0
            else:
                T *= 0.95

    def optimize(self, max_evals: int) -> Tuple[np.ndarray, float]:
        per_phase = max_evals // 3
        all_samples = []
        for _ in range(self.n_chains):
            x0 = np.random.uniform(*self.bounds, self.dim)
            samples = self._mcmc_chain(x0, per_phase // self.n_chains, temperature=1.0)
            all_samples.extend(samples)
            if self.n_evals >= max_evals:
                return self.best_x, self.best_f

        all_samples.sort(key=lambda s: s["f"])
        top = [s["x"] for s in all_samples[:3]]

        for x0 in top:
            self._greedy_search(x0, per_phase // len(top))
            if self.n_evals >= max_evals:
                return self.best_x, self.best_f

        self._sa(self.best_x, max_evals - self.n_evals)
        return self.best_x, self.best_f


# ---------------------------------------------------------------------------
# Random Search baseline
# ---------------------------------------------------------------------------

class RandomSearchOptimizer:
    def __init__(self, func, bounds, dim):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.history: list = []
        self.best_x = None
        self.best_f = np.inf
        self.n_evals = 0

    def optimize(self, max_evals):
        for _ in range(max_evals):
            x = np.random.uniform(*self.bounds, self.dim)
            f = self.func(x)
            self.n_evals += 1
            self.history.append({"x": x.copy(), "f": f, "eval": self.n_evals})
            if f < self.best_f:
                self.best_f, self.best_x = f, x.copy()
        return self.best_x, self.best_f


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    optimizer_name: str
    best_value: float
    runtime: float
    history: list
    n_blacklisted: int = 0


def run_suite(
    n_runs: int = 20,
    dim: int = 5,
    max_evals: int = 150,
    bounds: Tuple[float, float] = (-5, 5),
    delay: float = 0.01,
) -> Dict[str, List[BenchmarkResult]]:

    print(f"Benchmark: {n_runs} runs | dim={dim} | budget={max_evals}")
    print("=" * 70)

    func = lambda x: complex_benchmark_function(x, delay=delay)

    optimizers = {
        "YO_Hybrid":     (YOHybridOptimizer, {"n_chains": 3}),
        "Random_Search": (RandomSearchOptimizer, {}),
    }
    results = {k: [] for k in optimizers}

    for name, (cls, kwargs) in optimizers.items():
        print(f"\nBenchmarking {name}...")
        for run in range(n_runs):
            np.random.seed(run * 42)
            t0 = time.time()
            opt = cls(func, bounds, dim, **kwargs)
            opt.optimize(max_evals)
            elapsed = time.time() - t0
            nb = len(opt.blacklist) if hasattr(opt, "blacklist") else 0
            results[name].append(
                BenchmarkResult(name, opt.best_f, elapsed, opt.history, nb)
            )
            print(f"  Run {run+1}/{n_runs}: best={opt.best_f:.4f}  t={elapsed:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(results: dict) -> pd.DataFrame:
    rows = []
    for name, runs in results.items():
        vals = [r.best_value for r in runs]
        times = [r.runtime for r in runs]
        rows.append(
            {
                "Optimizer":         name,
                "Best Value (mean)": np.mean(vals),
                "Best Value (std)":  np.std(vals),
                "Runtime (mean)":    np.mean(times),
                "Runtime (std)":     np.std(times),
            }
        )
    df = pd.DataFrame(rows)
    if "Random_Search" in results:
        base = df.loc[df["Optimizer"] == "Random_Search", "Best Value (mean)"].values[0]
        df["Improvement (%)"] = (base - df["Best Value (mean)"]) / base * 100
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _best_so_far(history):
    best, out = np.inf, []
    for h in history:
        if h["f"] < best:
            best = h["f"]
        out.append(best)
    return np.array(out)


def plot_convergence(results, save="benchmark_convergence.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in results.items():
        max_len = max(len(r.history) for r in runs)
        mat = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            bsf = _best_so_far(r.history)
            mat[i, : len(bsf)] = bsf
        mean_bsf = np.nanmean(mat, axis=0)
        std_bsf  = np.nanstd(mat, axis=0)
        ax.plot(mean_bsf, label=name, linewidth=2)
        ax.fill_between(range(len(mean_bsf)), mean_bsf - std_bsf, mean_bsf + std_bsf, alpha=0.2)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best Value")
    ax.set_title("Convergence Curves (Mean ± Std)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()
    print(f"Saved: {save}")


def plot_distribution(results, save="benchmark_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    data   = [[r.best_value for r in runs] for runs in results.values()]
    labels = list(results.keys())
    ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Best Function Value")
    ax.set_title("Distribution of Best Values Found")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()
    print(f"Saved: {save}")


def plot_rolling_mean(results, save="benchmark_rolling_mean.png", window=10):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in results.items():
        max_len = max(len(r.history) for r in runs)
        mat = [[h["f"] for h in r.history] + [np.nan] * (max_len - len(r.history))
               for r in runs]
        mean_vals = np.nanmean(mat, axis=0)
        rolling   = pd.Series(mean_vals).rolling(window=window, min_periods=1).mean()
        ax.plot(rolling, label=name, linewidth=2)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(f"Rolling Mean (w={window})")
    ax.set_title("Exploration Behavior")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()
    print(f"Saved: {save}")


def plot_runtime(results, save="benchmark_runtime.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(results.keys())
    means = [np.mean([r.runtime for r in runs]) for runs in results.values()]
    stds  = [np.std([r.runtime for r in runs]) for runs in results.values()]
    ax.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Comparison (Mean ± Std)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()
    print(f"Saved: {save}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_RUNS    = 20
    DIM       = 5
    MAX_EVALS = 150
    BOUNDS    = (-5, 5)
    DELAY     = 0.01

    results  = run_suite(N_RUNS, DIM, MAX_EVALS, BOUNDS, DELAY)
    stats_df = compute_statistics(results)

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(stats_df.to_string(index=False))
    stats_df.to_csv("benchmark_summary.csv", index=False)
    print("\nSaved: benchmark_summary.csv")

    print("\nGenerating plots...")
    plot_convergence(results)
    plot_distribution(results)
    plot_rolling_mean(results)
    plot_runtime(results)

    print("\nOutputs:")
    for f in [
        "benchmark_summary.csv",
        "benchmark_convergence.png",
        "benchmark_distribution.png",
        "benchmark_rolling_mean.png",
        "benchmark_runtime.png",
    ]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
