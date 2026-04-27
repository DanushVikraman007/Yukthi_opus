"""
benchmarks/poiseuille/run.py
============================
2D Poiseuille (laminar channel) flow optimisation.

Goal: recover physical parameters (viscosity, inlet velocity, time-step)
that best match a target velocity field generated with known "true" values.

Uses YO Hybrid Optimizer vs Random Search as baseline.

Usage
-----
    python benchmarks/poiseuille/run.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)

print("2D Poiseuille Flow Optimisation — loading...")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PoiseuilleFlowSimulator:
    """
    Analytical 2D Poiseuille flow.

    u(y) = (dP/dx) / (2*mu) * y * (H - y)
    where dP/dx = 12 * mu * U_avg / H^2
    """

    def __init__(self, nx=50, ny=50, length=1.0, height=0.5):
        self.nx, self.ny = nx, ny
        self.length, self.height = length, height
        self.x = np.linspace(0, length, nx)
        self.y = np.linspace(0, height, ny)
        self.dx = length / (nx - 1)
        self.dy = height / (ny - 1)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def simulate(self, viscosity: float, inlet_velocity: float, dt: float) -> np.ndarray:
        """Return velocity field (ny × nx)."""
        dp_dx = 12 * viscosity * inlet_velocity / self.height**2
        u = np.zeros((self.ny, self.nx))
        for i, yc in enumerate(self.y):
            u[i, :] = (dp_dx / (2 * viscosity)) * yc * (self.height - yc)
        return u * (1.0 - 0.1 * dt)

    def generate_target(
        self,
        viscosity=0.01,
        inlet_velocity=1.0,
        dt=0.01,
        noise=0.02,
    ) -> np.ndarray:
        target = self.simulate(viscosity, inlet_velocity, dt)
        if noise > 0:
            target += np.random.normal(0, noise * np.max(target), target.shape)
            target = np.maximum(target, 0)
        return target


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def l2_error(pred, target):
    return float(np.sqrt(np.sum((pred - target) ** 2)))

def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def r2(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0

def norm_error_pct(pred, target):
    mx = np.max(np.abs(target))
    return float(np.mean(np.abs(pred - target)) / mx * 100) if mx else 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    best_value:       float
    best_params:      np.ndarray
    history:          List[float] = field(default_factory=list)
    best_history:     List[float] = field(default_factory=list)
    runtime:          float = 0.0
    evaluation_count: int = 0
    method_name:      str = "Unknown"
    r2_score:         float = 0.0
    rmse_val:         float = 0.0
    norm_error:       float = 0.0
    metadata:         Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YO Hybrid Optimizer
# ---------------------------------------------------------------------------

class YOHybridOptimizer:
    """YO Hybrid for Poiseuille parameter recovery (minimisation of L2 error)."""

    def __init__(
        self,
        objective_func: Callable,
        target_field: np.ndarray,
        bounds: List[Tuple[float, float]],
        param_names: List[str],
        n_chains: int = 3,
        burnin_ratio: float = 0.25,
        reheating_interval: int = 25,
        blacklist_tol: float = 0.05,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.96,
        reheating_factor: float = 1.8,
        verbose: bool = True,
    ):
        self.objective_func    = objective_func
        self.target_field      = target_field
        self.bounds            = np.array(bounds)
        self.param_names       = param_names
        self.dim               = len(bounds)
        self.n_chains          = n_chains
        self.burnin_ratio      = burnin_ratio
        self.reheating_interval= reheating_interval
        self.blacklist_tol     = blacklist_tol
        self.initial_temp      = initial_temp
        self.cooling_rate      = cooling_rate
        self.reheating_factor  = reheating_factor
        self.verbose           = verbose

        self.blacklist:    list = []
        self.history:      List[float] = []
        self.best_history: List[float] = []
        self.best_value:   float = float("inf")
        self.best_params:  Optional[np.ndarray] = None
        self.best_field:   Optional[np.ndarray] = None
        self.evals_used:   int = 0
        self.chains:       list = []

    def _blacklisted(self, p):
        p_norm = (p - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        for b in self.blacklist:
            b_norm = (b - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
            if np.linalg.norm(p_norm - b_norm) < self.blacklist_tol:
                return True
        return False

    def _clip(self, p):
        return np.clip(p, self.bounds[:, 0], self.bounds[:, 1])

    def _evaluate(self, p):
        field = self.objective_func(p)
        err   = l2_error(field, self.target_field)
        self.history.append(err)
        self.evals_used += 1
        if err < self.best_value:
            self.best_value  = err
            self.best_params = p.copy()
            self.best_field  = field.copy()
        self.best_history.append(self.best_value)
        return err, field

    def _mcmc_step(self, p, T, step_size):
        step = step_size * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        proposal = self._clip(p + np.random.normal(0, step))
        if self._blacklisted(proposal):
            cv, _ = self._evaluate(p)
            return p, cv
        cv, _ = self._evaluate(p)
        pv, _ = self._evaluate(proposal)
        delta  = pv - cv
        if np.random.random() < np.exp(-delta / (T + 1e-10)):
            return proposal, pv
        return p, cv

    def _greedy_step(self, p, n_nb, step_size):
        step = step_size * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.05
        bp, bv = p, self._evaluate(p)[0]
        for _ in range(n_nb):
            nb = self._clip(bp + np.random.normal(0, step))
            if not self._blacklisted(nb):
                v, _ = self._evaluate(nb)
                if v < bv:
                    bp, bv = nb, v
        return bp, bv

    def _sa_step(self, p, T, step_size):
        step = step_size * T * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.08
        proposal = self._clip(p + np.random.normal(0, step))
        if self._blacklisted(proposal):
            cv, _ = self._evaluate(p)
            return p, cv
        cv, _ = self._evaluate(p)
        pv, _ = self._evaluate(proposal)
        delta  = pv - cv
        if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
            return proposal, pv
        return p, cv

    def optimize(self, n_evaluations: int) -> OptimizationResult:
        t0 = time.time()
        if self.verbose:
            print("=" * 70)
            print(f"YO HYBRID — Poiseuille  params={self.param_names}  budget={n_evaluations}")
            print("=" * 70)

        # Init chains
        for _ in range(self.n_chains):
            p = np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])
            self.chains.append(p)
            self._evaluate(p)

        # Burnin
        burnin = int(n_evaluations * self.burnin_ratio)
        T      = self.initial_temp
        burnin_samples = []
        for i in range(burnin // self.n_chains):
            for ci in range(self.n_chains):
                self.chains[ci], err = self._mcmc_step(self.chains[ci], T, 1.0)
                burnin_samples.append((self.chains[ci].copy(), err))
            T *= self.cooling_rate
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Burnin iter {i+1}  best_L2={self.best_value:.6f}  T={T:.4f}")

        burnin_samples.sort(key=lambda s: s[1])
        self.chains = [s[0] for s in burnin_samples[: self.n_chains]]
        if self.verbose:
            print(f"\n  Burnin done. best_L2={self.best_value:.6f}\n")

        # Hybrid
        T, it = self.initial_temp, 0
        while self.evals_used < n_evaluations:
            it += 1
            for ci in range(self.n_chains):
                if self.evals_used >= n_evaluations:
                    break
                self.chains[ci], _ = self._mcmc_step(self.chains[ci], T * 0.6, 0.8)

            if self.evals_used < n_evaluations and it % 5 == 0:
                chain_errs = [self._evaluate(c)[0] for c in self.chains]
                best_ci    = int(np.argmin(chain_errs))
                n_nb = min(4, (n_evaluations - self.evals_used) // 3)
                if n_nb > 0:
                    self.chains[best_ci], _ = self._greedy_step(self.chains[best_ci], n_nb, 0.8)

            if it % self.reheating_interval == 0:
                T = min(self.initial_temp, T * self.reheating_factor)
                if self.verbose:
                    print(f"  Reheating iter={it}  T={T:.4f}")
            else:
                T *= self.cooling_rate

            for ci in range(self.n_chains):
                if self.evals_used >= n_evaluations:
                    break
                self.chains[ci], _ = self._sa_step(self.chains[ci], T, 0.8)

            if it % 20 == 0:
                chain_errs = [self._evaluate(c)[0] for c in self.chains]
                worst = int(np.argmax(chain_errs))
                self.blacklist.append(self.chains[worst].copy())
                self.chains[worst] = np.array([
                    np.random.uniform(lo, hi) for lo, hi in self.bounds
                ])

            if self.verbose and it % 15 == 0:
                params_str = ", ".join(
                    f"{self.param_names[j]}={self.best_params[j]:.4f}" for j in range(self.dim)
                )
                print(f"  iter={it}  evals={self.evals_used}/{n_evaluations}  best_L2={self.best_value:.6f}  [{params_str}]")

        runtime = time.time() - t0
        r2_val   = r2(self.best_field, self.target_field)
        rmse_val = rmse(self.best_field, self.target_field)
        ne       = norm_error_pct(self.best_field, self.target_field)

        if self.verbose:
            print(f"\n  Done. L2={self.best_value:.6f}  R²={r2_val:.6f}  RMSE={rmse_val:.6f}  time={runtime:.2f}s")
            for j, nm in enumerate(self.param_names):
                print(f"    {nm}: {self.best_params[j]:.6f}")

        return OptimizationResult(
            best_value=self.best_value,
            best_params=self.best_params,
            history=self.history,
            best_history=self.best_history,
            runtime=runtime,
            evaluation_count=self.evals_used,
            method_name="YO Hybrid",
            r2_score=r2_val,
            rmse_val=rmse_val,
            norm_error=ne,
            metadata={"n_chains": self.n_chains, "blacklist_size": len(self.blacklist)},
        )


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------

def random_search(objective_func, target_field, bounds, param_names, n_evaluations, verbose=True):
    bounds_arr = np.array(bounds)
    best_v, best_p, best_f = float("inf"), None, None
    history, best_history = [], []
    t0 = time.time()

    for i in range(n_evaluations):
        p = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        field = objective_func(p)
        err   = l2_error(field, target_field)
        history.append(err)
        if err < best_v:
            best_v, best_p, best_f = err, p.copy(), field.copy()
        best_history.append(best_v)
        if verbose and (i + 1) % 50 == 0:
            print(f"  eval {i+1}/{n_evaluations}  best_L2={best_v:.6f}")

    runtime = time.time() - t0
    r2_val   = r2(best_f, target_field)
    rmse_val = rmse(best_f, target_field)
    ne       = norm_error_pct(best_f, target_field)

    if verbose:
        print(f"  Done. L2={best_v:.6f}  R²={r2_val:.6f}  time={runtime:.2f}s")

    return OptimizationResult(
        best_value=best_v, best_params=best_p, history=history,
        best_history=best_history, runtime=runtime, evaluation_count=n_evaluations,
        method_name="Random Search", r2_score=r2_val, rmse_val=rmse_val, norm_error=ne,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_convergence(rs, yo, save="poiseuille_convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(rs.best_history, label="Random Search", linewidth=2, color="orange")
    ax.plot(yo.best_history, label="YO Hybrid",    linewidth=2, color="blue")
    ax.set_yscale("log"); ax.set_title("Best-So-Far Convergence")
    ax.set_xlabel("Evaluation"); ax.set_ylabel("L2 Error"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    w = 20
    ax.plot(pd.Series(rs.history).rolling(w, 1).mean(), label="Random Search", linewidth=2, color="orange")
    ax.plot(pd.Series(yo.history).rolling(w, 1).mean(), label="YO Hybrid",    linewidth=2, color="blue")
    ax.set_yscale("log"); ax.set_title(f"Smoothed History (w={w})")
    ax.set_xlabel("Evaluation"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(rs.history, bins=40, alpha=0.6, label="Random Search", color="orange", density=True)
    ax.hist(yo.history, bins=40, alpha=0.6, label="YO Hybrid",    color="blue",   density=True)
    ax.set_xlabel("L2 Error"); ax.set_ylabel("Density")
    ax.set_title("Distribution of Sampled Errors"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n   = min(len(rs.best_history), len(yo.best_history))
    imp = (np.array(rs.best_history[:n]) - np.array(yo.best_history[:n])) / (np.array(rs.best_history[:n]) + 1e-10) * 100
    ax.plot(imp, linewidth=2, color="green"); ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.fill_between(range(n), 0, imp, where=imp > 0, alpha=0.3, color="green")
    ax.set_xlabel("Evaluation"); ax.set_ylabel("Improvement (%)")
    ax.set_title("YO Hybrid vs Baseline Improvement"); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save, dpi=150); plt.show(); print(f"Saved: {save}")


def plot_fields(sim, target, baseline_params, yo_params, save="poiseuille_fields.png"):
    baseline_field = sim.simulate(*baseline_params)
    yo_field       = sim.simulate(*yo_params)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _cf(ax, field, title):
        im = ax.contourf(sim.X, sim.Y, field, levels=20, cmap="viridis")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.set_aspect("equal"); plt.colorbar(im, ax=ax, label="Velocity")

    _cf(axes[0, 0], target,        "Target")
    _cf(axes[0, 1], baseline_field,"Random Search")
    _cf(axes[0, 2], yo_field,      "YO Hybrid")

    def _err(ax, pred, title):
        err = np.abs(target - pred)
        im  = ax.contourf(sim.X, sim.Y, err, levels=20, cmap="Reds")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.set_aspect("equal"); plt.colorbar(im, ax=ax, label="Abs Error")

    _err(axes[1, 0], baseline_field, "Random Search Error")
    _err(axes[1, 1], yo_field,       "YO Hybrid Error")

    ax = axes[1, 2]
    ci  = target.shape[1] // 2
    ax.plot(target[:, ci],        sim.y, label="Target",        color="black", linewidth=3)
    ax.plot(baseline_field[:, ci],sim.y, label="Random Search", color="orange", linestyle="--")
    ax.plot(yo_field[:, ci],      sim.y, label="YO Hybrid",     color="blue",   linestyle="-.")
    ax.set_xlabel("Velocity"); ax.set_ylabel("y"); ax.set_title("Profile at Centre")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save, dpi=150); plt.show(); print(f"Saved: {save}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TARGET_VISCOSITY = 0.015
    TARGET_VELOCITY  = 1.2
    TARGET_DT        = 0.012
    N_EVALS          = 300
    GRID             = 50

    print("=" * 70)
    print(f"POISEUILLE FLOW BENCHMARK  grid={GRID}×{GRID}  budget={N_EVALS}")
    print(f"Target: viscosity={TARGET_VISCOSITY}  velocity={TARGET_VELOCITY}  dt={TARGET_DT}")
    print("=" * 70)

    sim    = PoiseuilleFlowSimulator(nx=GRID, ny=GRID)
    target = sim.generate_target(TARGET_VISCOSITY, TARGET_VELOCITY, TARGET_DT, noise=0.02)
    print(f"Target field: max_velocity={np.max(target):.4f}")

    bounds     = [(0.001, 0.05), (0.5, 2.0), (0.005, 0.02)]
    param_names= ["viscosity", "inlet_velocity", "dt"]
    objective  = lambda p: sim.simulate(*p)

    # Random Search
    print("\nRandom Search...")
    rs = random_search(objective, target, bounds, param_names, N_EVALS, verbose=True)

    # YO Hybrid
    print("\nYO Hybrid...")
    opt = YOHybridOptimizer(
        objective, target, bounds, param_names,
        n_chains=3, burnin_ratio=0.25, reheating_interval=25,
        blacklist_tol=0.05, initial_temp=1.0, cooling_rate=0.96,
        reheating_factor=1.8, verbose=True,
    )
    yo = opt.optimize(N_EVALS)

    # Summary table
    imp = (rs.best_value - yo.best_value) / rs.best_value * 100
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    df = pd.DataFrame(
        {
            "Metric":        ["L2 Error", "R²", "RMSE", "Norm Error %", "Runtime (s)"],
            "Random Search": [rs.best_value, rs.r2_score, rs.rmse_val, rs.norm_error, rs.runtime],
            "YO Hybrid":     [yo.best_value, yo.r2_score, yo.rmse_val, yo.norm_error, yo.runtime],
        }
    )
    print(df.to_string(index=False))
    print(f"\nImprovement: {imp:.2f}%")

    for j, nm in enumerate(param_names):
        tv = [TARGET_VISCOSITY, TARGET_VELOCITY, TARGET_DT][j]
        yv = yo.best_params[j]
        print(f"  {nm}: target={tv:.6f}  found={yv:.6f}  err={abs(yv-tv)/tv*100:.2f}%")

    df.to_csv("poiseuille_results.csv", index=False)
    print("Saved: poiseuille_results.csv")

    # Plots
    plot_convergence(rs, yo)
    plot_fields(sim, target, rs.best_params, yo.best_params)


if __name__ == "__main__":
    main()
