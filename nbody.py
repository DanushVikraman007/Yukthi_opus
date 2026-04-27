"""
benchmarks/nbody/run.py
=======================
N-Body gravitational simulation optimised with the YO Hybrid Optimizer.

Objective: find initial positions/velocities that minimise RMS energy
conservation error over a Leapfrog time integration.

Usage
-----
    python benchmarks/nbody/run.py
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


# ---------------------------------------------------------------------------
# N-Body Simulation
# ---------------------------------------------------------------------------

class NBodySimulation:
    """
    N-Body gravitational simulation (vectorised NumPy, Leapfrog integrator).
    Objective: minimise RMS energy conservation error.
    """

    def __init__(
        self,
        n_bodies: int = 100,
        G: float = 1.0,
        softening: float = 0.1,
        dt: float = 0.01,
        n_steps: int = 50,
        seed: int = 42,
    ):
        self.n_bodies  = n_bodies
        self.G         = G
        self.softening = softening
        self.dt        = dt
        self.n_steps   = n_steps
        np.random.seed(seed)

    def _accelerations(self, pos: np.ndarray) -> np.ndarray:
        dx = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(dx**2, axis=2) + self.softening**2
        np.fill_diagonal(r2, 1.0)
        r3 = r2 * np.sqrt(r2)
        return self.G * np.sum(dx / r3[:, :, np.newaxis], axis=1)

    def _total_energy(self, pos, vel) -> float:
        ke = 0.5 * np.sum(vel**2)
        dx = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r  = np.sqrt(np.sum(dx**2, axis=2) + self.softening**2)
        np.fill_diagonal(r, np.inf)
        pe = -self.G * np.sum(1.0 / r) / 2.0
        return ke + pe

    def integrate(self, pos: np.ndarray, vel: np.ndarray) -> float:
        """Return RMS energy error over all steps."""
        E0 = self._total_energy(pos, vel)
        errors = []
        for _ in range(self.n_steps):
            acc  = self._accelerations(pos)
            vel += 0.5 * self.dt * acc
            pos += self.dt * vel
            acc  = self._accelerations(pos)
            vel += 0.5 * self.dt * acc
            E    = self._total_energy(pos, vel)
            errors.append(abs((E - E0) / E0))
        return float(np.sqrt(np.mean(np.array(errors)**2)))

    def objective(self, params: np.ndarray) -> float:
        n = self.n_bodies * 3
        pos = params[:n].reshape(self.n_bodies, 3)
        vel = params[n:].reshape(self.n_bodies, 3)
        return self.integrate(pos.copy(), vel.copy())


# ---------------------------------------------------------------------------
# Dataclass for results
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
    metadata:         Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YO Hybrid Optimizer (continuous, minimisation)
# ---------------------------------------------------------------------------

class YOHybridOptimizer:
    """
    YO Hybrid for continuous parameter spaces (minimisation).
    """

    def __init__(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        n_chains: int = 3,
        burnin_ratio: float = 0.20,
        reheating_interval: int = 20,
        blacklist_tol: float = 0.10,
        initial_temp: float = 2.0,
        cooling_rate: float = 0.95,
        reheating_factor: float = 1.5,
        verbose: bool = True,
        save_dir: Optional[Path] = None,
        save_interval: int = 50,
    ):
        self.objective_func    = objective_func
        self.bounds            = np.array(bounds)
        self.dim               = len(bounds)
        self.n_chains          = n_chains
        self.burnin_ratio      = burnin_ratio
        self.reheating_interval= reheating_interval
        self.blacklist_tol     = blacklist_tol
        self.initial_temp      = initial_temp
        self.cooling_rate      = cooling_rate
        self.reheating_factor  = reheating_factor
        self.verbose           = verbose
        self.save_dir          = save_dir
        self.save_interval     = save_interval

        self.blacklist:    list = []
        self.history:      List[float] = []
        self.best_history: List[float] = []
        self.best_value:   float = float("inf")
        self.best_params:  Optional[np.ndarray] = None
        self.evals_used:   int = 0
        self.chains:       list = []

    def _blacklisted(self, x):
        return any(np.linalg.norm(x - b) < self.blacklist_tol for b in self.blacklist)

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _evaluate(self, x):
        v = self.objective_func(x)
        self.history.append(v)
        self.evals_used += 1
        if v < self.best_value:
            self.best_value  = v
            self.best_params = x.copy()
        self.best_history.append(self.best_value)
        return v

    def _mcmc_step(self, x, temperature, step_size):
        proposal = self._clip(x + np.random.normal(0, step_size, self.dim))
        if self._blacklisted(proposal):
            return x, self._evaluate(x)
        cv = self._evaluate(x)
        pv = self._evaluate(proposal)
        delta = pv - cv
        if delta < 0 or np.random.random() < np.exp(-delta / (temperature + 1e-10)):
            return proposal, pv
        return x, cv

    def _greedy_step(self, x, n_neighbors, step_size):
        bx, bv = x, self._evaluate(x)
        for _ in range(n_neighbors):
            nb = self._clip(bx + np.random.normal(0, step_size, self.dim))
            if not self._blacklisted(nb):
                v = self._evaluate(nb)
                if v < bv:
                    bx, bv = nb, v
        return bx, bv

    def _sa_step(self, x, temperature, step_size):
        proposal = self._clip(x + np.random.normal(0, step_size * temperature, self.dim))
        if self._blacklisted(proposal):
            return x, self._evaluate(x)
        cv = self._evaluate(x)
        pv = self._evaluate(proposal)
        delta = pv - cv
        if delta < 0 or np.random.random() < np.exp(-delta / (temperature + 1e-10)):
            return proposal, pv
        return x, cv

    def optimize(self, n_evaluations: int) -> "OptimizationResult":
        t0 = time.time()

        if self.verbose:
            print("=" * 70)
            print(f"YO HYBRID  dim={self.dim}  chains={self.n_chains}  budget={n_evaluations}")
            print("=" * 70)

        # Initialise chains
        for _ in range(self.n_chains):
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])
            self.chains.append(x0)
            self._evaluate(x0)

        # Phase 1: Burnin
        burnin = int(n_evaluations * self.burnin_ratio)
        T      = self.initial_temp
        burnin_samples = []
        for i in range(burnin // self.n_chains):
            for ci in range(self.n_chains):
                self.chains[ci], v = self._mcmc_step(self.chains[ci], T, 0.5)
                burnin_samples.append((self.chains[ci].copy(), v))
            T *= self.cooling_rate
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Burnin iter {i+1}  best={self.best_value:.4e}  T={T:.4f}")

        burnin_samples.sort(key=lambda s: s[1])
        self.chains = [s[0] for s in burnin_samples[: self.n_chains]]
        if self.verbose:
            print(f"  Burnin done. best={self.best_value:.4e}\n")

        # Phase 2: Hybrid
        T, it = self.initial_temp, 0
        while self.evals_used < n_evaluations:
            it += 1
            for ci in range(self.n_chains):
                if self.evals_used >= n_evaluations:
                    break
                self.chains[ci], _ = self._mcmc_step(self.chains[ci], T * 0.5, 0.3)

            if self.evals_used < n_evaluations and it % 5 == 0:
                chain_vals = [self.objective_func(c) for c in self.chains]
                best_ci    = int(np.argmin(chain_vals))
                neighbors  = min(5, (n_evaluations - self.evals_used) // 2)
                if neighbors > 0:
                    self.chains[best_ci], _ = self._greedy_step(self.chains[best_ci], neighbors, 0.15)

            if it % self.reheating_interval == 0:
                T = min(self.initial_temp, T * self.reheating_factor)
                if self.verbose:
                    print(f"  Reheating  iter={it}  T={T:.4f}")
            else:
                T *= self.cooling_rate

            for ci in range(self.n_chains):
                if self.evals_used >= n_evaluations:
                    break
                self.chains[ci], _ = self._sa_step(self.chains[ci], T, 0.3)

            if it % 15 == 0:
                vals    = [self.objective_func(c) for c in self.chains]
                worst   = int(np.argmax(vals))
                self.blacklist.append(self.chains[worst].copy())
                self.chains[worst] = np.array([
                    np.random.uniform(lo, hi) for lo, hi in self.bounds
                ])

            if self.save_dir and it % self.save_interval == 0:
                with open(self.save_dir / f"ckpt_{it}.pkl", "wb") as fh:
                    pickle.dump({"best_value": self.best_value, "best_params": self.best_params}, fh)

            if self.verbose and it % 10 == 0:
                print(f"  iter={it}  evals={self.evals_used}/{n_evaluations}  best={self.best_value:.4e}  T={T:.4f}")

        runtime = time.time() - t0
        if self.verbose:
            print(f"\nDone. best={self.best_value:.4e}  evals={self.evals_used}  time={runtime:.2f}s")

        return OptimizationResult(
            best_value=self.best_value,
            best_params=self.best_params,
            history=self.history,
            best_history=self.best_history,
            runtime=runtime,
            evaluation_count=self.evals_used,
            method_name="YO Hybrid",
            metadata={"n_chains": self.n_chains, "blacklist_size": len(self.blacklist)},
        )


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------

def random_search(objective_func, bounds, n_evaluations, verbose=True) -> OptimizationResult:
    bounds_arr = np.array(bounds)
    best_v, best_x = float("inf"), None
    history, best_history = [], []
    t0 = time.time()

    for i in range(n_evaluations):
        x = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        v = objective_func(x)
        history.append(v)
        if v < best_v:
            best_v, best_x = v, x.copy()
        best_history.append(best_v)
        if verbose and (i + 1) % 50 == 0:
            print(f"  eval {i+1}/{n_evaluations}  best={best_v:.4e}")

    runtime = time.time() - t0
    if verbose:
        print(f"  Done. best={best_v:.4e}  time={runtime:.2f}s")

    return OptimizationResult(
        best_value=best_v,
        best_params=best_x,
        history=history,
        best_history=best_history,
        runtime=runtime,
        evaluation_count=n_evaluations,
        method_name="Random Search",
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_convergence(rs: OptimizationResult, yo: OptimizationResult, save_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(rs.best_history, label="Random Search", linewidth=2, alpha=0.8)
    ax.plot(yo.best_history, label="YO Hybrid",    linewidth=2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Best-So-Far Convergence")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Energy Error (log)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    w  = 20
    ax.plot(pd.Series(rs.history).rolling(w, 1).mean(), label="Random Search", linewidth=2, alpha=0.8)
    ax.plot(pd.Series(yo.history).rolling(w, 1).mean(), label="YO Hybrid",    linewidth=2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Smoothed History (w={w})")
    ax.set_xlabel("Evaluation")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(rs.history, bins=50, alpha=0.6, label="Random Search", density=True)
    ax.hist(yo.history, bins=50, alpha=0.6, label="YO Hybrid",    density=True)
    ax.set_xlabel("Energy Error")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Sampled Values")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n   = min(len(rs.best_history), len(yo.best_history))
    imp = (np.array(rs.best_history[:n]) - np.array(yo.best_history[:n])) / (
        np.array(rs.best_history[:n]) + 1e-10
    ) * 100
    ax.plot(imp, linewidth=2, color="green")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.fill_between(range(n), 0, imp, where=imp > 0, alpha=0.3, color="green")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("YO Hybrid vs Random Search Improvement")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = "nbody_convergence.png"
    if save_dir:
        fname = str(save_dir / fname)
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def visualise_nbody(positions: np.ndarray, title="N-Body System", save_dir=None):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    c   = np.linalg.norm(positions, axis=1)
    sc  = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c=c, cmap="viridis", s=20, alpha=0.6, edgecolors="black", linewidth=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, pad=0.1, label="Distance from origin")
    plt.tight_layout()
    fname = "nbody_best_config.png"
    if save_dir:
        fname = str(save_dir / fname)
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_BODIES     = 100    # reduce to 10–20 for quick testing
    N_EVALS      = 200
    SEED         = 42

    print("=" * 70)
    print(f"N-BODY BENCHMARK  n_bodies={N_BODIES}  budget={N_EVALS}")
    print("=" * 70)

    sim    = NBodySimulation(n_bodies=N_BODIES, seed=SEED)
    bounds = [(-2.0, 2.0)] * (N_BODIES * 6)

    # Baseline
    print("\nRandom Search...")
    rs_result = random_search(sim.objective, bounds, N_EVALS, verbose=True)

    # YO Hybrid
    print("\nYO Hybrid...")
    opt = YOHybridOptimizer(
        objective_func=sim.objective,
        bounds=bounds,
        n_chains=3,
        burnin_ratio=0.2,
        reheating_interval=15,
        blacklist_tol=0.05,
        initial_temp=2.0,
        cooling_rate=0.95,
        reheating_factor=1.5,
        verbose=True,
    )
    yo_result = opt.optimize(N_EVALS)

    # Metrics
    imp = (rs_result.best_value - yo_result.best_value) / abs(rs_result.best_value) * 100
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Random Search: {rs_result.best_value:.6e}  ({rs_result.runtime:.2f}s)")
    print(f"YO Hybrid    : {yo_result.best_value:.6e}  ({yo_result.runtime:.2f}s)")
    if imp > 0:
        print(f"YO Hybrid improved by {imp:.2f}%")
    else:
        print(f"Baseline was better by {-imp:.2f}%")

    # Plots
    plot_convergence(rs_result, yo_result)
    best_pos = yo_result.best_params[: N_BODIES * 3].reshape(N_BODIES, 3)
    visualise_nbody(best_pos, title=f"Best Config (error={yo_result.best_value:.4e})")


if __name__ == "__main__":
    main()
