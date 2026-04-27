"""
yo_hybrid/optimizer.py
======================
Generic YO Hybrid Optimizer usable across all benchmarks.

Usage
-----
    from yo_hybrid import YOHybridOptimizer

    param_space = {
        'n_estimators': (50, 200, 'int'),
        'max_depth':    (5,  30,  'int'),
        'max_features': (0.3, 1.0, 'float'),
    }

    def objective(params: dict) -> float:
        # return a score to MAXIMISE
        ...

    opt = YOHybridOptimizer(param_space, n_iterations=100, n_chains=3, seed=42)
    best_params, best_score = opt.optimize_multichain(objective, verbose=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Stores the outcome of a single optimization run."""

    best_value: float
    best_params: dict
    history: List[float] = field(default_factory=list)
    best_history: List[float] = field(default_factory=list)
    runtime: float = 0.0
    evaluation_count: int = 0
    method_name: str = "YO Hybrid"
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class YOHybridOptimizer:
    """
    YO Hybrid Optimizer: Three-layer optimization with advanced features.

    Layers
    ------
    1. MCMC Proposal   — Gaussian perturbation, adaptive step size.
    2. Greedy Search   — Axis-aligned neighbourhood improvement (post-burnin).
    3. SA Controller   — Metropolis acceptance + adaptive reheating.

    Additional features
    -------------------
    - Blacklisting     : avoids revisiting evaluated configurations.
    - Post-burnin      : phase transition from exploration to exploitation.
    - Multi-chain      : runs N independent chains, returns global best.

    Parameters
    ----------
    param_space : dict
        ``{name: (min, max, type)}``  where *type* is ``'int'`` or ``'float'``.
    n_iterations : int
        Optimisation iterations **per chain**.
    n_chains : int
        Number of independent chains.
    initial_temp : float
        Starting temperature for the SA controller.
    cooling_rate : float
        Multiplicative temperature decay per iteration.
    burnin_ratio : float
        Fraction of iterations used as burnin (exploration) phase.
    stagnation_threshold : int
        Non-improving iterations before reheating triggers.
    reheating_factor : float
        Temperature multiplier on reheating.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        param_space: Dict[str, Tuple],
        n_iterations: int = 100,
        n_chains: int = 3,
        initial_temp: float = 5.0,
        cooling_rate: float = 0.95,
        burnin_ratio: float = 0.25,
        stagnation_threshold: int = 10,
        reheating_factor: float = 2.5,
        seed: Optional[int] = None,
    ):
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.n_chains = n_chains
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.burnin_ratio = burnin_ratio
        self.stagnation_threshold = stagnation_threshold
        self.reheating_factor = reheating_factor
        self.burnin_iterations = int(n_iterations * burnin_ratio)
        self.rng = np.random.RandomState(seed)

        # Populated by optimize_multichain
        self.best_params_all_chains: List[dict] = []
        self.best_scores_all_chains: List[float] = []
        self.optimization_history: List[List[dict]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_params(self) -> dict:
        params: dict = {}
        for name, (low, high, ptype) in self.param_space.items():
            if ptype == "int":
                params[name] = int(self.rng.randint(low, high + 1))
            else:
                params[name] = float(self.rng.uniform(low, high))
        return params

    def _propose_mcmc(self, current: dict, is_burnin: bool) -> dict:
        """Layer 1 — MCMC proposal with adaptive step size."""
        step_scale = 0.3 if is_burnin else 0.1
        proposed = current.copy()
        for name, (low, high, ptype) in self.param_space.items():
            sigma = (high - low) * step_scale
            proposed[name] = float(
                np.clip(current[name] + self.rng.normal(0, sigma), low, high)
            )
            if ptype == "int":
                proposed[name] = int(round(proposed[name]))
        return proposed

    def _greedy_local_search(
        self,
        params: dict,
        objective_func: Callable[[dict], float],
        current_score: float,
    ) -> Tuple[dict, float]:
        """Layer 2 — Axis-aligned greedy neighbourhood search."""
        best_params = params.copy()
        best_score = current_score
        for name, (low, high, ptype) in self.param_space.items():
            step = (high - low) * 0.05
            for direction in [step, -step]:
                candidate = params.copy()
                candidate[name] = float(
                    np.clip(params[name] + direction, low, high)
                )
                if ptype == "int":
                    candidate[name] = int(round(candidate[name]))
                score = objective_func(candidate)
                if score > best_score:
                    best_score = score
                    best_params = candidate.copy()
        return best_params, best_score

    def _acceptance_probability(
        self, current_score: float, proposed_score: float, temperature: float
    ) -> float:
        """Layer 3 — Metropolis criterion (maximisation convention)."""
        if proposed_score > current_score:
            return 1.0
        delta = proposed_score - current_score
        return math.exp(delta / (temperature + 1e-10))

    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        return tuple(sorted(params.items()))

    # ------------------------------------------------------------------
    # Single-chain run
    # ------------------------------------------------------------------

    def optimize_single_chain(
        self,
        objective_func: Callable[[dict], float],
        chain_id: int = 0,
        verbose: bool = False,
    ) -> Tuple[dict, float, List[dict]]:
        """
        Run a single chain of the YO optimizer.

        Returns
        -------
        best_params, best_score, history
        """
        current_params = self._initialize_params()
        current_score = objective_func(current_params)

        best_params = current_params.copy()
        best_score = current_score

        temperature = self.initial_temp
        blacklist: set = {self._params_to_tuple(current_params)}
        stagnation_counter = 0
        reheating_count = 0
        history: List[dict] = []

        if verbose:
            print(f"\nChain {chain_id}: start score = {current_score:.4f}")

        for iteration in range(self.n_iterations):
            is_burnin = iteration < self.burnin_iterations

            # ---- Layer 1: MCMC proposal ----
            proposed = self._propose_mcmc(current_params, is_burnin)
            p_tuple = self._params_to_tuple(proposed)
            for _ in range(5):
                if p_tuple not in blacklist:
                    break
                proposed = self._propose_mcmc(current_params, is_burnin)
                p_tuple = self._params_to_tuple(proposed)

            proposed_score = objective_func(proposed)

            # ---- Layer 2: Greedy local search (post-burnin) ----
            if not is_burnin:
                proposed, proposed_score = self._greedy_local_search(
                    proposed, objective_func, proposed_score
                )

            # ---- Layer 3: SA acceptance ----
            accept_prob = self._acceptance_probability(
                current_score, proposed_score, temperature
            )
            if self.rng.random() < accept_prob:
                current_params = proposed
                current_score = proposed_score
                blacklist.add(self._params_to_tuple(current_params))
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            # Reheating
            if stagnation_counter >= self.stagnation_threshold:
                temperature *= self.reheating_factor
                stagnation_counter = 0
                reheating_count += 1

            # Cooling (post-burnin only)
            if not is_burnin:
                temperature *= self.cooling_rate
            temperature = max(temperature, 0.01)

            history.append(
                {
                    "iteration": iteration,
                    "current_score": current_score,
                    "best_score": best_score,
                    "temperature": temperature,
                    "is_burnin": is_burnin,
                }
            )

            if verbose and (iteration + 1) % 25 == 0:
                phase = "Burnin" if is_burnin else "Exploit"
                print(
                    f"  [{phase}] iter {iteration+1}/{self.n_iterations} "
                    f"curr={current_score:.4f} best={best_score:.4f} T={temperature:.3f}"
                )

        if verbose:
            print(
                f"Chain {chain_id}: best={best_score:.4f}  reheats={reheating_count}"
            )

        return best_params, best_score, history

    # ------------------------------------------------------------------
    # Multi-chain run
    # ------------------------------------------------------------------

    def optimize_multichain(
        self,
        objective_func: Callable[[dict], float],
        verbose: bool = False,
    ) -> Tuple[dict, float]:
        """
        Run all chains and return the global best result.

        Returns
        -------
        best_params : dict
        best_score  : float
        """
        if verbose:
            print("=" * 60)
            print(
                f"YO HYBRID: {self.n_chains} chains × {self.n_iterations} iterations"
            )
            print("=" * 60)

        all_results = []
        for cid in range(self.n_chains):
            params, score, history = self.optimize_single_chain(
                objective_func, chain_id=cid, verbose=verbose
            )
            all_results.append((params, score, history))
            self.best_params_all_chains.append(params)
            self.best_scores_all_chains.append(score)
            self.optimization_history.append(history)

        best_idx = int(np.argmax([s for _, s, _ in all_results]))
        best_params, best_score, _ = all_results[best_idx]

        if verbose:
            print(f"\nBest chain: {best_idx}  score: {best_score:.4f}")

        return best_params, best_score
