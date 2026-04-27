"""
benchmarks/portfolio/run.py
===========================
Portfolio optimisation using YO Hybrid: maximise Sharpe ratio by finding
optimal asset weights. Fetches real stock data via yfinance.

Usage
-----
    pip install yfinance
    python benchmarks/portfolio/run.py
"""

import time
import warnings
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "yfinance", "-q"])
    import yfinance as yf

from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_stock_data(tickers: List[str], years: int = 2) -> pd.DataFrame:
    end   = datetime.now()
    start = end - timedelta(days=years * 365)
    print(f"Fetching {len(tickers)} tickers from {start.date()} to {end.date()}...")

    raw = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw.get("Adj Close", raw.get("Close"))
    else:
        col  = "Adj Close" if "Adj Close" in raw.columns else "Close"
        data = raw[[col]]
        data.columns = [tickers[0]]

    missing = data.isnull().sum() / len(data)
    valid   = missing[missing < 0.1].index.tolist()
    data    = data[valid].dropna()
    print(f"Valid tickers: {len(valid)}  days: {len(data)}")
    return data


def compute_metrics(prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    rets = prices.pct_change().dropna()
    mu   = rets.mean().values * 252
    cov  = rets.cov().values * 252
    return mu, cov, 0.04


# ---------------------------------------------------------------------------
# Portfolio objective
# ---------------------------------------------------------------------------

class PortfolioObjective:
    """Negative Sharpe ratio (minimise → maximise Sharpe)."""

    def __init__(self, mu, cov, rf, max_weight=0.20):
        self.mu, self.cov, self.rf = mu, cov, rf
        self.max_weight = max_weight
        self.n = len(mu)

    def __call__(self, weights: np.ndarray) -> float:
        w = np.clip(weights, 0, self.max_weight)
        w = w / w.sum()
        ret = np.dot(w, self.mu)
        vol = np.sqrt(np.dot(w, self.cov @ w))
        if vol < 1e-10:
            return 1e10
        return -(ret - self.rf) / vol

    def stats(self, weights: np.ndarray) -> dict:
        w   = np.clip(weights, 0, self.max_weight)
        w   = w / w.sum()
        ret = np.dot(w, self.mu)
        vol = np.sqrt(np.dot(w, self.cov @ w))
        return {"weights": w, "return": ret, "volatility": vol,
                "sharpe_ratio": (ret - self.rf) / vol}


# ---------------------------------------------------------------------------
# YO Hybrid (portfolio version, minimisation)
# ---------------------------------------------------------------------------

class YOHybridOptimizer:
    def __init__(
        self,
        objective_fn: Callable,
        n_assets: int,
        max_weight: float = 0.20,
        n_chains: int = 4,
        burnin: int = 100,
        n_iterations: int = 1000,
        sa_temp: float = 1.0,
        cooling: float = 0.95,
        reheat_threshold: int = 50,
        greedy_steps: int = 20,
    ):
        self.obj        = objective_fn
        self.n          = n_assets
        self.max_w      = max_weight
        self.n_chains   = n_chains
        self.burnin     = burnin
        self.n_iter     = n_iterations
        self.sa_temp    = sa_temp
        self.cooling    = cooling
        self.reheat_thr = reheat_threshold
        self.greedy_stp = greedy_steps
        self.blacklist: set = set()
        self.best:      float = np.inf
        self.best_w:    np.ndarray = None
        self.history:   list = []

    def _rand_w(self):
        w = np.random.dirichlet(np.ones(self.n))
        w = np.clip(w, 0, self.max_w)
        return w / w.sum()

    def _blacklisted(self, w, tol=0.05):
        return tuple(np.round(w, 3)) in self.blacklist

    def _add_bl(self, w):
        self.blacklist.add(tuple(np.round(w, 3)))

    def _mcmc(self, w, step=0.05):
        p = w + np.random.normal(0, step, self.n)
        p = np.clip(p, 0, self.max_w)
        return p / p.sum()

    def _greedy(self, w):
        bw, bs = w.copy(), self.obj(w)
        for _ in range(self.greedy_stp):
            for i in range(self.n):
                for d in [-0.02, -0.01, 0.01, 0.02]:
                    tw = bw.copy(); tw[i] += d
                    tw = np.clip(tw, 0, self.max_w); tw /= tw.sum()
                    if not self._blacklisted(tw):
                        s = self.obj(tw)
                        if s < bs:
                            bw, bs = tw, s
        return bw

    def _sa(self, w0):
        w, s = w0.copy(), self.obj(w0)
        bw, bs = w.copy(), s
        T, no_imp = self.sa_temp, 0
        for _ in range(self.n_iter):
            p = self._mcmc(w, step=0.03)
            if self._blacklisted(p):
                continue
            ps = self.obj(p)
            delta = ps - s
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                w, s = p, ps
                no_imp = 0
                if s < bs:
                    bw, bs = w.copy(), s
            else:
                no_imp += 1
            if no_imp >= self.reheat_thr:
                T = self.sa_temp * 0.5; no_imp = 0
            T *= self.cooling
        return bw, bs

    def _run_chain(self, cid):
        w   = self._rand_w()
        s   = self.obj(w)
        bw, bs = w.copy(), s
        samples = []
        for _ in range(self.burnin + self.n_iter):
            p = self._mcmc(w)
            if self._blacklisted(p):
                continue
            ps = self.obj(p)
            if ps < s or np.random.random() < np.exp(-(ps - s)):
                w, s = p, ps
            samples.append((w.copy(), s))
            if s < bs:
                bw, bs = w.copy(), s
        samples.sort(key=lambda x: x[1])
        print(f"  Chain {cid+1}: MCMC best={bs:.6f}")
        for wi, _ in samples[:3]:
            wi   = self._greedy(wi)
            sw, ss = self._sa(wi)
            if ss < bs:
                bw, bs = sw, ss
                print(f"  Chain {cid+1}: SA improved to {ss:.6f}")
        poor = samples[int(len(samples) * 0.5):]
        for wj, _ in poor[:50]:
            self._add_bl(wj)
        return bw, bs

    def optimize(self) -> dict:
        print(f"\n{'='*70}")
        print(f"YO HYBRID PORTFOLIO — {self.n_chains} chains × {self.n_iter} iters")
        print(f"{'='*70}\n")
        results = []
        for cid in range(self.n_chains):
            w, s = self._run_chain(cid)
            results.append((w, s))
            self.history.append(s)
            if s < self.best:
                self.best, self.best_w = s, w
        results.sort(key=lambda x: x[1])
        print(f"\nBest (neg Sharpe): {self.best:.6f}  → Sharpe: {-self.best:.6f}")
        return {"best_weights": self.best_w, "best_score": self.best,
                "all_chains": results, "history": self.history}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(weights, tickers, stats, history):
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    nz = weights > 0.001

    ax = fig.add_subplot(gs[0, 0])
    ax.pie(weights[nz], labels=np.array(tickers)[nz], autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 9})
    ax.set_title("Portfolio Allocation", fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    si = np.argsort(weights)[::-1]
    ax.barh(np.array(tickers)[si], weights[si], color=plt.cm.viridis(np.linspace(0, 1, len(tickers)))[si])
    ax.set_xlabel("Weight"); ax.set_title("Asset Weights (sorted)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(history, linewidth=2, color="steelblue")
    ax.set_xlabel("Chain"); ax.set_ylabel("Neg Sharpe")
    ax.set_title("Optimisation Convergence", fontsize=14, fontweight="bold"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    txt = (
        f"Return: {stats['return']*100:.2f}%   "
        f"Volatility: {stats['volatility']*100:.2f}%   "
        f"Sharpe: {stats['sharpe_ratio']:.4f}   "
        f"Assets held: {np.sum(weights > 0.001)}/{len(weights)}"
    )
    ax.text(0.1, 0.5, txt, fontsize=12, family="monospace", va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.suptitle("YO Hybrid — Portfolio Optimisation Results", fontsize=16, fontweight="bold")
    plt.savefig("portfolio_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: portfolio_results.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "JNJ", "UNH", "XOM", "PG", "HD",
]


def main():
    prices = fetch_stock_data(TICKERS, years=2)
    mu, cov, rf = compute_metrics(prices)
    valid_tickers = list(prices.columns)

    obj = PortfolioObjective(mu, cov, rf, max_weight=0.20)

    opt = YOHybridOptimizer(
        objective_fn=obj,
        n_assets=len(valid_tickers),
        max_weight=0.20,
        n_chains=4,
        burnin=100,
        n_iterations=1000,
        sa_temp=1.0,
        cooling=0.95,
        reheat_threshold=50,
        greedy_steps=20,
    )
    result = opt.optimize()

    best_w = result["best_weights"]
    stats  = obj.stats(best_w)

    print("\n" + "=" * 70)
    print("OPTIMISED PORTFOLIO")
    print("=" * 70)
    for tk, w in sorted(zip(valid_tickers, best_w), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {tk:10s} {w*100:6.2f}%")
    print(f"\nExpected Return : {stats['return']*100:.2f}%")
    print(f"Volatility      : {stats['volatility']*100:.2f}%")
    print(f"Sharpe Ratio    : {stats['sharpe_ratio']:.4f}")

    plot_results(best_w, valid_tickers, stats, result["history"])


if __name__ == "__main__":
    main()
