"""離散化の結果を保持するクラス。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscreteMarkovChain:
    """有限状態マルコフ連鎖。

    Attributes
    ----------
    grid : np.ndarray, shape (n,)
        状態の値（格子点）。
    P : np.ndarray, shape (n, n)
        遷移確率行列。P[i, j] は grid[i] から grid[j] へ移る確率。
    """

    grid: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        self.grid = np.asarray(self.grid, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        n = self.grid.shape[0]
        if self.P.shape != (n, n):
            raise ValueError(f"P の形が {self.P.shape} だが、({n}, {n}) が必要")
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("P の各行の和が 1 になっていない")

    @property
    def n(self) -> int:
        """状態数。"""
        return self.grid.shape[0]

    def stationary(self) -> np.ndarray:
        """定常分布 pi（pi @ P = pi, sum(pi) = 1）を返す。

        (P' - I) pi = 0 の1本を sum(pi) = 1 に置き換えた連立方程式を解く。
        """
        A = self.P.T - np.eye(self.n)
        A[-1, :] = 1.0
        b = np.zeros(self.n)
        b[-1] = 1.0
        return np.linalg.solve(A, b)

    def simulate(self, T: int, init: int | None = None, seed: int | None = None) -> np.ndarray:
        """長さ T の状態の値の系列を返す。

        Parameters
        ----------
        T : int
            系列の長さ。
        init : int, optional
            初期状態のインデックス。省略時は定常分布から引く。
        seed : int, optional
            乱数シード。同じ値なら同じ系列になる。
        """
        rng = np.random.default_rng(seed)
        cdf = np.cumsum(self.P, axis=1)
        idx = np.empty(T, dtype=int)
        if init is None:
            idx[0] = rng.choice(self.n, p=self.stationary())
        else:
            idx[0] = init
        u = rng.random(T)
        for t in range(1, T):
            idx[t] = min(np.searchsorted(cdf[idx[t - 1]], u[t]), self.n - 1)
        return self.grid[idx]
