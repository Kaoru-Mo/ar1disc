"""Tauchen (1986) 法による AR(1) 過程の離散化。

対象とする過程:
    y_t = mu + rho * y_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)

mu は定数項（QuantEcon の tauchen と同じ扱い）。過程の平均は mu / (1 - rho)。
"""

import numpy as np
from scipy.stats import norm

from .chain import DiscreteMarkovChain


def tauchen(n: int, rho: float, sigma: float, mu: float = 0.0, n_std: float = 3.0) -> DiscreteMarkovChain:
    """AR(1) 過程を n 状態のマルコフ連鎖で近似する。

    Parameters
    ----------
    n : int
        格子点の数（2 以上）。
    rho : float
        自己相関係数（|rho| < 1）。
    sigma : float
        ショック eps_t の標準偏差（> 0）。
    mu : float, default 0.0
        定数項。
    n_std : float, default 3.0
        格子の端を、無条件標準偏差の何倍に置くか。

    Returns
    -------
    DiscreteMarkovChain
    """
    if n < 2:
        raise ValueError("n は 2 以上にする")
    if not abs(rho) < 1:
        raise ValueError("|rho| < 1 でないと定常な過程にならない")
    if sigma <= 0:
        raise ValueError("sigma は正にする")

    # 平均を除いた過程 x_t = y_t - mu/(1-rho) の上で格子を作る
    std_y = sigma / np.sqrt(1 - rho**2)      # 無条件標準偏差
    x = np.linspace(-n_std * std_y, n_std * std_y, n)
    half_step = (x[1] - x[0]) / 2

    # z[i, j] = x[j] - rho * x[i] : 状態 i から見た x[j] までの距離
    z = x[None, :] - rho * x[:, None]
    upper = norm.cdf((z + half_step) / sigma)
    lower = norm.cdf((z - half_step) / sigma)

    P = upper - lower
    P[:, 0] = upper[:, 0]                    # 左端は (-inf, x_0 + h/2] をすべて受け持つ
    P[:, -1] = 1 - lower[:, -1]              # 右端は [x_{n-1} - h/2, inf) をすべて受け持つ

    return DiscreteMarkovChain(grid=x + mu / (1 - rho), P=P)
