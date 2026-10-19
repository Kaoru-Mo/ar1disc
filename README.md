# ar1disc

AR(1) 過程

$$y_t = \mu + \rho\, y_{t-1} + \varepsilon_t,\qquad \varepsilon_t \sim N(0,\sigma^2)$$

を、Tauchen (1986) の方法で有限状態マルコフ連鎖に離散化する。

`mu` は定数項で、過程の平均は `mu / (1 - rho)`。この扱いと引数の順番は QuantEcon の `qe.markov.tauchen(n, rho, sigma, mu=0.0, n_std=3)` に合わせてある。

## インストール

```bash
pip install git+https://github.com/Kaoru-Mo/ar1disc.git
```

Colab ではセルの先頭に `!` を付ける。

## 使い方

```python
from ar1disc import tauchen

mc = tauchen(n=21, rho=0.945, sigma=0.025)

mc.grid           # 格子点 (21,)
mc.P              # 遷移確率行列 (21, 21)
mc.stationary()   # 定常分布
mc.simulate(T=1000, seed=0)   # シミュレーション系列
```

所得過程として使うときは `np.exp(mc.grid)` で水準に戻す。

## テスト

```bash
pip install -e ".[test]"
pytest
```

QuantEcon の `tauchen` と格子・遷移行列が一致するかも確かめている（QuantEcon が入っていなければそのテストは skip される）。

## 参考文献

- Tauchen, G. (1986). Finite state Markov-chain approximations to univariate and vector autoregressions. *Economics Letters*, 20(2), 177–181.
