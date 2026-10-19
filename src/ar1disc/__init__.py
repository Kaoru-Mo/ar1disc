"""AR(1) 過程を有限状態マルコフ連鎖に離散化する小さなライブラリ。"""

from .chain import DiscreteMarkovChain
from .tauchen import tauchen

__all__ = ["DiscreteMarkovChain", "tauchen"]
__version__ = "0.1.0"
