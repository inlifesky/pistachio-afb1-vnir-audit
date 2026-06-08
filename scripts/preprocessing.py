"""
Spectral preprocessing transforms (sklearn-compatible) for project_pistachio_AFB1.
Implemented directly (not chemotools) because each is textbook and the heavy primitive
(Savitzky-Golay) is scipy's. ALL are leakage-safe when used inside a Pipeline:
  - SNV / SG1 / SG2 are per-row (stateless) -> fit is a no-op.
  - MSC is stateful (reference = TRAIN mean spectrum) -> implemented as a fitted transformer.
"""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


def _snv(X):
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _sg(X, deriv, window=11, poly=2):
    X = np.asarray(X, float)
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv, axis=1)


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction. Reference = mean spectrum of training rows."""
    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        out = np.empty_like(X)
        ref = self.ref_
        for i in range(X.shape[0]):
            b, a = np.polyfit(ref, X[i], 1)      # X_i ≈ a + b*ref
            out[i] = (X[i] - a) / (b if b != 0 else 1.0)
        return out


def snv_tf():  return FunctionTransformer(_snv)
def sg1_tf():  return FunctionTransformer(lambda X: _sg(X, 1))
def sg2_tf():  return FunctionTransformer(lambda X: _sg(X, 2))
def raw_tf():  return FunctionTransformer(lambda X: np.asarray(X, float))


def make_preprocessor(name):
    """Return a list of (step_name, transformer) for the named preprocessing."""
    if name == "raw":      return [("raw", raw_tf())]
    if name == "snv":      return [("snv", snv_tf())]
    if name == "msc":      return [("msc", MSC())]
    if name == "sg1":      return [("sg1", sg1_tf())]
    if name == "sg2":      return [("sg2", sg2_tf())]
    if name == "snv_sg1":  return [("snv", snv_tf()), ("sg1", sg1_tf())]
    raise ValueError(name)


PREPROCESSINGS = ["raw", "snv", "msc", "sg1", "sg2", "snv_sg1"]
