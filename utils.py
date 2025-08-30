# utils.py
import numpy as np

def make_random_sparse_a(m, s, rng=None):
    if rng is None: rng = np.random.RandomState()
    a = np.zeros(m)
    idx = rng.choice(m, s, replace=False)
    a[idx] = rng.randn(s)
    return a
