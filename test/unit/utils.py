import numpy as np

from src.utils import maxent_cset


def test_maxent_cset():
    vec_min = np.array([0.3, 0.4, 0, 0.1])
    vec_max = np.array([0.6, 0.8, 0.12, 0.17])

    out = maxent_cset(vec_min, vec_max)

    assert np.allclose(out, np.array([0.31, 0.4, 0.12, 0.17]))
