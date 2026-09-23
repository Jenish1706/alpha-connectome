import numpy as np

from src.utils.metric import global_weighted_pearson, weighted_pearson


def test_perfect_and_clipping():
    y = np.array([-4.0, -1.0, 0.5, 3.0])
    assert weighted_pearson(y, y) == 1.0
    assert weighted_pearson(y, np.clip(y, -2, 2)) == 1.0


def test_degenerate_cases_return_zero():
    assert weighted_pearson([0, 0], [1, 2]) == 0.0
    assert weighted_pearson([1, 1], [2, 3]) == 0.0
    assert weighted_pearson([], []) == 0.0


def test_mask_and_two_targets_are_pooled():
    y = np.array([[1., 10.], [2., 20.], [-1., -10.]])
    p = y.copy()
    assert global_weighted_pearson(y, p, mask=np.array([True, False, True])) == 1.0
