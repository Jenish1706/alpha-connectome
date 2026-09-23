import numpy as np
import pyarrow.parquet as pq
import pytest

from src.data.schema import SEQUENCE_LENGTH, WARMUP, iter_sequences
from src.utils.metric import WPAccumulator, global_weighted_pearson, global_wp, weighted_pearson


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


def test_float32_inputs_and_thresholds_follow_the_scorer():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(1000)
    # 1e-9 jitter vanishes in the float32 cast: the prediction is constant, so 0.
    assert weighted_pearson(y, 0.25 + 1e-9 * rng.standard_normal(1000)) == 0.0
    assert weighted_pearson(y, np.full(1000, 0.25)) == 0.0
    assert -1.0 <= weighted_pearson(y, -y) <= 1.0
    with pytest.raises(ValueError, match="nonfinite"):
        weighted_pearson([1.0, np.nan], [1.0, 2.0])


def _random_rows(rng, n):
    y = rng.standard_normal((n, 2)).astype(np.float32) * 1.5  # tails exercise clipping
    p = (0.4 * y + rng.standard_normal((n, 2))).astype(np.float32)
    steps = np.arange(n) % SEQUENCE_LENGTH
    need = steps >= WARMUP
    scored = need & (rng.random(n) < 0.3)
    return y, p, need, scored


def test_global_wp_uses_need_and_scored_mask():
    y, p, need, scored = _random_rows(np.random.default_rng(1), 3 * SEQUENCE_LENGTH)
    expected = global_weighted_pearson(y, p, mask=need & scored)
    assert global_wp(y, p, need, scored) == expected
    warm_nan = p.copy()
    warm_nan[~need] = np.nan  # predictions are only required after warm-up
    assert global_wp(y, warm_nan, need, scored) == expected
    assert global_wp(y, p, need) == global_weighted_pearson(y, p, mask=need)
    with pytest.raises(ValueError, match="no rows selected"):
        global_wp(y, p, need, np.zeros_like(need))
    bad = p.copy()
    bad[WARMUP] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        global_wp(y, bad, need, scored)


@pytest.mark.parametrize("chunk", [1_000, 777, SEQUENCE_LENGTH, 3 * SEQUENCE_LENGTH])
def test_accumulator_is_chunking_invariant(chunk):
    y, p, need, scored = _random_rows(np.random.default_rng(2), 3 * SEQUENCE_LENGTH)
    acc = WPAccumulator()
    for s in range(0, len(y), chunk):
        acc.update(y[s:s + chunk], p[s:s + chunk], need[s:s + chunk], scored[s:s + chunk])
    result = acc.result()
    assert result["weighted_pearson"] == pytest.approx(global_wp(y, p, need, scored), abs=1e-12)
    assert result["selected_rows"] == int((need & scored).sum())


def test_accumulators_merge_like_one_pass():
    y, p, need, scored = _random_rows(np.random.default_rng(3), 2 * SEQUENCE_LENGTH)
    half = SEQUENCE_LENGTH
    left = WPAccumulator().update(y[:half], p[:half], need[:half], scored[:half])
    right = WPAccumulator().update(y[half:], p[half:], need[half:], scored[half:])
    merged = left.merge(right).result()
    assert merged["weighted_pearson"] == pytest.approx(global_wp(y, p, need, scored), abs=1e-12)
    with pytest.raises(ValueError, match="no rows selected"):
        WPAccumulator().result()


def test_matches_starterpack_scorer(official):
    rng = np.random.default_rng(4)
    y, p, need, scored = _random_rows(rng, 3 * SEQUENCE_LENGTH)
    for k in range(2):
        assert weighted_pearson(y[:, k], p[:, k]) == official.weighted_pearson(y[:, k], p[:, k])
    reference = official.GlobalAccumulator()
    ours = WPAccumulator()
    for s in range(0, len(y), SEQUENCE_LENGTH):
        rows = slice(s, s + SEQUENCE_LENGTH)
        reference.add(y[rows], p[rows], need[rows] & scored[rows])
        ours.update(y[rows], p[rows], need[rows], scored[rows])
    expected, got = reference.result(), ours.result()
    for key in ("t0", "t1", "weighted_pearson", "selected_rows"):
        assert got[key] == pytest.approx(expected[key], abs=1e-12)


def test_real_validation_rows_match_starterpack_scorer(valid_path, official, check_groups):
    groups = check_groups(pq.ParquetFile(valid_path).metadata.num_row_groups)
    weights = np.random.default_rng(5).standard_normal((112, 2)).astype(np.float32)
    streamed, official_acc = WPAccumulator(), official.GlobalAccumulator()
    rows = {"y": [], "p": [], "need": [], "scored": []}
    for s in iter_sequences(valid_path, groups):  # one sequence in memory at a time
        prediction = (s.features @ weights / 10).astype(np.float32)
        streamed.update(s.targets, prediction, s.need_prediction, s.is_scored)
        official_acc.add(s.targets, prediction, s.is_scored & s.need_prediction)
        for key, value in zip(rows, (s.targets, prediction, s.need_prediction, s.is_scored), strict=True):
            rows[key].append(value)
    y, p, need, scored = (np.concatenate(v) for v in rows.values())

    mask = need & scored
    reference = np.mean([official.weighted_pearson(y[mask, k], p[mask, k]) for k in range(2)])
    assert global_wp(y, p, need, scored) == reference
    assert global_wp(y, y, need, scored) == 1.0
    assert streamed.result()["weighted_pearson"] == pytest.approx(reference, abs=1e-12)
    assert official_acc.result()["weighted_pearson"] == pytest.approx(reference, abs=1e-12)
