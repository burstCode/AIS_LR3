import numpy as np
import pytest

from fuzzy_lr3.aggregation import AGGREGATIONS, DEFAULT_AGGREGATION, aggregate
from fuzzy_lr3.implications import IMPLICATIONS


def test_max_aggregation():
    sets = [np.array([0.2, 0.9, 0.1]), np.array([0.5, 0.3, 0.8])]
    result = aggregate("max", sets)
    np.testing.assert_allclose(result, [0.5, 0.9, 0.8])


def test_min_aggregation():
    sets = [np.array([0.2, 0.9, 0.1]), np.array([0.5, 0.3, 0.8])]
    result = aggregate("min", sets)
    np.testing.assert_allclose(result, [0.2, 0.3, 0.1])


def test_every_implication_has_default_aggregation():
    assert DEFAULT_AGGREGATION.keys() == IMPLICATIONS.keys()
    assert set(DEFAULT_AGGREGATION.values()) <= AGGREGATIONS.keys()


def test_probabilistic_sum_two_sets():
    sets = [np.array([0.5]), np.array([0.5])]
    result = aggregate("probabilistic_sum", sets)
    # 0.5 + 0.5 - 0.5*0.5 = 0.75
    np.testing.assert_allclose(result, [0.75])


def test_bounded_sum_saturates_at_one():
    sets = [np.array([0.7]), np.array([0.7])]
    result = aggregate("bounded_sum", sets)
    np.testing.assert_allclose(result, [1.0])


def test_aggregate_rejects_empty_list():
    with pytest.raises(ValueError):
        aggregate("max", [])


def test_aggregate_single_set_is_identity():
    s = np.array([0.1, 0.4, 0.9])
    for op in ("max", "min", "probabilistic_sum", "bounded_sum"):
        np.testing.assert_allclose(aggregate(op, [s]), s)
