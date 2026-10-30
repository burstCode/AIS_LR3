import numpy as np
import pytest

from fuzzy_lr3.defuzzify import centroid


def test_centroid_of_symmetric_triangle_is_its_peak():
    universe = np.linspace(0, 10, 2001)
    membership = np.clip(1 - np.abs(universe - 5) / 5, 0, 1)

    assert centroid(universe, membership) == pytest.approx(5.0, abs=1e-3)


def test_centroid_of_rectangle_is_its_midpoint():
    universe = np.linspace(0, 10, 2001)
    membership = np.where((universe >= 2) & (universe <= 8), 1.0, 0.0)

    assert centroid(universe, membership) == pytest.approx(5.0, abs=1e-2)


def test_centroid_of_skewed_set_is_not_the_midpoint():
    universe = np.linspace(0, 10, 2001)
    # Правый треугольник: пик принадлежности у правого края.
    membership = np.clip((universe - 0) / 10, 0, 1)

    result = centroid(universe, membership)
    assert result > 5.0  # масса сосредоточена правее середины


def test_centroid_of_empty_set_falls_back_to_universe_midpoint():
    universe = np.linspace(0, 10, 101)
    membership = np.zeros_like(universe)

    assert centroid(universe, membership) == pytest.approx(5.0)
