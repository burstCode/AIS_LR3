import numpy as np
import pytest

from fuzzy_lr3.membership import Trapezoidal, make_membership_function


def test_triangular_shape():
    mf = Trapezoidal(0, 5, 5, 10)

    assert mf(0) == pytest.approx(0.0)
    assert mf(5) == pytest.approx(1.0)
    assert mf(10) == pytest.approx(0.0)
    assert mf(2.5) == pytest.approx(0.5)
    assert mf(7.5) == pytest.approx(0.5)


def test_trapezoidal_plateau():
    mf = Trapezoidal(0, 2, 6, 8)

    assert mf(1) == pytest.approx(0.5)
    assert mf(2) == pytest.approx(1.0)
    assert mf(4) == pytest.approx(1.0)
    assert mf(6) == pytest.approx(1.0)
    assert mf(7) == pytest.approx(0.5)
    assert mf(8) == pytest.approx(0.0)


def test_left_shoulder_no_division_by_zero():
    mf = Trapezoidal(0, 0, 0, 10)

    assert mf(0) == pytest.approx(1.0)
    assert mf(5) == pytest.approx(0.5)
    assert mf(10) == pytest.approx(0.0)


def test_right_shoulder_no_division_by_zero():
    mf = Trapezoidal(0, 10, 10, 10)

    assert mf(0) == pytest.approx(0.0)
    assert mf(5) == pytest.approx(0.5)
    assert mf(10) == pytest.approx(1.0)


def test_rejects_unordered_points():
    with pytest.raises(ValueError):
        Trapezoidal(5, 0, 1, 2)


def test_vectorized_call_returns_array():
    mf = Trapezoidal(0, 5, 5, 10)
    result = mf(np.array([0, 5, 10]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0])


def test_scalar_call_returns_python_float():
    mf = Trapezoidal(0, 5, 5, 10)
    assert isinstance(mf(5), float)


def test_make_membership_function_triangular():
    mf = make_membership_function({"type": "triangular", "points": [0, 5, 10]})
    assert mf(5) == pytest.approx(1.0)


def test_make_membership_function_trapezoidal():
    mf = make_membership_function({"type": "trapezoidal", "points": [0, 2, 6, 8]})
    assert mf(4) == pytest.approx(1.0)


def test_make_membership_function_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_membership_function({"type": "gaussian", "points": [0, 1, 2]})


def test_make_membership_function_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        make_membership_function({"type": "triangular", "points": [0, 1]})
