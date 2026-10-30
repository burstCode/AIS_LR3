import pytest

from fuzzy_lr3.implications import IMPLICATIONS, lukasiewicz, mamdani, larsen, zadeh


def test_mamdani():
    assert mamdani(0.6, 0.3) == pytest.approx(0.3)
    assert mamdani(0.2, 0.9) == pytest.approx(0.2)


def test_larsen():
    assert larsen(0.6, 0.3) == pytest.approx(0.18)


def test_lukasiewicz():
    assert lukasiewicz(0.6, 0.3) == pytest.approx(0.7)
    assert lukasiewicz(0.0, 0.0) == pytest.approx(1.0)


def test_zadeh():
    assert zadeh(0.6, 0.3) == pytest.approx(0.4)
    assert zadeh(0.2, 0.9) == pytest.approx(0.8)


def test_all_implications_return_values_in_unit_interval():
    for name, impl in IMPLICATIONS.items():
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            for b in (0.0, 0.25, 0.5, 0.75, 1.0):
                value = impl(a, b)
                assert 0.0 - 1e-9 <= value <= 1.0 + 1e-9, (name, a, b, value)


def test_full_antecedent_truth_reduces_implication_to_consequent():
    # При alpha=1 (предпосылка истинна целиком) классические импликации
    # Мамдани и Ларсена должны просто вернуть сам консеквент b.
    for b in (0.0, 0.3, 0.7, 1.0):
        assert mamdani(1.0, b) == pytest.approx(b)
        assert larsen(1.0, b) == pytest.approx(b)
