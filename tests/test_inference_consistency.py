"""Перекрёстная проверка корректности: два независимо реализованных
алгоритма должны сойтись на одном и том же результате там, где это следует
из теории:
  - при чётком (синглтон) входе полный композиционный механизм и уровни
    истинности совпадают для любой импликации и любой композиции;
  - при нечётком входе Мамдани + max-min совпадает с уровнями истинности,
    если alpha_k = sup_x min(A'(x), A_k(x)).
"""

from pathlib import Path

import numpy as np
import pytest

from fuzzy_lr3 import composition, truth_level
from fuzzy_lr3.defuzzify import centroid
from fuzzy_lr3.parser import load_rulebase

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"


@pytest.fixture(scope="module")
def single_input_rulebase():
    return load_rulebase(RULES_DIR / "single_input.json")


@pytest.mark.parametrize(
    ("implication", "comp_op"),
    [("mamdani", "max-min"), ("larsen", "max-prod")],
)
def test_compositional_matches_truth_level_for_crisp_input(single_input_rulebase, implication, comp_op):
    rb = single_input_rulebase
    input_var = rb.input_variable("Бюджет")

    # Берём точку ровно на сетке универсума, чтобы оба метода оперировали
    # одним и тем же числом без ошибки дискретизации при "привязке" синглтона.
    x0 = float(input_var.universe[423])

    comp_output = composition.infer(
        rb,
        composition.singleton(input_var.universe, x0),
        implication=implication,
        composition=comp_op,
        aggregation="max",
    )

    truth_result = truth_level.infer(rb, {"Бюджет": x0}, implication=implication, aggregation="max")

    np.testing.assert_allclose(comp_output, truth_result.output, atol=1e-9)


def test_compositional_and_truth_level_centroids_agree_for_mamdani(single_input_rulebase):
    rb = single_input_rulebase
    input_var = rb.input_variable("Бюджет")
    output_var = rb.output_variable
    x0 = float(input_var.universe[700])

    comp_output = composition.infer(
        rb,
        composition.singleton(input_var.universe, x0),
        implication="mamdani",
        composition="max-min",
    )
    truth_output = truth_level.infer(rb, {"Бюджет": x0}, implication="mamdani").output

    comp_crisp = centroid(output_var.universe, comp_output)
    truth_crisp = centroid(output_var.universe, truth_output)

    assert comp_crisp == pytest.approx(truth_crisp, abs=1e-6)


@pytest.mark.parametrize("implication", ["lukasiewicz", "zadeh"])
@pytest.mark.parametrize("comp_op", ["max-min", "max-prod"])
def test_crisp_input_mechanisms_coincide_for_logical_implications(single_input_rulebase, implication, comp_op):
    rb = single_input_rulebase
    input_var = rb.input_variable("Бюджет")
    x0 = float(input_var.universe[423])

    comp_output = composition.infer(
        rb, composition.singleton(input_var.universe, x0), implication=implication, composition=comp_op)
    truth_output = truth_level.infer(rb, {"Бюджет": x0}, implication=implication).output

    np.testing.assert_allclose(comp_output, truth_output, atol=1e-9)


def around(variable, center, spread):
    """Треугольное нечёткое наблюдение «около center»."""
    return np.clip(1 - np.abs(variable.universe - center) / spread, 0, 1)


def test_fuzzy_truth_level_matches_max_min_for_mamdani(single_input_rulebase):
    rb = single_input_rulebase
    fuzzy_input = around(rb.input_variable("Бюджет"), 150000, 20000)

    comp_output = composition.infer(rb, fuzzy_input, implication="mamdani", composition="max-min")
    truth_output = truth_level.infer_fuzzy(rb, {"Бюджет": fuzzy_input}, implication="mamdani").output

    np.testing.assert_allclose(comp_output, truth_output, atol=1e-9)


def test_fuzzy_truth_level_with_singleton_matches_crisp(single_input_rulebase):
    rb = single_input_rulebase
    input_var = rb.input_variable("Бюджет")
    x0 = float(input_var.universe[423])

    for implication in ("mamdani", "larsen", "lukasiewicz", "zadeh"):
        fuzzy = truth_level.infer_fuzzy(
            rb, {"Бюджет": composition.singleton(input_var.universe, x0)}, implication=implication).output
        crisp = truth_level.infer(rb, {"Бюджет": x0}, implication=implication).output
        np.testing.assert_allclose(fuzzy, crisp, atol=1e-9)


def test_fuzzy_input_separates_mechanisms(single_input_rulebase):
    rb = single_input_rulebase
    output_var = rb.output_variable
    fuzzy_input = around(rb.input_variable("Бюджет"), 150000, 20000)

    for implication in ("mamdani", "lukasiewicz", "zadeh"):
        max_min = composition.infer(rb, fuzzy_input, implication=implication, composition="max-min")
        max_prod = composition.infer(rb, fuzzy_input, implication=implication, composition="max-prod")
        assert centroid(output_var.universe, max_min) != pytest.approx(centroid(output_var.universe, max_prod), abs=0.01)


def test_fuzzy_input_differs_from_its_crisp_peak(single_input_rulebase):
    rb = single_input_rulebase
    fuzzy_input = around(rb.input_variable("Бюджет"), 150000, 20000)

    fuzzy = truth_level.infer_fuzzy(rb, {"Бюджет": fuzzy_input}, implication="mamdani").output
    crisp = truth_level.infer(rb, {"Бюджет": 150000.0}, implication="mamdani").output

    assert not np.allclose(fuzzy, crisp)


@pytest.mark.parametrize("implication", ["lukasiewicz", "zadeh"])
def test_logical_implications_depend_on_input(single_input_rulebase, implication):
    """При объединении через max такие импликации давали R(x, y) = 1 и
    постоянный выход 6.0 при любом входе."""
    rb = single_input_rulebase
    output_var = rb.output_variable

    relation = composition.build_relation(rb, implication=implication)
    assert relation.min() < 1.0

    outputs = [
        centroid(output_var.universe, truth_level.infer(rb, {"Бюджет": b}, implication=implication).output)
        for b in (65000.0, 135000.0)
    ]
    assert outputs[1] - outputs[0] > 1.0
