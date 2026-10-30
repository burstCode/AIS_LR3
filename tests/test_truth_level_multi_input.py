from pathlib import Path

import numpy as np
import pytest

from fuzzy_lr3 import truth_level
from fuzzy_lr3.defuzzify import centroid
from fuzzy_lr3.parser import load_rulebase

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"

# Разрешение монитора: 1440p = 2560x1440 px, в млн пикселей кадра.
NEUTRAL_RES_COMPACT = {"Разрешение монитора": 3.6864, "Желаемая компактность корпуса": 5.0}


@pytest.fixture(scope="module")
def multi_input_rulebase():
    return load_rulebase(RULES_DIR / "multi_input.json")


def crisp_output(rb, inputs, implication="mamdani"):
    result = truth_level.infer(rb, {**NEUTRAL_RES_COMPACT, **inputs}, implication=implication)
    return centroid(rb.output_variable.universe, result.output)


def test_requires_all_input_variables(multi_input_rulebase):
    with pytest.raises(ValueError, match="Не заданы"):
        truth_level.infer(multi_input_rulebase, {"Бюджет": 50000.0})


@pytest.mark.parametrize("implication", ["lukasiewicz", "zadeh"])
def test_logical_implications_depend_on_inputs(multi_input_rulebase, implication):
    low = crisp_output(multi_input_rulebase, {
        "Бюджет": 15000.0, "Требовательность задач": 1.0, "Важность видеокарты": 1.0,
    }, implication=implication)
    high = crisp_output(multi_input_rulebase, {
        "Бюджет": 200000.0, "Требовательность задач": 9.5, "Важность видеокарты": 9.5,
    }, implication=implication)

    assert high - low > 1.0


def test_fuzzy_inputs_with_singletons_match_crisp(multi_input_rulebase):
    rb = multi_input_rulebase
    crisp = {name: float(rb.input_variable(name).universe[400]) for name in rb.input_vars}
    fuzzy = {
        name: (rb.input_variable(name).universe == value).astype(float)
        for name, value in crisp.items()
    }

    np.testing.assert_allclose(
        truth_level.infer_fuzzy(rb, fuzzy).output,
        truth_level.infer(rb, crisp).output,
        atol=1e-9,
    )


def test_contradicting_rules_give_empty_output_for_logical_implications(multi_input_rulebase):
    # Правила #1 и #29 истинны полностью, а их следствия не пересекаются.
    result = truth_level.infer(multi_input_rulebase, {
        "Бюджет": 0.0, "Требовательность задач": 0.0, "Важность видеокарты": 0.0,
        "Разрешение монитора": 8.2944, "Желаемая компактность корпуса": 0.0,
    }, implication="lukasiewicz")

    assert result.output.max() == pytest.approx(0.0)


def test_budget_terms_are_the_budgets_from_lr1_and_lr2(multi_input_rulebase):
    budget = multi_input_rulebase.input_variable("Бюджет")

    assert list(budget.terms) == ["65000 ₽", "80000 ₽", "135000 ₽"]
    for term, value in [("65000 ₽", 65000.0), ("80000 ₽", 80000.0), ("135000 ₽", 135000.0)]:
        assert budget.membership(term, value) == pytest.approx(1.0)


def test_rulebase_has_at_least_the_required_rule_count(multi_input_rulebase):
    assert len(multi_input_rulebase.rules) >= 7


def test_low_budget_light_workload_yields_low_output(multi_input_rulebase):
    y = crisp_output(multi_input_rulebase, {
        "Бюджет": 10000.0,
        "Требовательность задач": 1.0,
        "Важность видеокарты": 1.0,
    })
    assert y < 4.0  # около терма "Низкий"/"Ниже среднего"


def test_high_budget_heavy_workload_high_gpu_yields_high_output(multi_input_rulebase):
    y = crisp_output(multi_input_rulebase, {
        "Бюджет": 200000.0,
        "Требовательность задач": 9.5,
        "Важность видеокарты": 9.5,
    })
    assert y > 9.0  # около терма "Очень высокий"


def test_output_is_monotonic_in_budget_other_inputs_fixed(multi_input_rulebase):
    fixed = {"Требовательность задач": 5.0, "Важность видеокарты": 5.0}

    budgets = [10000.0, 60000.0, 100000.0, 150000.0, 200000.0]
    outputs = [crisp_output(multi_input_rulebase, {"Бюджет": b, **fixed}) for b in budgets]

    # Нестрогая монотонность с допуском на погрешность округления - соседние
    # бюджеты иногда дают неразличимый выход (например, оба насыщают один
    # и тот же выходной терм), это не нарушение монотонности.
    for earlier, later in zip(outputs, outputs[1:]):
        assert later >= earlier - 1e-6


def test_output_stays_within_universe_bounds(multi_input_rulebase):
    lo, hi = multi_input_rulebase.output_variable.universe[[0, -1]]

    for budget in (0.0, 70000.0, 150000.0, 210000.0):
        for workload in (0.0, 5.0, 10.0):
            for gpu in (0.0, 5.0, 10.0):
                for res in (2.0736, 3.6864, 8.2944):
                    for compact in (0.0, 5.0, 10.0):
                        result = truth_level.infer(multi_input_rulebase, {
                            "Бюджет": budget,
                            "Требовательность задач": workload,
                            "Важность видеокарты": gpu,
                            "Разрешение монитора": res,
                            "Желаемая компактность корпуса": compact,
                        })
                        y = centroid(multi_input_rulebase.output_variable.universe, result.output)
                        assert lo <= y <= hi


def test_4k_resolution_pulls_output_up_even_on_low_budget(multi_input_rulebase):
    base = {"Бюджет": 10000.0, "Требовательность задач": 3.0, "Важность видеокарты": 3.0}

    y_1080p = crisp_output(multi_input_rulebase, {**base, "Разрешение монитора": 2.0736})
    y_4k = crisp_output(multi_input_rulebase, {**base, "Разрешение монитора": 8.2944})

    assert y_4k > y_1080p


def test_high_compactness_demand_caps_output_on_high_budget(multi_input_rulebase):
    base = {"Бюджет": 200000.0, "Требовательность задач": 5.0, "Важность видеокарты": 5.0,
            "Разрешение монитора": 3.6864}

    y_free_form = crisp_output(multi_input_rulebase, {**base, "Желаемая компактность корпуса": 0.0})
    y_compact = crisp_output(multi_input_rulebase, {**base, "Желаемая компактность корпуса": 10.0})

    assert y_compact < y_free_form
