"""Механизм вывода на основе уровней (степеней) истинности предпосылок
правил - упрощённый прямой алгоритм Мамдани. В отличие от полного
композиционного механизма (composition.py), работает для систем с любым
числом входных переменных и не требует построения отношения на
произведении универсумов.

Для каждого правила k:
  1. Степень истинности предпосылки (уровень отсечения) - T-норма min
     по всем условиям правила. Для чёткого входа x_i степень условия -
     это mu_i(x_i), для нечёткого входа A_i' - степень совместимости
     sup_x min(A_i'(x), mu_i(x)).
  2. Выходное множество правила "отсекается"/масштабируется импликацией:
     B_k'(y) = I(alpha_k, B_k(y)).
Итоговое выходное множество - агрегация множеств B_k'(y) по всем правилам
(max для Мамдани и Ларсена, min для Лукасевича и Заде).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aggregation import DEFAULT_AGGREGATION, aggregate
from .implications import IMPLICATIONS
from .variable import Rule, RuleBase


@dataclass
class TruthLevelResult:
    output: np.ndarray                 # агрегированное B'(y) по универсуму выхода
    rule_strengths: dict[int, float]   # alpha_k по каждому правилу
    rule_outputs: dict[int, np.ndarray]  # B_k'(y) по каждому правилу (до агрегации)


def firing_strength(rule_base: RuleBase, rule: Rule, inputs: dict[str, float]) -> float:
    """Степень истинности предпосылки правила: min по всем условиям (связка «И»)."""
    degrees = [
        float(rule_base.input_variable(var_name).membership(term_name, inputs[var_name]))
        for var_name, term_name in rule.antecedent.items()
    ]
    return min(degrees)


def fuzzy_firing_strength(rule_base: RuleBase, rule: Rule, inputs: dict[str, np.ndarray]) -> float:
    """Степень истинности предпосылки при нечётких входах: min по условиям
    от степени совместимости sup_x min(A'(x), mu(x)) входа с термом условия."""
    degrees = [
        float(np.max(np.minimum(inputs[var_name], rule_base.input_variable(var_name).term_curve(term_name))))
        for var_name, term_name in rule.antecedent.items()
    ]
    return min(degrees)


def infer(
    rule_base: RuleBase,
    inputs: dict[str, float],
    *,
    implication: str = "mamdani",
    aggregation: str | None = None,
) -> TruthLevelResult:
    """Вывод для чётких значений входов."""
    _check_inputs(rule_base, inputs)
    strengths = {rule.index: firing_strength(rule_base, rule, inputs) for rule in rule_base.rules}
    return _infer_from_strengths(rule_base, strengths, implication, aggregation)


def infer_fuzzy(
    rule_base: RuleBase,
    inputs: dict[str, np.ndarray],
    *,
    implication: str = "mamdani",
    aggregation: str | None = None,
) -> TruthLevelResult:
    """Вывод для нечётких входов: inputs - принадлежности A_i'(x) по универсуму
    каждой входной переменной."""
    _check_inputs(rule_base, inputs)
    strengths = {rule.index: fuzzy_firing_strength(rule_base, rule, inputs) for rule in rule_base.rules}
    return _infer_from_strengths(rule_base, strengths, implication, aggregation)


def _check_inputs(rule_base: RuleBase, inputs: dict) -> None:
    missing = set(rule_base.input_vars) - inputs.keys()
    if missing:
        raise ValueError(f"Не заданы значения входных переменных: {sorted(missing)}")


def _infer_from_strengths(
    rule_base: RuleBase,
    strengths: dict[int, float],
    implication: str,
    aggregation: str | None,
) -> TruthLevelResult:
    impl = IMPLICATIONS[implication]
    output_var = rule_base.output_variable

    rule_outputs: dict[int, np.ndarray] = {}
    clipped_sets = []

    for rule in rule_base.rules:
        b = output_var.term_curve(rule.consequent)
        clipped = impl(strengths[rule.index], b)

        rule_outputs[rule.index] = clipped
        clipped_sets.append(clipped)

    aggregated = aggregate(aggregation or DEFAULT_AGGREGATION[implication], clipped_sets)
    return TruthLevelResult(output=aggregated, rule_strengths=strengths, rule_outputs=rule_outputs)
