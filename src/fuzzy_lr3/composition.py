"""Полный композиционный механизм вывода (Compositional Rule of Inference)
для систем с одной входной переменной.

Для каждого правила строится нечёткое отношение
    R_k(x, y) = I(A_k(x), B_k(y))
по всем точкам дискретизированных универсумов входа x и выхода y
(I - выбранная импликация). Отношения правил агрегируются (max для
Мамдани и Ларсена, min для Лукасевича и Заде) в одно отношение R. Выход при входном нечётком множестве A'(x)
находится композицией:
    max-min:  B'(y) = max_x min(A'(x), R(x, y))
    max-prod: B'(y) = max_x [A'(x) * R(x, y)]

При вырожденном (чётком) входе A' - синглтон в точке x0 - обе формулы
сводятся к срезу R(x0, y), что проверяется в тестах.
"""

from __future__ import annotations

import numpy as np

from .aggregation import DEFAULT_AGGREGATION, aggregate
from .implications import IMPLICATIONS
from .variable import RuleBase


def build_relation(rule_base: RuleBase, *, implication: str, aggregation: str | None = None) -> np.ndarray:
    """Строит итоговое нечёткое отношение R(x, y) по всем правилам системы.

    Требует систему ровно с одной входной переменной. Если агрегация не
    задана, берётся подходящая для импликации (см. DEFAULT_AGGREGATION).
    """
    if len(rule_base.input_vars) != 1:
        raise ValueError(
            "Полная композиционная схема вывода реализована только для систем "
            "с одной входной переменной.")

    (input_name,) = rule_base.input_vars
    input_var = rule_base.input_variable(input_name)
    output_var = rule_base.output_variable
    impl = IMPLICATIONS[implication]

    relations = []
    for rule in rule_base.rules:
        a = input_var.term_curve(rule.antecedent[input_name])  # (nx,)
        b = output_var.term_curve(rule.consequent)              # (ny,)
        relations.append(impl(a[:, None], b[None, :]))          # (nx, ny)

    return aggregate(aggregation or DEFAULT_AGGREGATION[implication], relations)


def singleton(universe: np.ndarray, x0: float) -> np.ndarray:
    """Чёткое наблюдение x0 как вырожденное (синглтон) нечёткое множество."""
    membership = np.zeros_like(universe)
    idx = int(np.argmin(np.abs(universe - x0)))
    membership[idx] = 1.0
    return membership


def max_min_compose(input_membership: np.ndarray, relation: np.ndarray) -> np.ndarray:
    """B'(y) = max_x min(A'(x), R(x, y)) - max-min композиция."""
    return np.max(np.minimum(input_membership[:, None], relation), axis=0)


def max_prod_compose(input_membership: np.ndarray, relation: np.ndarray) -> np.ndarray:
    """B'(y) = max_x [A'(x) * R(x, y)] - max-произведение композиция."""
    return np.max(input_membership[:, None] * relation, axis=0)


COMPOSITIONS = {
    "max-min": max_min_compose,
    "max-prod": max_prod_compose,
}


def infer(
    rule_base: RuleBase,
    input_membership: np.ndarray,
    *,
    implication: str,
    composition: str,
    aggregation: str | None = None,
) -> np.ndarray:
    """Полный цикл: строит отношение по правилам и композирует его со входом."""
    relation = build_relation(rule_base, implication=implication, aggregation=aggregation)
    return COMPOSITIONS[composition](input_membership, relation)
