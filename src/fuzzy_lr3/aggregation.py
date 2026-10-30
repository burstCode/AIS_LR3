"""Операторы агрегации выходных нечётких множеств нескольких правил
(или нечётких отношений нескольких правил) в одно итоговое множество.

Выбор оператора зависит от смысла импликации. Мамдани и Ларсен - конъюнкции:
правило описывает точку, где оно верно, а не сработавшее правило даёт 0,
поэтому правила объединяются S-нормой (max и др.). Лукасевич и Заде -
логические импликации: не сработавшее правило даёт I(0, b) = 1 («ничего не
утверждает»), поэтому правила объединяются T-нормой (min) - иначе единицы
таких правил заливают весь результат и выход перестаёт зависеть от входа.
"""

from __future__ import annotations

from functools import reduce

import numpy as np


def agg_max(a, b):
    """Max (объединение по Заде) - стандартная агрегация для систем Мамдани."""
    return np.maximum(a, b)


def agg_min(a, b):
    """Min (пересечение по Заде) - агрегация для логических импликаций."""
    return np.minimum(a, b)


def agg_probabilistic_sum(a, b):
    """Вероятностная сумма: a + b - a*b."""
    return a + b - a * b


def agg_bounded_sum(a, b):
    """Ограниченная сумма (сумма Лукасевича): min(1, a + b)."""
    return np.minimum(1.0, a + b)


AGGREGATIONS = {
    "max": agg_max,
    "min": agg_min,
    "probabilistic_sum": agg_probabilistic_sum,
    "bounded_sum": agg_bounded_sum,
}

AGGREGATION_TITLES = {
    "max": "max (объединение)",
    "min": "min (пересечение)",
    "probabilistic_sum": "вероятностная сумма",
    "bounded_sum": "ограниченная сумма",
}


DEFAULT_AGGREGATION = {
    "mamdani": "max",
    "larsen": "max",
    "lukasiewicz": "min",
    "zadeh": "min",
}


def aggregate(op_name: str, sets: list[np.ndarray]) -> np.ndarray:
    """Последовательно агрегирует список нечётких множеств/отношений."""
    if not sets:
        raise ValueError("Список множеств для агрегации пуст.")

    op = AGGREGATIONS[op_name]
    return reduce(op, sets)
