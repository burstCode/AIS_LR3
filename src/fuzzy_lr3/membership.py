"""Функции принадлежности нечётких множеств: трапециевидная и её частный
случай - треугольная.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trapezoidal:
    """Трапециевидная функция принадлежности с вершинами a <= b <= c <= d.

    Растёт линейно от 0 при x=a до 1 при x=b, держит 1 на [b, c],
    затем убывает линейно до 0 при x=d. Треугольная функция - частный
    случай b == c; функция-"плечо" (плато до края домена) - случай
    a == b или c == d.
    """

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError(
                f"Параметры трапеции должны быть упорядочены (a<=b<=c<=d): {self}")

    def __call__(self, x: Any) -> np.ndarray | float:
        scalar_input = np.ndim(x) == 0
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.zeros_like(xs)

        if self.b > self.a:
            rising = (xs > self.a) & (xs < self.b)
            y[rising] = (xs[rising] - self.a) / (self.b - self.a)

        plateau = (xs >= self.b) & (xs <= self.c)
        y[plateau] = 1.0

        if self.d > self.c:
            falling = (xs > self.c) & (xs < self.d)
            y[falling] = (self.d - xs[falling]) / (self.d - self.c)

        return float(y[0]) if scalar_input else y


def make_membership_function(spec: dict) -> Trapezoidal:
    """Строит функцию принадлежности из описания term'а в файле базы правил.

    Поддерживаемые формы:
      {"type": "triangular", "points": [a, b, c]}       -> Trapezoidal(a, b, b, c)
      {"type": "trapezoidal", "points": [a, b, c, d]}    -> Trapezoidal(a, b, c, d)
    """
    mf_type = spec.get("type")
    points = spec.get("points")

    if mf_type == "triangular":
        if not points or len(points) != 3:
            raise ValueError(f"Треугольная функция принадлежности требует 3 точки: {spec}")
        a, b, c = points
        return Trapezoidal(a, b, b, c)

    if mf_type == "trapezoidal":
        if not points or len(points) != 4:
            raise ValueError(f"Трапециевидная функция принадлежности требует 4 точки: {spec}")
        a, b, c, d = points
        return Trapezoidal(a, b, c, d)

    raise ValueError(f"Неизвестный тип функции принадлежности: {mf_type!r}")
