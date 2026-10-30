"""Дефазификация методом центра тяжести (центроида)."""

from __future__ import annotations

import numpy as np


def centroid(universe: np.ndarray, membership: np.ndarray) -> float:
    """y* = интеграл(y * mu(y)) / интеграл(mu(y)) по дискретизированному универсуму.

    Если выходное множество пусто (нулевая площадь - ни одно правило не
    сработало либо, при агрегации min, правила прямо противоречат друг другу),
    возвращается середина универсума как вырожденный случай.
    """
    total_area = np.trapezoid(membership, universe)
    if total_area <= 1e-12:
        return float((universe[0] + universe[-1]) / 2.0)

    moment = np.trapezoid(universe * membership, universe)
    return float(moment / total_area)
