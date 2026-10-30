"""Вспомогательные функции построения графиков (термы, результаты вывода,
сравнение импликаций) для демонстрационных скриптов.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .variable import FuzzyVariable


def plot_terms(ax: plt.Axes, variable: FuzzyVariable, *, title: str | None = None) -> None:
    """Рисует все функции принадлежности переменной на одной оси."""
    for term_name, mf in variable.terms.items():
        ax.plot(variable.universe, mf(variable.universe), label=term_name)

    ax.set_title(title or variable.name)
    ax.set_xlabel(variable.name)
    ax.set_ylabel("μ")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


def plot_output_set(
    ax: plt.Axes,
    universe: np.ndarray,
    membership: np.ndarray,
    *,
    label: str,
    crisp_value: float | None = None,
) -> None:
    """Рисует одно выходное нечёткое множество и (опционально) чёткую точку
    после дефазификации."""
    line, = ax.plot(universe, membership, label=label)
    if crisp_value is not None:
        ax.axvline(crisp_value, color=line.get_color(), linestyle="--", alpha=0.6)
