"""Нечёткие импликации I(a, b), где a - степень истинности предпосылки,
b - принадлежность элемента выходного терма.

Используются как для построения нечёткого отношения в композиционном
механизме вывода, так и для "отсечения"/масштабирования консеквента
в механизме на основе уровней истинности предпосылок.
"""

from __future__ import annotations

import numpy as np


def mamdani(a, b):
    """Импликация Мамдани (min): I(a, b) = min(a, b)."""
    return np.minimum(a, b)


def larsen(a, b):
    """Импликация Ларсена (алгебраическое произведение): I(a, b) = a * b."""
    return a * b


def lukasiewicz(a, b):
    """Импликация Лукасевича: I(a, b) = min(1, 1 - a + b)."""
    return np.minimum(1.0, 1.0 - a + b)


def zadeh(a, b):
    """Импликация Заде: I(a, b) = max(1 - a, min(a, b))."""
    return np.maximum(1.0 - a, np.minimum(a, b))


IMPLICATIONS = {
    "mamdani": mamdani,
    "larsen": larsen,
    "lukasiewicz": lukasiewicz,
    "zadeh": zadeh,
}

IMPLICATION_TITLES = {
    "mamdani": "Мамдани (min)",
    "larsen": "Ларсен (произведение)",
    "lukasiewicz": "Лукасевич",
    "zadeh": "Заде",
}
