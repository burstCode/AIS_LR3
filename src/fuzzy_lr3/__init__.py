"""Оболочка нечёткой системы логического вывода (ЛР №3).

Публичный API:
  - load_rulebase(path)            - загрузка базы правил и нечётких множеств из JSON
  - composition.infer(...)         - полный композиционный механизм (1 вход)
  - truth_level.infer(...)         - механизм на основе уровней истинности (N чётких входов)
  - truth_level.infer_fuzzy(...)   - тот же механизм для нечётких входов
  - defuzzify.centroid(...)        - дефазификация методом центра тяжести
"""

from . import aggregation, composition, defuzzify, implications, truth_level
from .membership import Trapezoidal, make_membership_function
from .parser import load_rulebase
from .variable import FuzzyVariable, Rule, RuleBase

__all__ = [
    "aggregation",
    "composition",
    "defuzzify",
    "implications",
    "truth_level",
    "Trapezoidal",
    "make_membership_function",
    "load_rulebase",
    "FuzzyVariable",
    "Rule",
    "RuleBase",
]
