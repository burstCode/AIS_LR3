"""Лингвистическая переменная и свод правил нечёткой системы."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .membership import Trapezoidal


@dataclass
class FuzzyVariable:
    """Лингвистическая переменная: имя, дискретизированный универсум и термы."""

    name: str
    universe: np.ndarray
    terms: dict[str, Trapezoidal]

    def membership(self, term: str, x) -> np.ndarray | float:
        return self.terms[term](x)

    def term_curve(self, term: str) -> np.ndarray:
        """Значения функции принадлежности терма по всему универсуму."""
        return self.terms[term](self.universe)


@dataclass(frozen=True)
class Rule:
    """Правило нечёткой продукции: ЕСЛИ (И по condition'ам) ТО consequent."""

    index: int
    antecedent: dict[str, str]
    consequent: str

    def describe(self, output_var: str) -> str:
        conditions = " И ".join(f"{var}={term}" for var, term in self.antecedent.items())
        return f"ЕСЛИ {conditions} ТО {output_var}={self.consequent}"


@dataclass
class RuleBase:
    """Нечёткая база знаний: переменные, их роли и свод правил."""

    variables: dict[str, FuzzyVariable]
    input_vars: list[str]
    output_var: str
    rules: list[Rule] = field(default_factory=list)

    @property
    def output_variable(self) -> FuzzyVariable:
        return self.variables[self.output_var]

    def input_variable(self, name: str) -> FuzzyVariable:
        return self.variables[name]
