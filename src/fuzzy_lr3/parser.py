"""Загрузка нечётких множеств и свода правил из отдельного текстового
(JSON) файла - база знаний отделена от программного модуля, как и
в ЛР №1/№2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .membership import make_membership_function
from .variable import FuzzyVariable, Rule, RuleBase


def load_rulebase(path: str | Path, *, universe_points: int = 1001) -> RuleBase:
    """Разбирает и проверяет файл базы правил нечёткой системы.

    Ожидаемый формат:
    {
      "variables": {
        "<Имя переменной>": {
          "universe": [lo, hi],
          "terms": {"<Терм>": {"type": "triangular"|"trapezoidal", "points": [...]}}
        }
      },
      "input_vars": ["<Имя входной переменной>", ...],
      "output_var": "<Имя выходной переменной>",
      "rules": [{"if": {"<var>": "<term>", ...}, "then": "<term выходной переменной>"}]
    }
    """
    raw = _read_json(path)
    variables = _parse_variables(raw, universe_points=universe_points)
    input_vars, output_var = _parse_roles(raw, variables)
    rules = _parse_rules(raw, variables, input_vars, output_var)

    if len(rules) < 7:
        raise ValueError(
            f"В своде правил должно быть не менее 7 правил, найдено {len(rules)}.")

    return RuleBase(variables=variables, input_vars=input_vars, output_var=output_var, rules=rules)


def _read_json(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Файл {path} не является корректным JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Файл {path} должен содержать JSON-объект верхнего уровня.")

    return data


def _parse_variables(raw: dict, *, universe_points: int) -> dict[str, FuzzyVariable]:
    variables_spec = raw.get("variables")
    if not isinstance(variables_spec, dict) or not variables_spec:
        raise ValueError("Файл базы правил должен содержать непустой раздел 'variables'.")

    variables: dict[str, FuzzyVariable] = {}
    for name, spec in variables_spec.items():
        universe_bounds = spec.get("universe")
        if not universe_bounds or len(universe_bounds) != 2:
            raise ValueError(f"Переменная «{name}»: нужен универсум вида [lo, hi].")

        lo, hi = universe_bounds
        if lo >= hi:
            raise ValueError(f"Переменная «{name}»: universe lo={lo} должно быть меньше hi={hi}.")

        terms_spec = spec.get("terms")
        if not isinstance(terms_spec, dict) or not terms_spec:
            raise ValueError(f"Переменная «{name}»: нужен непустой раздел 'terms'.")

        universe = np.linspace(lo, hi, universe_points)
        terms = {
            term_name: make_membership_function(term_spec)
            for term_name, term_spec in terms_spec.items()
        }
        variables[name] = FuzzyVariable(name=name, universe=universe, terms=terms)

    return variables


def _parse_roles(raw: dict, variables: dict[str, FuzzyVariable]) -> tuple[list[str], str]:
    input_vars = raw.get("input_vars")
    if not input_vars or not isinstance(input_vars, list):
        raise ValueError("Файл базы правил должен содержать непустой список 'input_vars'.")

    output_var = raw.get("output_var")
    if not output_var or not isinstance(output_var, str):
        raise ValueError("Файл базы правил должен содержать строку 'output_var'.")

    for var_name in [*input_vars, output_var]:
        if var_name not in variables:
            raise ValueError(f"Переменная «{var_name}» упомянута, но не описана в 'variables'.")

    return list(input_vars), output_var


def _parse_rules(
    raw: dict,
    variables: dict[str, FuzzyVariable],
    input_vars: list[str],
    output_var: str,
) -> list[Rule]:
    rules_spec = raw.get("rules")
    if not isinstance(rules_spec, list) or not rules_spec:
        raise ValueError("Файл базы правил должен содержать непустой список 'rules'.")

    rules: list[Rule] = []
    for i, rule_spec in enumerate(rules_spec):
        antecedent = rule_spec.get("if")
        consequent = rule_spec.get("then")

        if not antecedent or not isinstance(antecedent, dict):
            raise ValueError(f"Правило #{i + 1}: нужно непустое условие 'if'.")
        if not consequent or not isinstance(consequent, str):
            raise ValueError(f"Правило #{i + 1}: нужно следствие 'then' (имя терма).")

        for var_name, term_name in antecedent.items():
            if var_name not in input_vars:
                raise ValueError(
                    f"Правило #{i + 1}: «{var_name}» не входная переменная этой системы.")
            if term_name not in variables[var_name].terms:
                raise ValueError(
                    f"Правило #{i + 1}: терм «{term_name}» не определён для «{var_name}».")

        if consequent not in variables[output_var].terms:
            raise ValueError(
                f"Правило #{i + 1}: терм «{consequent}» не определён для «{output_var}».")

        rules.append(Rule(index=i, antecedent=dict(antecedent), consequent=consequent))

    return rules
