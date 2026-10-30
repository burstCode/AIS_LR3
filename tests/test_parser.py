import json
from pathlib import Path

import pytest

from fuzzy_lr3.parser import load_rulebase

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"


MINIMAL_VALID = {
    "variables": {
        "X": {"universe": [0, 10], "terms": {"A": {"type": "triangular", "points": [0, 5, 10]}}},
        "Y": {"universe": [0, 10], "terms": {"B": {"type": "triangular", "points": [0, 5, 10]}}},
    },
    "input_vars": ["X"],
    "output_var": "Y",
    "rules": [{"if": {"X": "A"}, "then": "B"}] * 7,
}


def write_json(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_minimal_valid_rulebase(tmp_path):
    rb = load_rulebase(write_json(tmp_path, MINIMAL_VALID))

    assert rb.input_vars == ["X"]
    assert rb.output_var == "Y"
    assert len(rb.rules) == 7


def test_rejects_fewer_than_seven_rules(tmp_path):
    data = {**MINIMAL_VALID, "rules": MINIMAL_VALID["rules"][:3]}
    with pytest.raises(ValueError, match="7"):
        load_rulebase(write_json(tmp_path, data))


def test_rejects_rule_referencing_unknown_term(tmp_path):
    data = {**MINIMAL_VALID, "rules": [{"if": {"X": "НЕТ_ТАКОГО"}, "then": "B"}] * 7}
    with pytest.raises(ValueError, match="не определён"):
        load_rulebase(write_json(tmp_path, data))


def test_rejects_rule_consequent_with_unknown_term(tmp_path):
    data = {**MINIMAL_VALID, "rules": [{"if": {"X": "A"}, "then": "НЕТ_ТАКОГО"}] * 7}
    with pytest.raises(ValueError, match="не определён"):
        load_rulebase(write_json(tmp_path, data))


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("это не json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_rulebase(path)


def test_rejects_variable_with_inverted_universe(tmp_path):
    data = json.loads(json.dumps(MINIMAL_VALID))
    data["variables"]["X"]["universe"] = [10, 0]
    with pytest.raises(ValueError):
        load_rulebase(write_json(tmp_path, data))


def test_real_single_input_rulebase_loads_and_has_at_least_seven_rules():
    rb = load_rulebase(RULES_DIR / "single_input.json")

    assert rb.input_vars == ["Бюджет"]
    assert rb.output_var == "Класс сборки"
    assert len(rb.rules) >= 7


def test_real_multi_input_rulebase_loads_with_at_least_three_inputs():
    rb = load_rulebase(RULES_DIR / "multi_input.json")

    assert len(rb.input_vars) >= 3
    assert rb.output_var == "Класс сборки"
    assert len(rb.rules) >= 7
