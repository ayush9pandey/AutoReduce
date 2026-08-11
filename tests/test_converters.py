from pathlib import Path

import numpy as np
import pytest
from sympy import Symbol

from autoreduce import load_ode_model, load_sbml


MODEL_FILE = Path(__file__).parent / "models" / "example_1.xml"


def test_load_sbml_reports_missing_file():
    missing_file = Path(__file__).parent / "models" / "missing.xml"

    with pytest.raises(FileNotFoundError, match="SBML file not found"):
        load_sbml(missing_file)


def test_load_sbml_rejects_unknown_output():
    with pytest.raises(ValueError, match="did not match any species"):
        load_sbml(MODEL_FILE, outputs=["not_a_species"])


def test_load_sbml_selects_matching_output(capsys):
    system = load_sbml(MODEL_FILE, outputs=["P"])

    captured = capsys.readouterr()
    assert "Your output 'P' is now set using the system.C matrix!" in captured.out
    np.testing.assert_array_equal(system.C, np.array([[0.0, 0.0, 1.0]]))


def test_load_sbml_creates_params_dict():
    system = load_sbml(MODEL_FILE)

    assert system.params_dict == dict(zip(system.params, system.params_values))


def test_load_sbml_can_rename_species_for_analysis():
    system = load_sbml(
        MODEL_FILE,
        outputs=["protein_E"],
        rename_species={"protein_E": "enzyme"},
    )

    assert system.x[0] == Symbol("enzyme")
    assert Symbol("enzyme") in system.f[1].free_symbols
    np.testing.assert_array_equal(system.C, np.array([[1.0, 0.0, 0.0]]))


def test_load_ode_model_rejects_unknown_output():
    with pytest.raises(ValueError, match="did not match any state"):
        load_ode_model(2, outputs=["x2"])


def test_load_ode_model_accepts_known_output():
    x, f, params = load_ode_model(2, 1, outputs=["x1"])

    assert [str(state) for state in x] == ["x0", "x1"]
    assert [str(ode) for ode in f] == ["f0", "f1"]
    assert [str(param) for param in params] == ["P0"]
