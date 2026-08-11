# Copyright (c) 2020, Ayush Pandey. All rights reserved.
# See LICENSE file in the project root directory for details.

import numpy as np
import pytest
from sympy import Symbol

from autoreduce import solve_conservation_laws, solve_timescale_separation
from autoreduce.system.system import System


def test_system_equality(system_1):
    """Check that equivalent `System` objects compare equal."""
    system_copy = System(
        system_1.x,
        system_1.f,
        params=system_1.params,
        x_init=system_1.x_init,
        params_values=system_1.params_values,
        C=system_1.C,
        g=system_1.g,
        h=system_1.h,
        u=system_1.u,
        input_values=system_1.input_values,
    )
    assert system_1 == system_copy


def test_system_attributes(system_1):
    """Check that core symbolic system attributes are stored intact."""
    assert len(system_1.x) == 4
    assert len(system_1.f) == 4
    assert len(system_1.params) == 3
    assert system_1.params_values == [2, 4, 6]
    assert system_1.params_dict == {
        system_1.params[0]: 2,
        system_1.params[1]: 4,
        system_1.params[2]: 6,
    }
    assert np.array_equal(system_1.x_init, np.ones(4))
    assert system_1.C is None
    assert system_1.g is None
    assert system_1.h is None
    assert system_1.u is None


def test_system_can_be_created_with_params_dict():
    """Create a System from params_dict instead of params and params_values."""
    x = [Symbol("x")]
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    system = System(
        x,
        [-k1 * x[0] + k2],
        params_dict={k1: 2, k2: 4},
        x_init=[1.0],
    )

    assert system.params == [k1, k2]
    assert system.params_values == [2, 4]
    assert system.params_dict == {k1: 2, k2: 4}


def test_system_rejects_params_dict_with_params_values(system_1):
    """Avoid ambiguous parameter inputs."""
    with pytest.raises(ValueError, match="params_dict"):
        System(
            system_1.x,
            system_1.f,
            params=system_1.params,
            params_values=system_1.params_values,
            params_dict=system_1.params_dict,
            x_init=system_1.x_init,
        )


def test_system_get_and_set_param(system_1):
    """Read and write one parameter by exact symbol key."""
    first_param = system_1.params[0]

    assert system_1.get_param(first_param) == 2
    assert system_1.set_param(first_param, 10) == 10
    assert system_1.get_param(first_param) == 10
    assert system_1.params_values[0] == 10


def test_system_get_param_requires_exact_key(system_1):
    """Parameter names are exact keys in params_dict."""
    with pytest.raises(ValueError, match="was not found"):
        system_1.get_param("k1")


def test_system_update_param_dict(system_1):
    """Push direct params_dict edits back to params_values."""
    second_param = system_1.params[1]

    system_1.params_dict[second_param] = 20
    system_1.update_param_dict()

    assert system_1.params_values[1] == 20


def test_system_set_param_dict(system_1):
    """Set parameter values from a dictionary."""
    first_param = system_1.params[0]
    third_param = system_1.params[2]

    system_1.set_param_dict({first_param: 12, third_param: 30})

    assert system_1.params_dict[first_param] == 12
    assert system_1.params_dict[third_param] == 30
    assert system_1.params_values == [12, 4, 30]


def test_system_set_param_dict_requires_exact_keys(system_1):
    """set_param_dict uses exact parameter keys."""
    with pytest.raises(ValueError, match="were not found"):
        system_1.set_param_dict({"k1": 12})


def test_public_imports_are_exported():
    """Confirm common system and reduction APIs are importable."""
    import autoreduce.solvers
    import autoreduce.system
    import autoreduce.utils

    assert autoreduce.System is System
    assert autoreduce.system.System is System
    assert autoreduce.solvers.get_ODE.__name__ == "get_ODE"
    assert not hasattr(autoreduce, "get_reducible")
    with pytest.raises(AttributeError):
        getattr(autoreduce.utils, "get_reducible")
    with pytest.raises(AttributeError):
        getattr(autoreduce.utils, "get_ODE")
    assert solve_conservation_laws.__name__ == "solve_conservation_laws"
    assert solve_timescale_separation.__name__ == "solve_timescale_separation"


def test_system_string_and_pretty_print(system_1, capsys):
    """Check concise and rich text display for System objects."""
    assert str(system_1) == str(system_1.f)

    system_1.pretty_print()
    captured = capsys.readouterr()
    assert "AutoReduce System object with 4 state variables" in captured.out
    assert "system equations:" in captured.out
    assert r"\left[" in captured.out
