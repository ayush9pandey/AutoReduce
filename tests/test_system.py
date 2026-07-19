# Copyright (c) 2020, Ayush Pandey. All rights reserved.
# See LICENSE file in the project root directory for details.

import numpy as np

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
    assert np.array_equal(system_1.x_init, np.ones(4))
    assert system_1.C is None
    assert system_1.g is None
    assert system_1.h is None
    assert system_1.u is None


def test_public_imports_are_exported():
    """Confirm common system and reduction APIs are importable."""
    import autoreduce.system
    import autoreduce.utils

    assert autoreduce.System is System
    assert autoreduce.system.System is System
    assert autoreduce.utils.get_ODE.__name__ == "get_ODE"
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
