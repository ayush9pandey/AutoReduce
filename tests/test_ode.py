#  Copyright (c) 2020, Build-A-Cell. All rights reserved.
#  See LICENSE file in the project root directory for details.

import pytest  # type: ignore
import libsbml  # type: ignore
import numpy as np  # type: ignore
from scipy.integrate import odeint  # type: ignore

from autoreduce.ode import ODE
from autoreduce.system import System
from autoreduce.utils import get_ODE


def test_ode_objects(system):
    import numpy as np  # type: ignore

    timepoints = np.linspace(0, 10, 100)
    ode_object = ODE(
        system.x,
        system.f,
        system.params,
        system.C,
        system.g,
        system.h,
        system.u,
        params_values=[2, 4, 6],
        x_init=np.ones(4),
        input_values=system.input_values,
        timepoints=timepoints,
    )
    ode_object_same = get_ODE(system, timepoints=timepoints)
    assert isinstance(ode_object, ODE)
    assert isinstance(ode_object, System)
    assert ode_object.f == ode_object_same.f
    assert isinstance(ode_object.get_system(), System)

    # Compare values instead of objects
    ode_system = ode_object.get_system()
    assert ode_system.x == system.x
    assert ode_system.f == system.f
    assert ode_system.params == system.params
    assert ode_system.params_values == system.params_values
    assert np.array_equal(ode_system.x_init, system.x_init)
    assert ode_system.C == system.C
    assert ode_system.g == system.g
    assert ode_system.h == system.h
    assert ode_system.u == system.u
    assert ode_system.input_values == system.input_values


def test_ode_solutions(system):
    """
    Solve the ODE manually and solve using
    ODE class to compare the two solutions
    """
    timepoints = np.linspace(0, 10, 100)
    params_values = [2, 4, 6]
    ode_object = ODE(
        system.x,
        system.f,
        system.params,
        system.C,
        system.g,
        system.h,
        system.u,
        params_values=params_values,
        x_init=np.ones(4),
        input_values=system.input_values,
        timepoints=timepoints,
    )
    solutions = ode_object.solve_system()

    assert isinstance(solutions, np.ndarray)

    # Solve manually:
    def scipy_odeint_func(x, t):
        k1, k2, k3 = params_values
        A, B, C, D = x
        return np.array(
            [
                -k1 * A**2 * B + k2 * C,
                -k1 * A**2 * B + k2 * C,
                k1 * A**2 * B - k2 * C - k3 * C,
                k3 * C,
            ]
        )

    solutions_manual = odeint(scipy_odeint_func, y0=np.ones(4), t=timepoints)
    assert (solutions == solutions_manual).all()
