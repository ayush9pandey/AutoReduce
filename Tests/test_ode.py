#  Copyright (c) 2020, Build-A-Cell. All rights reserved.
#  See LICENSE file in the project root directory for details.

import pytest
import numpy as np
from scipy.integrate import odeint
from sympy import Symbol

from autoreduce.ode import ODE
from autoreduce.system import System 
from autoreduce.utils import get_ODE


@pytest.fixture
def test_ode_system():
    """
    Create a test system for ODE testing
    """
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    D = Symbol("D")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    k3 = Symbol("k3")
    
    params = [k1, k2, k3]
    x = [A, B, C, D]
    f = [-k1 * A**2 * B + k2 * C,
         -k1 * A**2 * B + k2 * C,
         k1 * A**2 * B - k2 * C - k3 * C,
         k3 * C]
    
    init_cond = np.ones(4)
    params_values = [2, 4, 6]
    timepoints = np.linspace(0, 10, 100)
    
    system = System(x, f, params=params, x_init=init_cond, params_values=params_values)
    ode_object = ODE(x, f, params, None, None, None, None,
                    params_values=params_values, x_init=init_cond,
                    input_values=None, timepoints=timepoints)
    
    return system, ode_object, timepoints, params_values


def test_ode_objects(test_ode_system):
    """
    Test ODE object creation and properties
    """
    system, ode_object, timepoints, _ = test_ode_system
    
    # Test ODE creation
    assert isinstance(ode_object, ODE)
    assert isinstance(ode_object, System)
    
    # Test ODE creation from system
    ode_object_same = get_ODE(system, timepoints=timepoints)
    assert ode_object.f == ode_object_same.f
    
    # Test system extraction
    extracted_system = ode_object.get_system()
    assert isinstance(extracted_system, System)
    
    # Test system equality
    assert extracted_system.x == system.x
    assert extracted_system.f == system.f
    assert extracted_system.params == system.params
    assert np.array_equal(extracted_system.x_init, system.x_init)
    assert extracted_system.params_values == system.params_values


def test_ode_solutions(test_ode_system):
    """
    Test ODE solutions by comparing with manual scipy solution
    """
    _, ode_object, timepoints, params_values = test_ode_system
    
    # Get solution from ODE class
    solutions = ode_object.solve_system()
    assert isinstance(solutions, np.ndarray)
    
    # Solve manually using scipy
    def scipy_odeint_func(x, t):            
        k1, k2, k3 = params_values
        A, B, C, D = x
        return np.array([-k1 * A**2 * B + k2 * C,
                        -k1 * A**2 * B + k2 * C,
                        k1 * A**2 * B - k2 * C - k3 * C,
                        k3 * C])
    
    solutions_manual = odeint(scipy_odeint_func, y0=np.ones(4), t=timepoints)
    assert np.array_equal(solutions, solutions_manual)
