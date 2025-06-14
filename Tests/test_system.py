#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

import pytest
import numpy as np
from autoreduce.system import System
from sympy import Symbol
import warnings


@pytest.fixture
def test_systems():
    """
    Create two identical test systems for equality testing
    """
    # Create a simple test system
    A = Symbol("A")
    B = Symbol("B")
    k = Symbol("k")
    
    x = [A, B]
    f = [-k * A, k * A]
    params = [k]
    x_init = np.array([1.0, 0.0])
    params_values = [2.0]
    
    system1 = System(x, f, params=params, x_init=x_init, params_values=params_values)
    system2 = System(x, f, params=params, x_init=x_init, params_values=params_values)
    return system1, system2


def test_system_equality(test_systems):
    """
    Test all properties of two systems for equality
    """
    system1, system2 = test_systems
    
    # Test states
    assert system1.x == system2.x
    
    # Test ODEs
    assert system1.f == system2.f
    
    # Test output functions
    assert system1.g == system2.g
    assert system1.h == system2.h
    
    # Test parameters
    assert system1.params == system2.params
    assert system1.params_values == system2.params_values
    
    # Test initial conditions
    if system1.x_init is not None and system2.x_init is not None:
        assert np.array_equal(system1.x_init, system2.x_init)
    
    # Test C matrix
    if system1.C is not None and system2.C is not None:
        assert np.array_equal(system1.C, system2.C)


def test_system_inequality():
    """
    Test that different systems are not equal
    """
    # Create two different systems
    A = Symbol("A")
    B = Symbol("B")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    
    # System 1
    x1 = [A, B]
    f1 = [-k1 * A, k1 * A]
    params1 = [k1]
    x_init1 = np.array([1.0, 0.0])
    params_values1 = [2.0]
    system1 = System(x1, f1, params=params1, x_init=x_init1, params_values=params_values1)
    
    # System 2 (different parameter value)
    x2 = [A, B]
    f2 = [-k2 * A, k2 * A]
    params2 = [k2]
    x_init2 = np.array([1.0, 0.0])
    params_values2 = [3.0]  # Different parameter value
    system2 = System(x2, f2, params=params2, x_init=x_init2, params_values=params_values2)
    
    # Test that they are not equal
    assert system1.params_values != system2.params_values
    assert system1.f != system2.f  # Different parameter symbol
    
