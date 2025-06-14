#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

import pytest
import warnings
from sympy import Symbol
import numpy as np
from autoreduce.system import System
from autoreduce.utils import get_reducible


@pytest.fixture
def test_system():
    """
    Create a test system for time scale separation testing
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
    
    system = System(x, f, params=params, x_init=init_cond,
                    params_values=params_values)
    reducible_system = get_reducible(system)
    return system, reducible_system, x, params


def test_reduced_models(test_system):
    """
    Test various reduced model combinations
    """
    _, reducible_system, x, params = test_system
    A, B, C, D = x
    k1, k2, k3 = params
    
    # Test AB reduction
    answer_AB = [-A**2*B*k1*k3/(k2 + k3), -A**2*B*k1*k3/(k2 + k3)]
    with pytest.warns(UserWarning, match='Solve time-scale separation failed'):
        reduced_system, collapsed_system = reducible_system.solve_timescale_separation([A, B])
        assert reduced_system.f == answer_AB
        assert collapsed_system.x == [C, D]
    
    # Test AD reduction
    with pytest.warns(UserWarning, match='Solve time-scale separation failed'):
        reduced_system, collapsed_system = reducible_system.solve_timescale_separation([A, D])
        assert reduced_system.f == [0, 0]
        assert collapsed_system.x == [B, C]
    
    # Test AC reduction
    with pytest.warns(UserWarning, match='Solve time-scale separation failed'):
        reduced_system, collapsed_system = reducible_system.solve_timescale_separation([A, C])
        assert reduced_system.f == [0, -C*k3]
        assert collapsed_system.x == [B, D]
    
    # Test BCD reduction (success case)
    reduced_system, collapsed_system = reducible_system.solve_timescale_separation([B, C, D])
    assert reduced_system.f == [0, -k3 * C, k3 * C]
    assert reduced_system.x == [B, C, D]
    assert collapsed_system.x == [A]
    
    # Test ACD reduction (success case)
    reduced_system, collapsed_system = reducible_system.solve_timescale_separation([A, C, D])
    assert reduced_system.f == [0, -k3 * C, k3 * C]
    assert reduced_system.x == [A, C, D]
    assert collapsed_system.x == [B]
    
    # Test CD reduction (success case)
    reduced_system, collapsed_system = reducible_system.solve_timescale_separation([C, D])
    assert reduced_system.f == [-k3 * C, k3 * C]
    assert reduced_system.x == [C, D]
    assert collapsed_system.x == [A, B]
    
    # Test ABD reduction (success case)
    answer_ABD = [-A**2*B*k1*k3/(k2 + k3),
                  -A**2*B*k1*k3/(k2 + k3),
                  A**2*B*k1*k3/(k2 + k3)]
    reduced_system, collapsed_system = reducible_system.solve_timescale_separation([A, B, D])
    assert reduced_system.f == answer_ABD
    assert reduced_system.x == [A, B, D]
    assert collapsed_system.x == [C]