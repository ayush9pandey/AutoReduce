
#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

import libsbml # type: ignore
import warnings
import pytest # type: ignore

@pytest.fixture
def system_1_setup():
    system1 = None
    return system1

@pytest.fixture
def system_2_setup():
    system2 = None
    return system2

def test_system_equality(system1 = None, system2 = None):
    """
    Test all properties of two systems for equality
    """
    if system1 is None and system2 is None:
        return 
    elif system1 is None or system2 is None:
        warnings.warn('One of the System objects is None.')
    else:
        test_states()
        test_f()
        test_g()
        test_h()
        test_params()
        test_initial_conditions()
        test_params_values()
        test_C()

def test_states(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert system1.x == system2.x

def test_f(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert system1.f == system2.f

def test_g(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert system1.g == system2.g

def test_h(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert system1.h == system2.h

def test_params(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert system1.params == system2.params

def test_params_values(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        assert (system1.params_values, system2.params_values)

def test_initial_conditions(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        if system1.x_init is not None and system2.x_init is not None:
            assert system1.x_init == system2.x_init

def test_C(system1=None, system2=None):
    if system1 is not None and system2 is not None:
        if system1.C is not None and system2.C is not None:
            assert system1.C == system2.C
