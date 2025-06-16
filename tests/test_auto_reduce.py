#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from sympy import Symbol  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore

from autoreduce.system import System
from autoreduce.utils import get_reducible
from autoreduce.model_reduction import Reduce
import libsbml  # type: ignore
import warnings


@pytest.fixture
def system():
    """
    This method gets executed before every test. It sets up a test CRN:
    2A + B (k1)<->(k2) C, C (k3)--> D
    """
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    D = Symbol("D")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    k3 = Symbol("k3")

    params = [k1, k2, k3]
    # States:
    x = [A, B, C, D]

    # ODE in Sympy for the given test CRN with mass-action kinetics
    f = [
        -k1 * A**2 * B + k2 * C,
        -k1 * A**2 * B + k2 * C,
        k1 * A**2 * B - k2 * C - k3 * C,
        k3 * C,
    ]
    init_cond = np.ones(len(x))
    C = None
    g = None
    h = None
    u = None
    input_values = None
    params_values = [2, 4, 6]
    system = System(
        x,
        f,
        params=params,
        x_init=init_cond,
        params_values=params_values,
        C=C,
        g=g,
        h=h,
        u=u,
        input_values=input_values,
    )
    return system


@pytest.fixture
def reducible_system(system):
    """
    Pytest fixture that returns a reducible system
    """
    reducible_system = get_reducible(system)
    return reducible_system


def test_get_reduced_model(reducible_system):
    """
    This function creates a reducible System object
    that can be used to create reduced models given
    the x_hat (the list of states in reduced model).
    All other states are collapsed to be at quasi-steady
    state and both the reduced and the collapsed models
    are returned.
    """
    x_hat = []
    assert isinstance(reducible_system, System)
    reduced_system, collapsed_system = (
        reducible_system.solve_timescale_separation(x_hat)
    )
    if reduced_system is not None:
        assert isinstance(reduced_system, System)
    if collapsed_system is not None:
        assert isinstance(collapsed_system, System)
