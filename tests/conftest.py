from sympy import Symbol  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
from pathlib import Path

from autoreduce.system import System
from autoreduce.utils import get_reducible
from autoreduce.converters import load_sbml


@pytest.fixture
def system_1():
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
    system_1 = System(
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
    return system_1


@pytest.fixture
def reducible_system_1(system_1):
    """
    Pytest fixture that returns a reducible system
    """
    reducible_system_1 = get_reducible(system_1)
    return reducible_system_1


@pytest.fixture
def system_2():
    """
    This method gets executed before every test. It sets up a test CRN
    using BioCRNpyler.
    """
    test_dir = Path(__file__).parent
    sbml_file = test_dir / "models" / "example_1.xml"
    system_2 = load_sbml(str(sbml_file))
    return system_2
