import numpy as np
import pytest
from sympy import Symbol


def test_control_adapter_when_dependency_is_installed():
    """Convert a python-control nonlinear IO system when control is present."""
    control = pytest.importorskip("control")

    from autoreduce.system.control import from_nonlinear_io_system
    from autoreduce.system.system import System

    def update(_t, x, _u, params):
        return [-params["k"] * x[0]]

    io_system = control.NonlinearIOSystem(update, None, states=1)
    x = [Symbol("x")]
    k = Symbol("k")
    system = from_nonlinear_io_system(
        io_system,
        x,
        params=[k],
        params_values=[2.0],
        x_init=[1.0],
    )

    assert isinstance(system, System)
    assert system.f == [-k * x[0]]


def test_dmd_projection_when_dependency_is_installed():
    """Fit a PyDMD model when PyDMD is present."""
    pytest.importorskip("pydmd")

    from autoreduce.reductions.projection.dmd import fit_dmd

    time = np.linspace(0.0, 1.0, 8)
    snapshots = np.vstack([np.exp(-time), np.exp(-2.0 * time)])
    reduction = fit_dmd(snapshots, svd_rank=1)

    assert reduction.snapshots.shape == snapshots.shape
    assert reduction.reduced_dimension == 1


def test_pydmd_linear_operator_adapter():
    """Convert a DMD-style linear operator into a symbolic System."""
    from autoreduce.system.pydmd import from_linear_operator
    from autoreduce.system.system import System

    x0 = Symbol("x0")
    x1 = Symbol("x1")
    system = from_linear_operator(
        np.array([[0, 1], [-2, -3]]),
        [x0, x1],
        discrete_time=False,
    )

    assert isinstance(system, System)
    assert system.x == [x0, x1]
    assert system.f == [x1, -2 * x0 - 3 * x1]
