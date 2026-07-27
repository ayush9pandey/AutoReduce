import numpy as np  # type: ignore

from autoreduce.solvers.ode import ODE
from autoreduce.solvers.ssm import SSM
from autoreduce.solvers.utils import solve_sensitivity
from autoreduce.system.system import System
from autoreduce.utils.converters import load_ODE_model


def test_solver_objects_from_symbolic_model():
    """Build ODE and SSM solver objects from a symbolic system."""
    x, f, params = load_ODE_model(2, 2)
    f[0] = -(x[0] ** 2) + params[0] * x[1]
    f[1] = -params[1] * x[1]
    output_matrix = np.array([[0, 1]]).tolist()
    system = System(x, f, params=params, C=output_matrix)

    timepoints = np.linspace(0, 4, 3)
    params_values = [2, 4]
    x_init = [0, 10]

    ode_solver = ODE(
        system.x,
        system.f,
        params=system.params,
        params_values=params_values,
        C=system.C,
        x_init=x_init,
        timepoints=timepoints,
    )
    ssm_solver = SSM(
        system.x,
        system.f,
        params=system.params,
        params_values=params_values,
        C=system.C,
        x_init=x_init,
        timepoints=timepoints,
    )

    assert isinstance(ode_solver.solve_system(), np.ndarray)
    assert isinstance(ssm_solver.compute_J([2, 1]), np.ndarray)
    assert isinstance(ssm_solver.compute_Zj([2, 1], 1), np.ndarray)


def test_direct_sensitivity_solver_matches_linear_analytic_solution():
    """Solve sensitivity ODEs directly for a one-state decay model."""
    x, f, params = load_ODE_model(1, 1)
    k = params[0]
    f[0] = -k * x[0]
    system = System(
        x,
        f,
        params=params,
        params_values=[0.4],
        x_init=[5.0],
        C=np.array([[1.0]]),
    )

    timepoints = np.linspace(0.0, 5.0, 101)
    sensitivities = solve_sensitivity(system, timepoints)
    analytical = -timepoints * 5.0 * np.exp(-0.4 * timepoints)

    np.testing.assert_allclose(sensitivities[:, 0, 0], analytical, atol=1e-6)
