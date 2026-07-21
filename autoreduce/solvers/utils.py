"""Convenience functions for AutoReduce solvers."""

import numpy as np


def get_ODE(system_obj, timepoints, **kwargs):
    """Create an ODE solver for a system."""
    from autoreduce.solvers.ode import ODE

    return ODE(
        system_obj.x,
        system_obj.f,
        C=system_obj.C,
        g=system_obj.g,
        h=system_obj.h,
        u=system_obj.u,
        params=system_obj.params,
        params_values=system_obj.params_values,
        x_init=system_obj.x_init,
        timepoints=timepoints,
        **kwargs,
    )


def solve_ode(system_obj, timepoints, **kwargs):
    """Solve a system over the given timepoints."""
    return get_ODE(system_obj, timepoints).solve_system(**kwargs)


def solve_ODE_SSM(system_obj, timepoints_ode, timepoints_ssm, **kwargs):
    """Return state, output, and local sensitivity solutions."""
    ode = get_ODE(system_obj, timepoints_ode)
    x_sol = ode.solve_system().T
    y = system_obj.C @ x_sol
    sensitivities = solve_sensitivity(system_obj, timepoints_ssm, **kwargs)
    return x_sol, y, sensitivities


def get_SSM(system_obj, timepoints, **kwargs):
    """Create an SSM solver for a system."""
    from autoreduce.solvers.ssm import SSM

    return SSM(
        system_obj.x,
        system_obj.f,
        g=system_obj.g,
        C=system_obj.C,
        h=system_obj.h,
        u=system_obj.u,
        params=system_obj.params,
        params_values=system_obj.params_values,
        x_init=system_obj.x_init,
        timepoints=timepoints,
        **kwargs,
    )


def solve_sensitivity(system_obj, timepoints, normalize=False, **kwargs):
    """Solve the coupled state and local sensitivity equations."""
    from scipy.integrate import odeint
    from sympy import Matrix, lambdify

    x_symbols = list(system_obj.x)
    param_symbols = [] if system_obj.params is None else list(system_obj.params)
    n_states = len(x_symbols)
    n_params = len(param_symbols)
    if n_params == 0:
        return np.zeros((len(timepoints), 0, n_states))

    f_matrix = Matrix(system_obj.f)
    jacobian = f_matrix.jacobian(x_symbols)
    parameter_jacobian = f_matrix.jacobian(param_symbols)
    rhs = lambdify(
        (x_symbols, param_symbols),
        (f_matrix, jacobian, parameter_jacobian),
        modules="numpy",
    )

    params_values = list(system_obj.params_values)
    sensitivity_init = np.zeros((n_states, n_params))
    if getattr(system_obj, "parameter_dependent_ic", False):
        ic_parameters = getattr(system_obj, "ic_parameters", None)
        if ic_parameters is not None:
            for state_index, ic_parameter in enumerate(ic_parameters):
                if ic_parameter in param_symbols:
                    param_index = param_symbols.index(ic_parameter)
                    sensitivity_init[state_index, param_index] = 1.0

    y0 = np.concatenate(
        [
            np.asarray(system_obj.x_init, dtype=float),
            sensitivity_init.reshape(n_states * n_params),
        ]
    )

    def extended_rhs(t, y):
        state = y[:n_states]
        sensitivity = y[n_states:].reshape((n_states, n_params))
        f_eval, jacobian_eval, parameter_jacobian_eval = rhs(
            state, params_values
        )
        f_eval = np.asarray(f_eval, dtype=float).reshape(n_states)
        jacobian_eval = np.asarray(jacobian_eval, dtype=float).reshape(
            n_states, n_states
        )
        parameter_jacobian_eval = np.asarray(
            parameter_jacobian_eval, dtype=float
        ).reshape(n_states, n_params)
        sensitivity_rhs = jacobian_eval @ sensitivity + parameter_jacobian_eval
        return np.concatenate(
            [f_eval, sensitivity_rhs.reshape(n_states * n_params)]
        )

    solution = odeint(extended_rhs, y0, timepoints, tfirst=True, **kwargs)
    state_solution = solution[:, :n_states]
    sensitivities = solution[:, n_states:].reshape(
        len(timepoints), n_states, n_params
    )
    sensitivities = np.transpose(sensitivities, (0, 2, 1))

    if normalize:
        normalized = np.zeros_like(sensitivities)
        for param_index, param_value in enumerate(params_values):
            numerator = sensitivities[:, param_index, :] * param_value
            normalized[:, param_index, :] = np.divide(
                numerator,
                state_solution,
                out=np.zeros_like(numerator),
                where=state_solution != 0,
            )
        return normalized
    return sensitivities


def solve_ssm(system_obj, timepoints, normalize=False, **kwargs):
    """Compute the local sensitivity matrix for a system."""
    return solve_sensitivity(
        system_obj, timepoints, normalize=normalize, **kwargs
    )


def get_ode_solutions(system_obj, timepoints, **kwargs):
    """Solve a system over time and return transposed state trajectories."""
    return get_ODE(system_obj, timepoints, **kwargs).solve_system().T


def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="#"
):
    """
    Print a terminal progress bar.

    Parameters
    ----------
    iteration
        Current iteration.
    total
        Total number of iterations.
    prefix
        Text printed before the bar.
    suffix
        Text printed after the percentage.
    decimals
        Number of decimal places in the percentage.
    length
        Character length of the bar.
    fill
        Character used for the filled part of the bar.
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    if iteration == total:
        print()

