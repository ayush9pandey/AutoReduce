"""Convenience constructors for solvers and reducible systems."""


def get_ODE(system_obj, timepoints, **kwargs):
    """
    For the given timepoints,
    create an ODE class object for this System object.
    """
    from autoreduce.solvers.ode import ODE

    ode_obj = ODE(
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
    return ode_obj


def solve_ode(system_obj, timepoints, **kwargs):
    """Solve a `System` over the given timepoints.

    This is the direct user-facing wrapper around the `ODE` solver. Use
    `get_ODE` only when the solver object itself is needed.
    """
    return get_ODE(system_obj, timepoints).solve_system(**kwargs)


def solve_ODE_SSM(system_obj, timepoints_ode, timepoints_ssm, **kwargs):
    """
    For the given timepoints,
    returns the full solution
    (states, sensitivity coefficients, outputs)
    """
    ode = get_ODE(system_obj, timepoints_ode)
    x_sol = ode.solve_system().T
    y = system_obj.C @ x_sol
    Ss = solve_sensitivity(system_obj, timepoints_ssm)
    return x_sol, y, Ss


def get_SSM(system_obj, timepoints, **kwargs):
    """
    For the given timepoints,
    create an SSM class object for this System object.
    """
    from autoreduce.solvers.ssm import SSM

    ssm_obj = SSM(
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
    return ssm_obj


def solve_sensitivity(system_obj, timepoints, normalize=False, **kwargs):
    """Solve the coupled state and local sensitivity equations.

    Returns an array indexed by time point, parameter, and state. This direct
    path builds symbolic Jacobians and solves
    ``dS_j/dt = J(x, p) S_j + df/dp_j`` alongside the original ODE.
    """
    import numpy as np
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
    """Compute the local sensitivity matrix for a `System`."""
    return solve_sensitivity(
        system_obj, timepoints, normalize=normalize, **kwargs
    )


def get_reducible(
    system_obj, timepoints_ode=None, timepoints_ssm=None, **kwargs
):
    """Create a `Reduce` object from a base `System`."""
    from autoreduce.reductions.timescale import Reduce

    red_obj = Reduce(
        system_obj.x,
        system_obj.f,
        C=system_obj.C,
        params=system_obj.params,
        g=system_obj.g,
        h=system_obj.h,
        u=system_obj.u,
        params_values=system_obj.params_values,
        x_init=system_obj.x_init,
        input_values=getattr(system_obj, "input_values", None),
        parameter_dependent_ic=getattr(
            system_obj, "parameter_dependent_ic", False
        ),
        ic_parameters=getattr(system_obj, "ic_parameters", None),
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        **kwargs,
    )
    return red_obj


def solve_timescale_separation(
    system_obj,
    slow_states,
    fast_states=None,
    timepoints_ode=None,
    timepoints_ssm=None,
    **kwargs,
):
    """Solve time-scale separation directly from a `System`."""
    from autoreduce.reductions.timescale import solve_timescale_separation

    return solve_timescale_separation(
        system_obj,
        slow_states,
        fast_states=fast_states,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        **kwargs,
    )


def reduce_utils(reduce_obj, **kwargs):
    """Create a `ReduceUtils` object from a `Reduce` object."""
    from autoreduce.reductions.timescale import ReduceUtils

    reduce_utils_obj = ReduceUtils(
        reduce_obj.x,
        reduce_obj.f,
        C=reduce_obj.C,
        params=reduce_obj.params,
        g=reduce_obj.g,
        h=reduce_obj.h,
        u=reduce_obj.u,
        params_values=reduce_obj.params_values,
        x_init=reduce_obj.x_init,
        input_values=getattr(reduce_obj, "input_values", None),
        timepoints_ode=reduce_obj.timepoints_ode,
        timepoints_ssm=reduce_obj.timepoints_ssm,
        error_tol=reduce_obj.error_tol,
        nstates_tol=reduce_obj.nstates_tol,
    )
    return reduce_utils_obj


def get_ode_solutions(system_obj, timepoints, **kwargs):
    """Solve a system over time and return transposed state trajectories."""
    x_sol = get_ODE(system_obj, timepoints, **kwargs).solve_system().T
    return x_sol


def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="#"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals
                                  in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()
