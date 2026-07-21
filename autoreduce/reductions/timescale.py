"""Time-scale separation reduction methods."""

import warnings

import numpy as np
import sympy
from sympy import Eq, solve

from autoreduce.reductions.core import _as_reducible, create_system
from autoreduce.reductions.utils import (
    sympy_solve_and_substitute,
    sympy_variables_exist,
)

__all__ = [
    "explore_all_QSS_models",
    "reduce_with_input",
    "solve_timescale_separation",
    "solve_timescale_separation_with_input",
]


def solve_timescale_separation(
    system_obj,
    slow_states,
    fast_states=None,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
    **kwargs,
):
    """Solve a time-scale separation reduction for a system."""
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
    )
    return _solve_timescale_separation(
        reducible_system,
        slow_states,
        fast_states=fast_states,
        **kwargs,
    )


def _solve_timescale_separation(
    reducible_system, slow_states, fast_states=None, **kwargs
):
    debug = kwargs.get("debug", False)
    if not slow_states:
        return reducible_system.get_system(), None
    x, f, x_init = (
        reducible_system.x,
        reducible_system.f,
        reducible_system.x_init,
    )

    len_slow_states = len(slow_states)
    x_hat_init = [None] * len_slow_states
    f_hat = [None] * len_slow_states

    max_len_fast_states = len(x) - len(slow_states)
    x_c = [None] * max_len_fast_states
    x_c_init = [None] * max_len_fast_states
    f_c = [None] * max_len_fast_states
    x_hat = slow_states
    if fast_states:
        x_c = fast_states
    else:
        count_x_c = 0
        for state in x:
            if state not in slow_states:
                x_c[count_x_c] = state
                count_x_c += 1
        fast_states = x_c

    if len(slow_states) + len(fast_states) != len(reducible_system.x):
        raise RuntimeError(
            "Number of slow states plus number of fast states must equal "
            "the number of total states."
        )
    for state in slow_states:
        if state in fast_states:
            raise RuntimeError(
                "Found a state that is both fast and slow. Pass disjoint "
                "slow_states and fast_states."
            )

    for state in x:
        state_index = x.index(state)
        if state in x_c:
            x_c_index = x_c.index(state)
            f_c[x_c_index] = f[state_index]
            if reducible_system.parameter_dependent_ic:
                param_as_ic = reducible_system.ic_parameters[state_index]
                reducible_system.set_ic_from_params(
                    x_c_init, param_as_ic, x_c_index
                )
            else:
                x_c_init[x_c_index] = x_init[state_index]
        if state in x_hat:
            x_hat_index = x_hat.index(state)
            f_hat[x_hat_index] = f[state_index]
            if reducible_system.parameter_dependent_ic:
                param_as_ic = reducible_system.ic_parameters[state_index]
                reducible_system.set_ic_from_params(
                    x_hat_init, param_as_ic, x_hat_index
                )
            else:
                x_hat_init[x_hat_index] = x_init[state_index]
    reducible_system.f_hat = f_hat
    reducible_system.f_c = f_c
    if debug:
        print("Reduced set of variables is", x_hat)
        print("f_hat = ", reducible_system.f_hat)
        print("Collapsed set of variables is", x_c)

    loop_sanity = True
    count = 0
    solution_dict = {}
    while (
        sympy_variables_exist(
            ode_function=reducible_system.f_hat, variables_to_check=x_c
        )[0]
        and loop_sanity
    ):
        (
            reducible_system.f_hat,
            solution_dict,
            reducible_system.f_c,
        ) = sympy_solve_and_substitute(
            ode_function=reducible_system.f_hat,
            collapsed_states=x_c,
            collapsed_dynamics=reducible_system.f_c,
            solution_dict=solution_dict,
            debug=debug,
        )
        if count > 2:
            warnings.warn(
                "Solve time-scale separation failed. Check model consistency."
            )
            print(
                f"Did not work to retain: {slow_states} because either a "
                "collapsed state variable appears"
            )
            print(" in the reduced model or a solution is not possible.")
            loop_sanity = False
            return None, None
        count += 1

    for i, _ in enumerate(x_hat):
        for j, _ in enumerate(reducible_system.f_c):
            reducible_system.f_c[j] = reducible_system.f_c[j].subs(
                x_hat[i], x_hat_init[i]
            )

    C_hat = reducible_system.create_C_hat(x_hat)
    for index, _ in enumerate(f_hat):
        f_hat[index] = sympy.simplify(f_hat[index])
    for index, _ in enumerate(f_c):
        f_c[index] = sympy.simplify(f_c[index])
    reduced_sys = create_system(
        x_hat,
        reducible_system.f_hat,
        params=reducible_system.params,
        C=C_hat,
        params_values=reducible_system.params_values,
        x_init=x_hat_init,
        timepoints_ode=reducible_system.timepoints_ode,
        timepoints_ssm=reducible_system.timepoints_ssm,
    )
    fast_subsystem = create_system(
        x_c,
        reducible_system.f_c,
        params=reducible_system.params,
        params_values=reducible_system.params_values,
        x_init=x_c_init,
        timepoints_ode=reducible_system.timepoints_ode,
        timepoints_ssm=reducible_system.timepoints_ssm,
    )
    reduced_sys.fast_states = fast_states
    print(f"Successful solution obtained with states: {reduced_sys.x}!")
    return reduced_sys, fast_subsystem


def solve_timescale_separation_with_input(
    system_obj,
    attempt_states,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
):
    """Solve time-scale separation for a system with explicit inputs."""
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
    )
    return _solve_timescale_separation_with_input(
        reducible_system, attempt_states
    )


def _solve_timescale_separation_with_input(reducible_system, attempt_states):
    attempt = []
    for state in attempt_states:
        attempt.append(reducible_system.x.index(state))
    print("attempting to retain:", attempt)
    x_c = []
    fast_states = []
    f_c = []
    f_hat = []
    x_hat_init = []
    x_c_init = []
    x_hat = []
    x, f, g, u, x_init = (
        reducible_system.x,
        reducible_system.f,
        reducible_system.g,
        reducible_system.u,
        reducible_system.x_init,
    )
    f_g = [fi + gi for fi, gi in zip(f, g)]
    for i in range(reducible_system.n):
        if i not in attempt:
            x_c.append(x[i])
            f_c.append(f_g[i])
            x_c_init.append(x_init[i])
        else:
            f_hat.append(f_g[i])
            x_hat.append(x[i])
            x_hat_init.append(x_init[i])

    solved_states = []
    lookup_collapsed = {}
    for i, _ in enumerate(x_c):
        x_c_sub = solve(Eq(f_c[i], 0), x_c[i])
        lookup_collapsed[x_c[i]] = x_c_sub
        if len(x_c_sub) == 0:
            fast_states.append([])
            continue
        elif len(x_c_sub) > 1:
            for sub in x_c_sub:
                if sub == 0:
                    x_c_sub.remove(0)
        else:
            for sym in x_c_sub[0].free_symbols:
                if sym in solved_states and sym in x:
                    f_c[i] = f_c[i].subs(sym, lookup_collapsed[sym][0])
                    x_c_sub = solve(Eq(f_c[i], 0), x_c[i])
                    if len(x_c_sub) > 1:
                        print("Multiple solutions obtained.")
                        print("Choosing non-zero solution, check consistency.")
                        print(" The solutions are ", x_c_sub)
                        for sub in x_c_sub:
                            if sub == 0:
                                x_c_sub.remove(0)
                    lookup_collapsed[x_c[i]] = x_c_sub
                else:
                    solved_states.append(x_c[i])
            fast_states.append(x_c_sub[0])

    for i in range(len(fast_states)):
        if fast_states[i] == []:
            continue
        for j in range(len(f_hat)):
            f_hat[j] = f_hat[j].subs(x_c[i], fast_states[i])
        for j in range(len(f_c)):
            f_c[j] = f_c[j].subs(x_c[i], fast_states[i])

    for i in range(len(x_hat)):
        for j in range(len(f_c)):
            f_c[j] = f_c[j].subs(x_hat[i], x_hat_init[i])

    output_states = reducible_system.get_output_states()
    C_hat = np.zeros((np.shape(reducible_system.C)[0], np.shape(x_hat)[0]))
    is_output = 0
    for i in range(len(x_hat)):
        if x_hat[i] in output_states:
            is_output = 1
        for row_ind in range(np.shape(C_hat)[0]):
            C_hat[row_ind][i] = 1 * is_output

    flag = False
    free_symbols_all = []
    for fi in f_hat:
        fi = sympy.sympify(fi)
        for sym in fi.free_symbols:
            if sym not in free_symbols_all:
                free_symbols_all.append(sym)
    bugged_states = []
    for syms in free_symbols_all:
        if syms not in x_hat + u + reducible_system.params:
            bugged_states.append(syms)
            flag = True
    if flag:
        warnings.warn("Check model consistency")
        print(
            f"The time-scale separation that retains states {attempt}, "
            "does not work"
        )
        print(
            f"because the state variables {bugged_states} appear in the "
            "reduced model"
        )

    reduced_sys = create_system(
        x_hat,
        f_hat,
        params=reducible_system.params,
        C=C_hat,
        params_values=reducible_system.params_values,
        x_init=x_hat_init,
        timepoints_ode=reducible_system.timepoints_ode,
        timepoints_ssm=reducible_system.timepoints_ssm,
    )
    fast_subsystem = create_system(
        x_c,
        f_c,
        params=reducible_system.params,
        params_values=reducible_system.params_values,
        x_init=x_c_init,
        timepoints_ode=reducible_system.timepoints_ode,
        timepoints_ssm=reducible_system.timepoints_ssm,
    )
    reduced_sys.x_c = x_c
    reduced_sys.bugged_states = bugged_states
    reduced_sys.fast_states = fast_states
    return reduced_sys, fast_subsystem


def explore_all_QSS_models(
    system_obj,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
    **kwargs,
):
    """Explore candidate QSS reductions for an autonomous system."""
    constructor_kwargs = {}
    for option in ("error_tol", "nstates_tol", "nstates_tol_min"):
        if option in kwargs:
            constructor_kwargs[option] = kwargs.pop(option)
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
        **constructor_kwargs,
    )
    skip_numerical_computations = kwargs.get(
        "skip_numerical_computations", False
    )
    skip_error_computation = kwargs.get("skip_error_computation", False)
    skip_robustness_computation = kwargs.get(
        "skip_robustness_computation", False
    )
    if reducible_system.u is not None:
        raise ValueError("For models with inputs use reduce_with_input.")
    results_dict = {}
    possible_reductions = reducible_system.get_all_combinations()
    if not len(possible_reductions):
        print("No possible reduced models found.")
        print(" Try increasing tolerance for number of states.")
        return
    for attempt in possible_reductions:
        if len(attempt) < reducible_system.nstates_tol_min:
            continue
        elif len(attempt) > reducible_system.nstates_tol:
            continue
        attempt_states = [reducible_system.x[i] for i in attempt]
        reduced_sys, fast_subsystem = _solve_timescale_separation(
            reducible_system, attempt_states, **kwargs
        )
        if reduced_sys is None or fast_subsystem is None:
            continue
        if skip_numerical_computations:
            results_dict[reduced_sys] = None
        else:
            if skip_error_computation:
                e = np.nan
            else:
                e = reducible_system.get_error_metric(reduced_sys)
            if skip_robustness_computation:
                Se = np.nan
                R = np.nan
            else:
                Se, R = reducible_system.get_robustness_metric(
                    reduced_sys, **kwargs
                )
            results_dict[reduced_sys] = [e, Se, R]
    reducible_system.results_dict = results_dict
    return reducible_system.results_dict


def reduce_with_input(
    system_obj,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
    **kwargs,
):
    """Compute candidate QSS reductions for systems with explicit inputs."""
    constructor_kwargs = {}
    for option in ("error_tol", "nstates_tol", "nstates_tol_min"):
        if option in kwargs:
            constructor_kwargs[option] = kwargs.pop(option)
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
        **constructor_kwargs,
    )
    if reducible_system.u is None:
        raise ValueError("For models with no inputs use explore_all_QSS_models.")
    results_dict = {}
    possible_reductions = reducible_system.get_all_combinations()
    if not len(possible_reductions):
        print("No possible reduced models found.")
        print(" Try increasing tolerance for number of states.")
        return
    for attempt in possible_reductions:
        attempt_states = [reducible_system.x[i] for i in attempt]
        reduced_sys, fast_subsystem = _solve_timescale_separation_with_input(
            reducible_system, attempt_states
        )
        if reduced_sys is None or fast_subsystem is None:
            continue
        e = reducible_system.get_error_metric_with_input(reduced_sys)
        Se, R = reducible_system.get_robustness_metric_with_input(reduced_sys)
        results_dict[reduced_sys] = [e, Se, R]
    reducible_system.results_dict = results_dict
    return reducible_system.results_dict

