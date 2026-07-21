"""Symbolic helper functions for reduction algorithms."""

import warnings

import sympy
from sympy import Eq, solve

__all__ = [
    "sympy_get_steady_state_solutions",
    "sympy_solve_and_substitute",
    "sympy_variables_exist",
]


def sympy_variables_exist(ode_function, variables_to_check, **kwargs):
    """Check whether variables appear in symbolic ODE expressions."""
    flag = False
    all_free_symbols = []
    debug = kwargs.get("debug", False)
    if debug:
        print(
            "In sympy_variables_exist. Checking for presence of ",
            variables_to_check,
        )
    for fi in ode_function:
        fi = sympy.sympify(fi)
        for sym in fi.free_symbols:
            if sym not in all_free_symbols:
                all_free_symbols.append(sym)

    variables_that_appear = []
    for sym in all_free_symbols:
        if sym in variables_to_check:
            variables_that_appear.append(sym)
            flag = True
    if flag and debug:
        print("Found! The following: ", variables_that_appear)
    return flag, variables_that_appear


def sympy_solve_and_substitute(
    ode_function,
    collapsed_states,
    collapsed_dynamics,
    solution_dict,
    debug=False,
):
    """Solve collapsed-state equations and substitute into ODEs."""
    for state in collapsed_states:
        index = collapsed_states.index(state)
        dynamics = collapsed_dynamics[index]
        if debug:
            print("In sympy_solve_and_substitute, solving for ", state)
            print("From ", dynamics)
        solution_dict = sympy_get_steady_state_solutions(
            collapsed_variables=[state],
            collapsed_dynamics=[dynamics],
            solution_dict=solution_dict,
            debug=debug,
        )
        if debug:
            print("Solution found: ", solution_dict)
            print("current state", state)
        if solution_dict[state] is None or len(solution_dict[state]) == 0:
            continue
        for func in ode_function:
            func = sympy.sympify(func)
            func_index = ode_function.index(func)
            ode_function[func_index] = func.subs(
                state, solution_dict[state][0]
            )
        for func in collapsed_dynamics:
            if func == dynamics:
                continue
            func_index = collapsed_dynamics.index(func)
            collapsed_dynamics[func_index] = func.subs(
                state, solution_dict[state][0]
            )
        if debug:
            print("Updated f_hat now is ", ode_function)
    return ode_function, solution_dict, collapsed_dynamics


def sympy_get_steady_state_solutions(
    collapsed_variables, collapsed_dynamics, solution_dict=None, debug=False
):
    """
    Solve each collapsed variable from its steady-state equation.

    Returns a dictionary mapping each collapsed variable to the SymPy
    solutions found for that variable.
    """
    if solution_dict is None:
        solution_dict = {}
    x_c = collapsed_variables
    f_c = collapsed_dynamics
    for i, _ in enumerate(x_c):
        x_c_sub = solve(Eq(f_c[i], 0), x_c[i])
        if x_c_sub is None or len(x_c_sub) == 0:
            print(f"Could not find solution for: {x_c[i]} from {f_c[i]}")
            warnings.warn(
                "Solve time-scale separation failed. Check model consistency."
            )
        elif len(x_c_sub) > 1:
            if debug:
                print(f"Multiple solutions obtained for {x_c[i]}.")
                print("Choosing one non-zero solution, check consistency. ")
                print(f"The solutions are {x_c_sub}.")
                print(" Highly recommend manually solving for this")
                print(" variable first then try this function.")
            for sub in x_c_sub:
                if sub == 0:
                    x_c_sub.remove(0)
        elif not any(x_c_sub):
            warnings.warn(
                "Solve time-scale separation failed. Check model consistency."
            )
            if debug:
                warnings.warn(f"Zero solution(s) for: {x_c[i]} from {f_c[i]}.")
        solution_dict[x_c[i]] = x_c_sub
    return solution_dict

