"""Core reduction objects and shared reduction utilities."""

import numpy as np
from scipy.linalg import block_diag, eigvals, norm, solve_lyapunov

from autoreduce.solvers import utils as solver_utils
from autoreduce.system.system import System

__all__ = [
    "Reduce",
    "ReduceUtils",
    "create_system",
    "get_error_metric",
    "get_reducible",
    "get_robustness_metric",
]


class Reduce(System):
    """
    Working system used to build and compare reduced models.

    The object stores reduction tolerances, candidate-reduction results, and
    shared routines used by the specific reduction algorithms.
    """

    def __init__(
        self,
        x,
        f,
        params=None,
        C=None,
        g=None,
        h=None,
        u=None,
        params_values=None,
        x_init=None,
        input_values=None,
        timepoints_ode=None,
        timepoints_ssm=None,
        error_tol=None,
        nstates_tol=None,
        nstates_tol_min=None,
        **kwargs,
    ):
        super().__init__(
            x,
            f,
            params,
            C,
            g,
            h,
            u,
            params_values,
            x_init,
            input_values,
            timepoints_ode=timepoints_ode,
            timepoints_ssm=timepoints_ssm,
            **kwargs,
        )
        self.f_hat = []
        self.nstates_tol = self.n - 1 if nstates_tol is None else nstates_tol
        self.nstates_tol_min = (
            1 if nstates_tol_min is None else nstates_tol_min
        )
        self.error_tol = 1e6 if error_tol is None else error_tol
        self.results_dict = {}
        self.x_c = []

    def get_output_states(self):
        """Return state symbols that appear in the system output."""
        if self.C is None and self.h is None:
            return []
        outputs = np.dot(np.array(self.C), np.array(self.x))
        if type(outputs) is not np.ndarray:
            outputs = [outputs]
        else:
            outputs = outputs.tolist()
        output_symbols = [list(i.free_symbols) for i in outputs]
        return [item for sublist in output_symbols for item in sublist]

    def get_all_combinations(self):
        """
        Return retained-state index sets allowed by the current tolerances.

        Candidate sets that exclude output states or exceed `nstates_tol` are
        removed before returning.
        """
        from itertools import combinations

        possible_reductions = []
        n = self.n
        for i in range(n):
            if i != n - 1:
                comb = combinations(list(range(n)), i + 1)
                possible_reductions.append(list(comb))
        possible_reductions = [
            list(item) for sublist in possible_reductions for item in sublist
        ]
        self.all_combinations = [i for i in possible_reductions]
        output_states = self.get_output_states()
        restart = False
        x = self.x
        for attempt in self.all_combinations:
            states_attempt = [x[i] for i in attempt]
            if (
                not len(set(states_attempt).intersection(set(output_states)))
                == len(output_states)
                or len(attempt) > self.nstates_tol
            ):
                restart = True
            if restart:
                possible_reductions.remove(attempt)
                restart = False

        if self.u is not None:
            for i, _ in enumerate(self.g):
                if self.g[i] != 0 and i in possible_reductions:
                    possible_reductions.remove(i)
        return possible_reductions

    def get_T(self, attempt):
        """Construct transformation matrices for retained and collapsed states."""
        non_attempt = [i for i in range(self.n) if i not in attempt]
        T = np.zeros((self.n, self.n))
        n_hat = len(attempt)
        n = self.n
        n_c = n - n_hat
        T1 = np.zeros((self.n, n_hat))
        T2 = np.zeros((self.n, n_c))
        for ni in range(0, n_hat):
            set_T = False
            for i in range(n):
                if i in attempt and not set_T:
                    T[ni, i] = 1
                    attempt.remove(i)
                    set_T = True
        for ni in range(n_hat, n):
            set_T = False
            for i in range(n):
                if i in non_attempt and not set_T:
                    T[ni, i] = 1
                    non_attempt.remove(i)
                    set_T = True
        T1 = T[0:n, 0:n_hat]
        T2 = T[0:n, n_hat : n + 1]  # noqa: E203
        return T, T1, T2

    def create_C_hat(self, x_hat):
        """Return the linear output matrix for a reduced state vector."""
        if self.C is None:
            return None
        output_states = self.get_output_states()
        C_hat = np.zeros((np.shape(self.C)[0], np.shape(x_hat)[0]))
        is_output = 0
        for i in range(len(x_hat)):
            if x_hat[i] in output_states:
                is_output = 1
            for row_ind in range(np.shape(C_hat)[0]):
                C_hat[row_ind][i] = 1 * is_output
        return C_hat

    def get_error_metric_with_input(self, reduced_sys):
        """Return the full-vs-reduced output error for systems with inputs."""
        if self.timepoints_ode is None:
            raise ValueError(
                "Set timepoints_ode before calling get_error_metric_with_input."
            )
        reduced_ode = solver_utils.get_ODE(reduced_sys, self.timepoints_ode)
        x_sol, _, _ = self.get_solutions()
        y = self.C @ x_sol
        x_sols_hat = reduced_ode.solve_system().T
        y_hat = np.array(reduced_sys.C) @ np.array(x_sols_hat)
        if np.shape(y) != np.shape(y_hat):
            raise ValueError(
                "The output dimensions must be the same for reduced and full "
                "model. Choose C and C_hat accordingly."
            )
        e = np.linalg.norm(y - y_hat)
        if np.isnan(e):
            print("The error is NaN, something wrong...continuing.")
        return e

    def get_error_metric(self, reduced_sys):
        """Return the full-vs-reduced output error."""
        if self.timepoints_ode is None:
            raise ValueError(
                "Set timepoints_ode before calling get_error_metric."
            )
        reduced_ode = solver_utils.get_ODE(reduced_sys, self.timepoints_ode)
        x_sol, _, _ = self.get_solutions()
        y = self.C @ x_sol
        x_sols_hat = reduced_ode.solve_system().T
        y_hat = np.array(reduced_sys.C) @ np.array(x_sols_hat)
        if np.shape(y) != np.shape(y_hat):
            raise ValueError(
                "The output dimensions must be the same for reduced and full "
                "model. Choose C and C_hat accordingly."
            )
        e = np.linalg.norm(y - y_hat)
        if np.isnan(e):
            print("The error is NaN, something wrong...continuing.")
        return e

    def get_robustness_metric_with_input(self, reduced_sys):
        """Return the robustness metric for systems with inputs."""
        return

    def get_robustness_metric(self, reduced_sys, **kwargs):
        """Compute robustness metrics comparing full and reduced systems."""
        method = kwargs.get("method", "direct")
        timepoints_ssm = self.timepoints_ssm
        if timepoints_ssm is None:
            raise ValueError(
                "Pass timepoints_ssm to get_robustness_metric to specify "
                "the timepoints for sensitivity computations."
            )
        if self.timepoints_ode is None:
            raise ValueError(
                "Set timepoints_ode before calling get_robustness_metric."
            )
        system_obj = self.get_system()
        x_sols = solver_utils.get_ODE(system_obj, timepoints_ssm).solve_system().T
        full_ssm = solver_utils.get_SSM(system_obj, timepoints_ssm)
        S = full_ssm.compute_SSM()
        self.S = S
        reduced_ssm = solver_utils.get_SSM(reduced_sys, timepoints_ssm)
        x_sols_hat = (
            solver_utils.get_ODE(reduced_sys, timepoints_ssm).solve_system().T
        )
        x_sols = np.reshape(x_sols, (len(timepoints_ssm), self.n))
        x_sols_hat = np.reshape(
            x_sols_hat, (len(timepoints_ssm), reduced_sys.n)
        )
        Se = np.zeros(len(self.params_values))
        S_hat = reduced_ssm.compute_SSM()
        reduced_sys.S = S_hat
        S_bar = np.concatenate((S, S_hat), axis=2)
        S_bar = np.reshape(
            S_bar,
            (
                len(timepoints_ssm),
                self.n + reduced_sys.n,
                len(self.params_values),
            ),
        )
        C_bar = np.concatenate((self.C, -1 * reduced_sys.C), axis=1)
        C_bar = np.reshape(
            C_bar, (np.shape(self.C)[0], (self.n + reduced_sys.n))
        )
        weighted_Se_sum = 0
        P_prev = None
        prev_time = None
        if method == "bound":
            for j, _ in enumerate(self.params_values):
                S_metric_max = 0
                sens_max = 0
                max_eig_P = 0
                max_eig_dot_P = 0
                for k in range(len(self.timepoints_ssm)):
                    curr_time = self.timepoints_ssm[k]
                    J = full_ssm.compute_J(x_sols[k, :])
                    J_hat = reduced_ssm.compute_J(x_sols_hat[k, :])
                    J_bar = block_diag(J, J_hat)
                    P = solve_lyapunov(J_bar, -1 * C_bar.T @ C_bar)
                    eig_P = max(eigvals(P))
                    if max_eig_P < eig_P:
                        max_eig_P = eig_P
                    if k != 0:
                        dot_P = (P - P_prev) / (curr_time - prev_time)
                        eig_dot_P = max(eigvals(dot_P))
                        if max_eig_dot_P < eig_dot_P:
                            max_eig_dot_P = eig_dot_P
                    Z = full_ssm.compute_Zj(x_sols[k, :], j)
                    Z_hat = reduced_ssm.compute_Zj(x_sols_hat[k, :], j)
                    Z_bar = np.concatenate((Z, Z_hat), axis=0)
                    Z_bar = np.reshape(Z_bar, ((self.n + reduced_sys.n), 1))
                    S_metric = norm(Z_bar.T @ P @ S_bar[k, :, j])
                    if S_metric > S_metric_max:
                        S_metric_max = S_metric
                    sens_norm = norm(S_bar[k, :, j]) ** 2
                    if sens_norm > sens_max:
                        sens_max = sens_norm
                    P_prev = P
                    prev_time = curr_time
                    solver_utils.printProgressBar(
                        int(j + k * len(self.params_values)),
                        len(timepoints_ssm) * len(self.params_values) - 1,
                        prefix="Robustness Metric Progress:",
                        suffix="Complete",
                        length=50,
                    )
                dot_P_term = (
                    max_eig_dot_P * len(reduced_ssm.timepoints) * sens_max
                )
                Se[j] = (
                    max_eig_P
                    + 2 * len(reduced_ssm.timepoints) * S_metric_max
                    + dot_P_term
                )
                weighted_Se_sum += self.params_values[j] * Se[j]
        elif method == "direct":
            for j in range(len(self.params_values)):
                Se[j] = norm(C_bar @ S_bar[:, :, j].T)
                weighted_Se_sum += self.params_values[j] * Se[j]
        err_norm = norm(self.get_error_metric(reduced_sys))
        R = 1 / (1 + (weighted_Se_sum / err_norm))
        reduced_sys.R = R
        reduced_sys.Se = Se
        return Se, R

    def get_solutions(self):
        """Return full-model ODE, SSM-time ODE, and SSM objects."""
        if self.timepoints_ode is None:
            raise ValueError(
                "Set timepoints_ode before calling get_solutions."
            )
        system_obj = self.get_system()
        x_sol = solver_utils.get_ode_solutions(
            system_obj, self.timepoints_ode
        )
        if self.timepoints_ssm is None:
            return x_sol, None, None
        x_sol2 = solver_utils.get_ode_solutions(
            system_obj, self.timepoints_ssm
        )
        full_ssm = solver_utils.get_SSM(system_obj, self.timepoints_ssm)
        return x_sol, x_sol2, full_ssm

    def find_conserved_sets(self, search_depth, **kwargs):
        """Find conserved species sets using the conservation module."""
        from autoreduce.reductions.conservation import find_conserved_sets

        return find_conserved_sets(self, search_depth=search_depth, **kwargs)

    def setup_conservation_laws(
        self, total_quantities: dict, conserved_sets: list
    ):
        """Create conservation-law expressions from conserved species sets."""
        from autoreduce.reductions.conservation import setup_conservation_laws

        return setup_conservation_laws(self, total_quantities, conserved_sets)

    def solve_conservation_laws(
        self,
        conservation_laws=None,
        total_quantities=None,
        conserved_sets=None,
        states_to_eliminate=None,
        search_depth=None,
        **kwargs,
    ):
        """Apply conservation laws using the conservation module."""
        from autoreduce.reductions.conservation import solve_conservation_laws

        return solve_conservation_laws(
            self,
            conservation_laws=conservation_laws,
            total_quantities=total_quantities,
            conserved_sets=conserved_sets,
            states_to_eliminate=states_to_eliminate,
            search_depth=search_depth,
            in_place=True,
            **kwargs,
        )

    def set_conservation_laws(self, conservation_laws, states_to_eliminate):
        """Apply conservation laws using the conservation module."""
        from autoreduce.reductions.conservation import apply_conservation_laws

        return apply_conservation_laws(
            self,
            conservation_laws=conservation_laws,
            states_to_eliminate=states_to_eliminate,
        )

    def solve_approximations(self):
        """Run abundance-based approximations from the abundance module."""
        from autoreduce.reductions.abundance import solve_approximations

        return solve_approximations(self)

    def solve_timescale_separation(
        self, slow_states, fast_states=None, **kwargs
    ):
        """Solve a time-scale separation reduction."""
        from autoreduce.reductions.timescale import solve_timescale_separation

        return solve_timescale_separation(
            self,
            slow_states,
            fast_states=fast_states,
            in_place=True,
            **kwargs,
        )

    def solve_timescale_separation_with_input(self, attempt_states):
        """Solve time-scale separation for systems with inputs."""
        from autoreduce.reductions.timescale import (
            solve_timescale_separation_with_input,
        )

        return solve_timescale_separation_with_input(
            self, attempt_states, in_place=True
        )

    def explore_all_QSS_models(self, **kwargs):
        """Explore candidate QSS reductions for autonomous systems."""
        from autoreduce.reductions.timescale import explore_all_QSS_models

        return explore_all_QSS_models(self, in_place=True, **kwargs)

    def reduce_with_input(self):
        """Compute candidate QSS reductions for systems with explicit inputs."""
        from autoreduce.reductions.timescale import reduce_with_input

        return reduce_with_input(self, in_place=True)

    def reduce_general(self):
        """Return the placeholder result set for the general reduction path."""
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print("No possible reduced models found.")
            print(" Try increasing tolerance for number of states.")
            return

        self.results_dict = results_dict
        return self.results_dict

    def compute_reduced_model(self):
        """Dispatch to the reduction workflow matching the system structure."""
        if self.C is not None and self.g is None:
            print("Using model reduction algorithm with y = Cx")
            print(" linear output relationship and no inputs (g = 0).")
            self.results_dict = self.explore_all_QSS_models()
            return self.results_dict
        print("Using general model reduction algorithm")
        print(" with inputs and nonlinear output relationship")
        self.results_dict = self.reduce_general()
        return self.results_dict

    def get_system(self):
        """Return the current reduction object as a base `System`."""
        return System(
            self.x,
            self.f,
            params=self.params,
            C=self.C,
            g=self.g,
            h=self.h,
            u=self.u,
            params_values=self.params_values,
            x_init=self.x_init,
            input_values=self.input_values,
            timepoints_ode=self.timepoints_ode,
            timepoints_ssm=self.timepoints_ssm,
            parameter_dependent_ic=getattr(
                self, "parameter_dependent_ic", False
            ),
            ic_parameters=getattr(self, "ic_parameters", None),
        )


class ReduceUtils(Reduce):
    """Utility methods for reduction result objects."""

    def __init__(
        self,
        x,
        f,
        params=None,
        C=None,
        g=None,
        h=None,
        u=None,
        params_values=None,
        x_init=None,
        input_values=None,
        timepoints_ode=None,
        timepoints_ssm=None,
        error_tol=None,
        nstates_tol=None,
    ):
        super().__init__(
            x=x,
            f=f,
            params=params,
            C=C,
            g=g,
            h=h,
            u=u,
            params_values=params_values,
            x_init=x_init,
            input_values=input_values,
            timepoints_ode=timepoints_ode,
            timepoints_ssm=timepoints_ssm,
            error_tol=error_tol,
            nstates_tol=nstates_tol,
        )

    def write_results(self, filename):
        """Write model reduction results to a text file."""
        from sympy.printing import latex

        with open(filename, "w") as f1:
            f1.write("Model reduction results.\n")
            for key, value in self.results_dict.items():
                f1.write("A possible reduced model: \n \n")
                f1.write("\n$x_{hat} = ")
                f1.write(str(key.x))
                f1.write("$\n\n\n\n")
                for k in range(len(key.f)):
                    f1.write("\n$f_{hat}(" + str(k + 1) + ") = ")
                    f1.write(latex(key.f[k]))
                    f1.write("$\n\n")
                f1.write("\n\n\n")
                f1.write("\nError metric:")
                f1.write(str(value[0]))
                f1.write("\n\n\n")
                f1.write("\nRobustness metric:")
                f1.write(str(value[1]))
                f1.write("\n\n\n")
                f1.write("Other properties")
                f1.write("\n\n\n")
                f1.write("\n C = ")
                f1.write(str(key.C))
                f1.write("\n$ g = ")
                f1.write(str(key.g))
                f1.write("$\n h = ")
                f1.write(str(key.h))
                if hasattr(key, "x_sol"):
                    f1.write("$\n Solutions : \n")
                    f1.write(str(key.x_sol))
                    f1.write("\n\n\n\n")
                f1.write("\n Sensitivity Solutions : \n")
                f1.write(str(key.S))
                f1.write("\n\n\n\n")

    def get_valid_reduced_models(self, nstates_tol=None, error_tol=None):
        """Return reduced models satisfying state-count and error tolerances."""
        if nstates_tol is None:
            nstates_tol = self.nstates_tol
        if error_tol is None:
            error_tol = self.error_tol
        valid_reduced_models = []
        results_dict = self.results_dict
        for key, value in results_dict.items():
            error = value[0]
            if error <= error_tol and len(key.x) <= nstates_tol:
                valid_reduced_models.append(key)
        self.valid_reduced_models = valid_reduced_models
        return valid_reduced_models


def create_system(
    x,
    f,
    params=None,
    C=None,
    g=None,
    h=None,
    u=None,
    params_values=None,
    x_init=None,
    input_values=None,
    timepoints_ode=None,
    timepoints_ssm=None,
    parameter_dependent_ic=False,
    ic_parameters=None,
):
    """Create a base `System` from symbolic model data."""
    return System(
        x,
        f=f,
        params=params,
        C=C,
        g=g,
        h=h,
        u=u,
        params_values=params_values,
        x_init=x_init,
        input_values=input_values,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        parameter_dependent_ic=parameter_dependent_ic,
        ic_parameters=ic_parameters,
    )


def _copy_model_value(value):
    """Copy model containers without deep-copying symbolic expressions."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [
            item.copy() if isinstance(item, (list, np.ndarray)) else item
            for item in value
        ]
    if isinstance(value, tuple):
        return list(value)
    return value


def get_reducible(
    system_obj, timepoints_ode=None, timepoints_ssm=None, **kwargs
):
    """Create a `Reduce` working object from a base `System`."""
    if not isinstance(system_obj, System):
        raise TypeError("system_obj must be an AutoReduce System object.")
    if timepoints_ode is None:
        timepoints_ode = getattr(system_obj, "timepoints_ode", None)
    if timepoints_ssm is None:
        timepoints_ssm = getattr(system_obj, "timepoints_ssm", None)

    return Reduce(
        _copy_model_value(system_obj.x),
        _copy_model_value(system_obj.f),
        C=_copy_model_value(system_obj.C),
        params=_copy_model_value(system_obj.params),
        g=_copy_model_value(system_obj.g),
        h=_copy_model_value(system_obj.h),
        u=_copy_model_value(system_obj.u),
        params_values=_copy_model_value(system_obj.params_values),
        x_init=_copy_model_value(system_obj.x_init),
        input_values=_copy_model_value(
            getattr(system_obj, "input_values", None)
        ),
        parameter_dependent_ic=getattr(
            system_obj, "parameter_dependent_ic", False
        ),
        ic_parameters=_copy_model_value(
            getattr(system_obj, "ic_parameters", None)
        ),
        timepoints_ode=_copy_model_value(timepoints_ode),
        timepoints_ssm=_copy_model_value(timepoints_ssm),
        **kwargs,
    )


def _as_reducible(
    system_obj,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
    **kwargs,
):
    """Return a `Reduce` object for direct reduction calls."""
    if in_place and isinstance(system_obj, Reduce):
        if timepoints_ode is not None:
            system_obj.timepoints_ode = timepoints_ode
        if timepoints_ssm is not None:
            system_obj.timepoints_ssm = timepoints_ssm
        for option in ("error_tol", "nstates_tol", "nstates_tol_min"):
            if option in kwargs and kwargs[option] is not None:
                setattr(system_obj, option, kwargs[option])
        return system_obj
    if in_place:
        raise ValueError("in_place=True requires a Reduce object.")
    if not isinstance(system_obj, System):
        raise TypeError("system_obj must be an AutoReduce System object.")
    return get_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        **kwargs,
    )


def get_error_metric(
    system_obj,
    reduced_system,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
):
    """Compute the full-vs-reduced output error from systems."""
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
    )
    return reducible_system.get_error_metric(reduced_system)


def get_robustness_metric(
    system_obj,
    reduced_system,
    timepoints_ode=None,
    timepoints_ssm=None,
    in_place=False,
    **kwargs,
):
    """Compute robustness metrics from systems."""
    reducible_system = _as_reducible(
        system_obj,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        in_place=in_place,
    )
    return reducible_system.get_robustness_metric(reduced_system, **kwargs)

