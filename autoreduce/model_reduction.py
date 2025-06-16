"""Model reduction"""

import warnings
from itertools import combinations

from sympy import Symbol, solve, Eq  # type: ignore
import sympy  # type: ignore
import numpy as np  # type: ignore

from scipy.linalg import solve_lyapunov, block_diag  # type: ignore
from scipy.linalg import eigvals, norm  # type: ignore

from .system import System
from autoreduce import utils


class Reduce(System):
    """
    The class can be used to compute the various
    possible reduced models for the System object
    and then find out the best reduced
    model choice using doi : https://doi.org/10.1101/640276
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
        timepoints_ode=None,
        timepoints_ssm=None,
        error_tol=None,
        nstates_tol=None,
        nstates_tol_min=None,
        **kwargs,
    ):
        super().__init__(
            x, f, params, C, g, h, u, params_values, x_init, **kwargs
        )
        self.f_hat = []  # Should be a list of Sympy objects
        if nstates_tol is None:
            self.nstates_tol = self.n - 1
        else:
            self.nstates_tol = nstates_tol
        if nstates_tol_min is None:
            self.nstates_tol_min = 1
        else:
            self.nstates_tol_min = nstates_tol_min
        if error_tol is None:
            self.error_tol = 1e6
        else:
            self.error_tol = error_tol
        if timepoints_ode is None:
            self.timepoints_ode = np.linspace(0, 100, 100)
        else:
            self.timepoints_ode = timepoints_ode
        if timepoints_ssm is None:
            self.timepoints_ssm = np.linspace(0, 100, 10)
        else:
            self.timepoints_ssm = timepoints_ssm
        self.results_dict = {}
        self.x_c = []
        self.x_sol = None
        self.x_sol2 = None
        self.full_ssm = None
        return

    def get_output_states(self):
        if self.C is None and self.h is None:
            return []
        outputs = np.dot(np.array(self.C), np.array(self.x))  # Get y = C*x
        if type(outputs) is not np.ndarray:
            outputs = [outputs]
        else:
            outputs = outputs.tolist()
        output_symbols = [list(i.free_symbols) for i in outputs]
        output_states = [
            item for sublist in output_symbols for item in sublist
        ]
        return output_states

    def get_all_combinations(self):
        """
        Combinatorially create sets of all states
        that can be reduced in self.all_reductions.
        In addition, returns the possible reductions
        list after removing the sets that
        contain states involved in the outputs.
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

        # Remove state(s) that consist of input(s)
        if self.u is not None:
            for i, _ in enumerate(self.g):
                if self.g[i] != 0:
                    if i in possible_reductions:
                        # This index state variable should
                        # not be in possible_reductions list
                        possible_reductions.remove(i)
        return possible_reductions

    def get_T(self, attempt):
        non_attempt = [i for i in range(self.n) if i not in attempt]
        T = np.zeros((self.n, self.n))
        n_hat = len(attempt)
        n = self.n
        n_c = n - n_hat
        T1 = np.zeros((self.n, n_hat))
        T2 = np.zeros((self.n, n_c))
        # For x_hat
        for ni in range(0, n_hat):
            set_T = False
            for i in range(n):
                if i in attempt and not set_T:
                    T[ni, i] = 1
                    attempt.remove(i)
                    set_T = True
        # For x_c
        for ni in range(n_hat, n):
            set_T = False
            for i in range(n):
                if i in non_attempt and not set_T:
                    T[ni, i] = 1
                    non_attempt.remove(i)
                    set_T = True
        T1 = T[0:n, 0:n_hat]
        T2 = T[0:n, n_hat : n + 1]
        return T, T1, T2

    def get_error_metric_with_input(self, reduced_sys):
        """
        Returns the error defined as the 2-norm of y - y_hat.
        y = Cx and y_hat = C_hat x_hat OR
        y = h(x, P), y_hat = h_hat(x_hat, P).
        Important : What is the input?
        """
        reduced_ode = utils.get_ODE(reduced_sys, self.timepoints_ode)
        x_sol, _, _ = self.get_solutions()
        y = self.C @ x_sol
        x_sols_hat = reduced_ode.solve_system().T
        reduced_sys.x_sol = x_sols_hat
        y_hat = np.array(reduced_sys.C) @ np.array(x_sols_hat)
        if np.shape(y) == np.shape(y_hat):
            e = np.linalg.norm(y - y_hat)
        else:
            raise ValueError(
                "The output dimensions must be the same for"
                + "reduced and full model. Choose C and C_hat accordingly"
            )
        if np.isnan(e):
            print("The error is NaN, something wrong...continuing.")
        return e

    def get_error_metric(self, reduced_sys):
        # Give option for get_error_metric(sys1, sys2)
        """
        Returns the error defined as the 2-norm of y - y_hat.
        y = Cx and y_hat = C_hat x_hat OR
        y = h(x, P), y_hat = h_hat(x_hat, P)
        """
        reduced_ode = utils.get_ODE(reduced_sys, self.timepoints_ode)
        x_sol, _, _ = self.get_solutions()
        y = self.C @ x_sol
        x_sols_hat = reduced_ode.solve_system().T
        reduced_sys.x_sol = x_sols_hat
        y_hat = np.array(reduced_sys.C) @ np.array(x_sols_hat)
        if np.shape(y) == np.shape(y_hat):
            e = np.linalg.norm(y - y_hat)
        else:
            raise ValueError(
                "The output dimensions must be the same for"
                + "reduced and full model. Choose C and C_hat accordingly"
            )
        if np.isnan(e):
            print("The error is NaN, something wrong...continuing.")
        return e

    def get_robustness_metric_with_input(self, reduced_sys):
        return

    def get_robustness_metric(self, reduced_sys, **kwargs):
        # Create an option so the default way this is
        # done is given two systems compute robustness metric.
        # Implementing Theorem 2
        if "method" in kwargs:
            method = kwargs.get("method")
        else:
            method = "direct"
        timepoints_ssm = self.timepoints_ssm
        _, x_sols, full_ssm = self.get_solutions()
        S = full_ssm.compute_SSM()
        self.S = S
        reduced_ssm = utils.get_SSM(reduced_sys, timepoints_ssm)
        x_sols_hat = (
            utils.get_ODE(reduced_sys, timepoints_ssm).solve_system().T
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
                    # print(J)
                    # print(J_bar)
                    # if np.isnan(J).any() or np.isnan(J_hat).any()
                    # or np.isfinite(J).all() or np.isfinite(J_hat).all():
                    #     warnings.warn('NaN or inf found in Jacobians, continuing')
                    #     continue
                    P = solve_lyapunov(J_bar, -1 * C_bar.T @ C_bar)
                    eig_P = max(eigvals(P))
                    # if k == 0: # used when proof I thought said that lambda_max_P was at time 0 for IC term.
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
                    utils.printProgressBar(
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

    def get_invariant_manifold(self, reduced_sys):
        timepoints_ode = self.timepoints_ode
        x_c = reduced_sys.x_c
        fast_states = reduced_sys.fast_states
        x_hat = reduced_sys.x
        x_sols_hat = reduced_sys.get_ODE().solve_system().T
        x_sol_c = np.zeros((len(timepoints_ode), np.shape(x_c)[0]))
        # Get the collapsed states by substituting the solutions
        # into the algebraic relationships obtained
        for i in range(np.shape(x_sols_hat)[0]):
            # for each reduced variable (because collapsed variables are only
            # functions of reduced variables, algebraically)
            for k in range(len(x_sols_hat[:, i])):
                for j in range(len(x_c)):
                    subs_result = fast_states[j].subs(
                        x_hat[i], x_sols_hat[:, i][k]
                    )
                    if subs_result == fast_states[j]:
                        continue
                    elif isinstance(subs_result, sympy.Expr):
                        # continue substituting other variables, until you get a float
                        fast_states[j] = subs_result
                    else:
                        x_sol_c[k][j] = fast_states[j].subs(
                            x_hat[i], x_sols_hat[:, i][k]
                        )
        return x_sol_c

    def solve_timescale_separation(
        self, slow_states, fast_states=None, **kwargs
    ):
        """
        This function solves the time-scale separation
        problem for the System object passed through.
        Arguments:
        * slow_states: List of states assumed to
        have slow dynamics => retained in the reduced model.
        This is list of Sympy Symbol objects
        corresponding to the System.x list.
        * fast_states: List of states assumed to
        have fast dynamics => states that will be collapsed. Usually,
        this is automatically populated as all those
        states that are in System.x but not in slow_states.
        """
        # Get 'debug' keyword (if called in debug = True mode):
        if "debug" in kwargs:
            debug = kwargs.get("debug")
        else:
            debug = False
        # If slow_states is empty, then reduced model = given model
        # and collapsed model is None
        if not slow_states:
            return self.get_system(), None
        x, f, x_init = self.x, self.f, self.x_init

        len_slow_states = len(slow_states)
        x_hat = [None] * len_slow_states
        x_hat_init = [None] * len_slow_states
        f_hat = [None] * len_slow_states

        max_len_fast_states = len(x) - len(slow_states)
        x_c = [None] * max_len_fast_states
        x_c_init = [None] * max_len_fast_states
        f_c = [None] * max_len_fast_states
        # print('f_c',f_c)
        # Populate the list of states that will retained (x_hat)
        # and those that will be collapsed (x_c)
        x_hat = slow_states
        # Make sure fast_states (direct sum) slow_states is all of it.
        # Check if fast states are already provided as well:
        if fast_states:
            x_c = fast_states
        else:
            # If not, automatically fill it up.
            count_x_c = 0
            for i in x:
                if i not in slow_states:
                    x_c[count_x_c] = i
                    count_x_c += 1
            fast_states = x_c
        # Consistency check for slow and fast states:
        if len(slow_states) + len(fast_states) != len(self.x):
            raise RuntimeError(
                "Number of slow states plus number of fast states must equal the number of total states."
            )
        for state in slow_states:
            if state in fast_states:
                raise RuntimeError(
                    "Found a state that is both fast and slow! Unfortunately, that is not yet possible in this reality."
                )
        # Now populate the default corresponding
        # f_c and f_hat dynamics from self.f
        # and inital conditions from self.x_init
        for i in x:
            state_index = x.index(i)
            if i in x_c:
                x_c_index = x_c.index(i)
                f_c[x_c_index] = f[state_index]
                if self.parameter_dependent_ic:
                    param_as_ic = self.ic_parameters[state_index]
                    value_ic = self.set_ic_from_params(
                        x_c_init, param_as_ic, x_c_index
                    )
                else:
                    x_c_init[x_c_index] = x_init[state_index]
            if i in x_hat:
                x_hat_index = x_hat.index(i)
                f_hat[x_hat_index] = f[state_index]
                if self.parameter_dependent_ic:
                    param_as_ic = self.ic_parameters[state_index]
                    value_ic = self.set_ic_from_params(
                        x_hat_init, param_as_ic, x_hat_index
                    )
                else:
                    x_hat_init[x_hat_index] = x_init[state_index]
        self.f_hat = f_hat
        self.f_c = f_c
        if debug:
            print("Reduced set of variables is", x_hat)
            print("f_hat = ", self.f_hat)
            print("Collapsed set of variables is", x_c)

        # Get the reduced (slow system) dynamics:
        # Check after substituting each solution into
        # f_hat whether resulting f_hat sympy ODE
        # has any remaining variables that should have been collapsed.
        loop_sanity = True
        count = 0
        solution_dict = {}
        while (
            sympy_variables_exist(
                ode_function=self.f_hat, variables_to_check=x_c
            )[0]
            and loop_sanity
        ):
            # print(sympy_solve_and_substitute(ode_function = self.f_hat, collapsed_states = x_c,
            #                                 collapsed_dynamics = self.f_c,
            #                                 solution_dict = solution_dict,
            #                                 debug = debug))
            self.f_hat, solution_dict, self.f_c = sympy_solve_and_substitute(
                ode_function=self.f_hat,
                collapsed_states=x_c,
                collapsed_dynamics=self.f_c,
                solution_dict=solution_dict,
                debug=debug,
            )
            if count > 2:
                loop_sanity = False
                warnings.warn(
                    "Solve time-scale separation failed. Check model consistency."
                )
                print(
                    f"Did not work to retain: {slow_states}"
                    " because either a collapsed state-variables appears"
                )
                print(" in the reduced model or a solution is not possible.")
                return None, None
            count += 1

        # Get the collapsed (fast system) dynamics to create collapsed_system
        for i, _ in enumerate(x_hat):
            for j in range(len(self.f_c)):
                # The slow variables stay at steady state in the fast subsystem
                self.f_c[j] = self.f_c[j].subs(x_hat[i], x_hat_init[i])

        # Create C_hat
        C_hat = self.create_C_hat(x_hat)
        for index, _ in enumerate(f_hat):
            f_hat[index] = sympy.simplify(f_hat[index])
        for index, _ in enumerate(f_c):
            f_c[index] = sympy.simplify(f_c[index])
        reduced_sys = create_system(
            x_hat,
            self.f_hat,
            params=self.params,
            C=C_hat,
            params_values=self.params_values,
            x_init=x_hat_init,
        )
        fast_subsystem = create_system(
            x_c,
            self.f_c,
            params=self.params,
            params_values=self.params_values,
            x_init=x_c_init,
        )
        reduced_sys.fast_states = fast_states
        # If you got to here,
        print(f"Successful solution obtained with states: {reduced_sys.x}!")
        return reduced_sys, fast_subsystem

    def solve_timescale_separation_with_input(self, attempt_states):
        attempt = []
        for state in attempt_states:
            attempt.append(self.x.index(state))
        print("attempting to retain:", attempt)
        x_c = []
        fast_states = []
        f_c = []
        f_hat = []
        x_hat_init = []
        x_c_init = []
        x_hat = []
        x, f, g, u, x_init = self.x, self.f, self.g, self.u, self.x_init
        f_g = [fi + gi for fi, gi in zip(f, g)]
        for i in range(self.n):
            if i not in attempt:
                x_c.append(x[i])
                f_c.append(f_g[i])
                x_c_init.append(x_init[i])
            else:
                f_hat.append(f_g[i])
                x_hat.append(x[i])
                x_hat_init.append(x_init[i])
        # print('Reduced set of variables is', x_hat)
        # print('f_hat = ',f_hat)
        # print('Collapsed set of variables is', x_c)

        solved_states = []
        lookup_collapsed = {}
        for i, _ in enumerate(x_c):
            x_c_sub = solve(Eq(f_c[i], 0), x_c[i])
            lookup_collapsed[x_c[i]] = x_c_sub
            if len(x_c_sub) == 0:
                # print('Could not find solution for this collapsed variable :
                # {0} from {1}'.format(x_c[i], f_c[i]))
                fast_states.append([])
                continue
            elif len(x_c_sub) > 1:
                # print('Multiple solutions obtained. Chooosing non-zero solution,
                # check consistency. The solutions are ', x_c_sub)
                for sub in x_c_sub:
                    if sub == 0:
                        x_c_sub.remove(0)
            else:
                for sym in x_c_sub[0].free_symbols:
                    if sym in solved_states and sym in x:
                        # print('The state {0} has been solved for but appears
                        # in the solution for the next variable,
                        # making the sub with
                        # {1} into the corresponding f_c
                        # and solving again should
                        # fix this.'.format(sym, lookup_collapsed[sym][0]))
                        f_c[i] = f_c[i].subs(sym, lookup_collapsed[sym][0])
                        # print('Updating old x_c_sub then')
                        x_c_sub = solve(Eq(f_c[i], 0), x_c[i])
                        if len(x_c_sub) > 1:
                            print("Multiple solutions obtained.")
                            print(
                                "Chooosing non-zero solution,"
                                "check consistency."
                            )
                            print(" The solutions are ", x_c_sub)
                            for sub in x_c_sub:
                                if sub == 0:
                                    x_c_sub.remove(0)
                        # print('with ',x_c_sub)
                        lookup_collapsed[x_c[i]] = x_c_sub
                    else:
                        solved_states.append(x_c[i])
                # print('Solved for {0} to get {1}'.format(x_c[i], x_c_sub[0]))
                # This x_c_sub should not contain previously eliminated
                # variables otherwise circles continue
                fast_states.append(x_c_sub[0])

        for i in range(len(fast_states)):
            if fast_states[i] == []:
                continue
            for j in range(len(f_hat)):
                # print('Substituting {0} for variable {1} into f_hat{2}'.format(fast_states[i], x_c[i], j))
                f_hat[j] = f_hat[j].subs(x_c[i], fast_states[i])
                # print('f_hat = ',f_hat[j])
            for j in range(len(f_c)):
                # print('Substituting {0} for variable {1} into f_c{2}'.format(fast_states[i], x_c[i], j))
                f_c[j] = f_c[j].subs(x_c[i], fast_states[i])
                # print('f_c = ',f_c[j])

        # Continue
        for i in range(len(x_hat)):
            for j in range(len(f_c)):
                # The slow variables stay at steady state in the fast subsystem
                f_c[j] = f_c[j].subs(x_hat[i], x_hat_init[i])

        # Create C_hat
        output_states = self.get_output_states()
        C_hat = np.zeros((np.shape(self.C)[0], np.shape(x_hat)[0]))
        is_output = 0
        for i in range(len(x_hat)):
            if x_hat[i] in output_states:
                is_output = 1
            for row_ind in range(np.shape(C_hat)[0]):
                C_hat[row_ind][i] = 1 * is_output

        # Create list of all free symbols
        flag = False
        free_symbols_all = []
        for fi in f_hat:
            fi = sympy.sympify(fi)
            for sym in fi.free_symbols:
                if sym not in free_symbols_all:
                    free_symbols_all.append(sym)
        bugged_states = []
        for syms in free_symbols_all:
            if syms not in x_hat + u + self.params:
                bugged_states.append(syms)
                flag = True
        if flag:
            warnings.warn("Check model consistency")
            print(
                f"The time-scale separation that retains states {attempt},\
                  does not work"
            )
            print(
                f"because the state-variables {bugged_states} \
                  appear in the reduced model"
            )
            # return None, None

        reduced_sys = create_system(
            x_hat,
            f_hat,
            params=self.params,
            C=C_hat,
            params_values=self.params_values,
            x_init=x_hat_init,
        )
        fast_subsystem = create_system(
            x_c,
            f_c,
            params=self.params,
            params_values=self.params_values,
            x_init=x_c_init,
        )
        reduced_sys.x_c = x_c
        reduced_sys.bugged_states = bugged_states
        reduced_sys.fast_states = fast_states
        return reduced_sys, fast_subsystem

    def create_C_hat(self, x_hat):
        """
        Returns C_hat matrix for the reduced system
        given the x_hat (reduced system state vector)
        and using the C matrix for the full system (if any)
        """
        if self.C is None:
            C_hat = None
        else:
            output_states = self.get_output_states()
            C_hat = np.zeros((np.shape(self.C)[0], np.shape(x_hat)[0]))
            is_output = 0
            for i in range(len(x_hat)):
                if x_hat[i] in output_states:
                    is_output = 1
                for row_ind in range(np.shape(C_hat)[0]):
                    C_hat[row_ind][i] = 1 * is_output
        return C_hat

    def get_conservation_laws(self, num_conservation_laws: int, **kwargs):
        """Finds sets of conserved species.
        Only linear combinations with coefficient = 1 supported.

        Args:
            num_conservation_laws (int): The null space of the
                                       stoichiometry matrix. In other words,
                                       the number of expected conservation laws.
        Returns:
            List of conserved species (list)
        """
        all_conserved_sets = []
        ode_list = [i for i in self.f if i != 0]
        for d in range(num_conservation_laws):
            curr_depth = d + 1
            for i in combinations(ode_list, curr_depth):
                sum_terms = 0
                for element in i:
                    if element == 0:
                        continue
                    sum_terms += element
                if sum_terms == 0:
                    conserved_species = []
                    for element in i:
                        for ode_i in range(len(self.f)):
                            if str(self.f[ode_i]) == str(element):
                                conserved_species.append(self.x[ode_i])
                    if len(conserved_species) <= 1:
                        continue
                    all_conserved_sets.append(conserved_species)
        if not all_conserved_sets:
            raise ValueError(
                "No conserved sets found. Try increasing the "
                "depth of search by increasing the number of "
                "possible conservation laws: num_conservation_laws"
            )
        return all_conserved_sets

    def setup_conservation_laws(
        self, total_quantities: dict, conserved_sets: list
    ):
        """Setup conservation laws and return a
        list of conservation laws where
        each conservation law is each sublist of
        conserved_sets equated to the corresponding
        total quantity in the total quantities dictionary.

        Args:
            total_quantities (dict): Dictionary with total quantities
                                     string keys and total value
            conserved_sets (list): A list of list consisting
            where each sublist is a
                                   set of species that are conserved
        Returns:
            conservation_laws (list): A list of conservation laws
        """
        # Setup conservation laws
        params = self.params
        params_values = self.params_values
        conservation_laws = []
        for conserved_set, tot in zip(conserved_sets, total_quantities.keys()):
            total_symbol = Symbol(tot)
            params.append(total_symbol)
            params_values.append(total_quantities[tot])
            law = 0
            for x in conserved_set:
                law += x
            law += -total_symbol
            conservation_laws.append(law)
        self.params = params
        self.params_values = params_values
        return conservation_laws

    def solve_conservation_laws(
        self,
        conservation_laws: list = None,
        total_quantities: dict = None,
        conserved_sets: list = None,
        states_to_eliminate: list = None,
        num_conservation_laws: int = 0,
        **kwargs,
    ):
        """User interface wrapper to find and set
        conservation laws for a given Reduce System object

        Args:
            conservation_laws (list, optional): A list consisting of
                                                conservation laws
                                                in the form: LHS - RHS.
                                                The RHS is assumed to be zero.
                                                If None is provided,
                                                then attempts to find
                                                conservation laws,
                                                if num_conservation_laws set.
            total_quantities (dict, optional): A dictionary of total
                                               quantities with keys
                                               consisting of strings of
                                               total quantities
                                               (RHS of conservation law)
                                               and a float value.
                                               If None provided, then a
                                               dict with parameter name
                                               is created from the state
                                               name appended by keyword
                                               "_total" and with zero value.
            conserved_sets (list of list, optional): A list of list where each
                                                     sublist consists
                                                     of species in System.x,
                                                     for which,
                                                     if corresponding elements
                                                     in System.f
                                                     are added would be equal
                                                     to zero.
                                                     If None is provided,
                                                     then attempts to find the
                                                     conserved_sets,
                                                     if num_conservation_laws
                                                     is set.
            states_to_eliminate (list, optional): A list of states to eliminate
                                                  from the set
                                                  of conserved species.
                                                  Each element in this
                                                  list must correspond to each
                                                  sublist in conserved_sets
                                                  and/or conservation_laws,
                                                  depending on
                                                  which is passed in.
                                                  If None is provided,
                                                  then creates a default
                                                  list of
                                                  states to eliminate from
                                                  variables in each law in
                                                  conservation_laws list.
            num_conservation_laws (int, optional): The dimension of the
                                                   nullspace of the
                                                   stoichiometry
                                                   matrix. In other words,
                                                   the number of expected
                                                   conservation laws.
                                                   Defaults to 0 but then
                                                   expects that either
                                                   conservation_laws or
                                                   conserved_sets is given.
        Returns:
            conserved_system (Reduce): The reduced system with
                                       conservation laws applied.

        """
        debug = kwargs.get("debug", False)
        if (
            num_conservation_laws == 0
            and conserved_sets is None
            and conservation_laws is None
        ):
            raise ValueError(
                "Must pass in something to set conservation laws! "
                "Either the list of conservation_laws, or number of conservation "
                "laws through num_conservation_laws "
                "argument or the conserved_sets list"
            )
        if (
            conservation_laws is None
            and num_conservation_laws == 0
            and conserved_sets is not None
        ):
            if conserved_sets:
                self.num_conservation_laws = len(conserved_sets)
                self.conserved_sets = conserved_sets
            else:
                raise ValueError("List of conserved sets must not be empty.")
        elif (
            conservation_laws is None
            and num_conservation_laws != 0
            and conserved_sets is None
        ):
            self.num_conservation_laws = num_conservation_laws
            self.conserved_sets = self.get_conservation_laws(
                self.num_conservation_laws
            )
        else:
            self.conserved_sets = conserved_sets

        if conservation_laws is None:
            if total_quantities is None:
                total_quantities = {}
                for c_set in self.conserved_sets:
                    total_quantities[str(c_set[0]) + "_total"] = 0
                self.total_quantities = total_quantities
            else:
                self.total_quantities = total_quantities
            self.conservation_laws = self.setup_conservation_laws(
                self.total_quantities, self.conserved_sets
            )
            print("Found conservation laws:", self.conservation_laws)
        else:
            self.conservation_laws = conservation_laws
        # Remove duplicate laws
        for law_i, law in enumerate(self.conservation_laws):
            list_conservation_laws = list(self.conservation_laws)
            list_conservation_laws.remove(law)
            if law in list_conservation_laws:
                if debug:
                    print(
                        "Found duplicate law {0} on index {1}."
                        "This will be removed. Check conservation_laws"
                        "attribute to confirm.".format(law, law_i)
                    )
                self.conservation_laws.remove(law)
        if self.conservation_laws is not None and states_to_eliminate is None:
            # Conservation laws are passed in as a list
            # but states_to_eliminate list is not available
            # Then, create it by choosing one variable from each law
            states_to_eliminate = []
            for law in self.conservation_laws:
                list_of_symbols_in_law = list(law.free_symbols)
                chosen_var = None
                index = 0
                while chosen_var is None:
                    if list_of_symbols_in_law[index] in self.x:
                        chosen_var = list_of_symbols_in_law[index]
                    index += 1
                    if index == len(list_of_symbols_in_law):
                        raise ValueError(
                            "No variable found in conservation"
                            "law {0} that can be eliminated".format(law)
                        )
                states_to_eliminate.append(chosen_var)
            self.states_to_eliminate = states_to_eliminate
            print("Choosing states to eliminate:", self.states_to_eliminate)
        else:
            self.states_to_eliminate = states_to_eliminate

        self.f = self.set_conservation_laws(
            conservation_laws=self.conservation_laws,
            states_to_eliminate=self.states_to_eliminate,
        )
        return self

    def set_conservation_laws(self, conservation_laws, states_to_eliminate):
        """
        From the conserved_quantities list,
        this method computes the expressions
        for each of the state indices in states_to_eliminate,
        and substitutes into the full model dynamics.
        Both lists should contain symbolic variables
        referencing states in self.f.
        Returns the dynamics self.f.

        Args:
            conservation_laws (list): List of conservation laws
            states_to_eliminate (list): List of Symbols of states
                                            to eliminate when applying the
                                            conservation laws

        Returns:
            Conserved ODE[list]: Conserved ODE as a list of expressions.
        """
        states_to_eliminate_new = []
        for state in states_to_eliminate:
            states_to_eliminate_new.append(self.x.index(state))
        states_to_eliminate = states_to_eliminate_new
        for i in range(len(states_to_eliminate)):
            state = self.x[states_to_eliminate[i]]
            state_sub = solve(Eq(conservation_laws[i], 0), state)
            for j in range(len(self.f)):
                self.f[j] = self.f[j].subs(state, state_sub[0])

        arr_x = np.array(self.x)
        self.x = np.delete(arr_x, states_to_eliminate).tolist()
        arr_f = np.array(self.f)
        self.f = np.delete(arr_f, states_to_eliminate).tolist()
        self.x_init = np.delete(self.x_init, states_to_eliminate).tolist()
        if self.parameter_dependent_ic:
            self.ic_parameters = np.delete(
                self.ic_parameters, states_to_eliminate
            ).tolist()
        self.C = np.delete(np.array(self.C), states_to_eliminate, axis=1)
        self.n = self.n - len(states_to_eliminate)
        return self.f

    def solve_approximations(self):
        pass

    def get_solutions(self):
        if self.x_sol is None:
            x_sol = utils.get_ode_solutions(
                self.get_system(), self.timepoints_ode
            )
            self.x_sol = x_sol
        if self.x_sol2 is None:
            x_sol2 = utils.get_ode_solutions(
                self.get_system(), self.timepoints_ssm
            )
            self.x_sol2 = x_sol2
        if self.full_ssm is None:
            full_ssm = utils.get_SSM(self.get_system(), self.timepoints_ssm)
            self.full_ssm = full_ssm
        return self.x_sol, self.x_sol2, self.full_ssm

    def reduce_simple(self, **kwargs):
        if "skip_numerical_computations" in kwargs:
            skip_numerical_computations = kwargs.get(
                "skip_numerical_computations"
            )
        else:
            skip_numerical_computations = False
        if "skip_error_computation" in kwargs:
            skip_error_computation = kwargs.get("skip_error_computation")
        else:
            skip_error_computation = False
        if "skip_robustness_computation" in kwargs:
            skip_robustness_computation = kwargs.get(
                "skip_robustness_computation"
            )
        else:
            skip_robustness_computation = False
        if self.u is not None:
            raise ValueError("For models with inputs use reduce_with_input.")
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print("No possible reduced models found.")
            print(" Try increasing tolerance for number of states.")
            return
        for attempt in possible_reductions:
            if len(attempt) < self.nstates_tol_min:
                continue
            elif len(attempt) > self.nstates_tol:
                continue
            attempt_states = [self.x[i] for i in attempt]
            # Create reduced systems
            reduced_sys, fast_subsystem = self.solve_timescale_separation(
                attempt_states, **kwargs
            )
            if reduced_sys is None or fast_subsystem is None:
                continue
            if skip_numerical_computations:
                results_dict[reduced_sys] = None
            else:
                # Get metrics for this reduced system
                if skip_error_computation:
                    e = np.nan
                else:
                    e = self.get_error_metric(reduced_sys)
                if skip_robustness_computation:
                    Se = np.nan
                    R = np.nan
                else:
                    Se, R = self.get_robustness_metric(reduced_sys, **kwargs)
                results_dict[reduced_sys] = [e, Se, R]
        self.results_dict = results_dict
        return self.results_dict

    def reduce_with_input(self):
        if self.u is None:
            raise ValueError("For models with no inputs use reduce_simple")
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print("No possible reduced models found.")
            print(" Try increasing tolerance for number of states.")
            return
        for attempt in possible_reductions:
            attempt_states = [self.x[i] for i in attempt]
            # Create reduced systems
            r_sys, f_sys = self.solve_timescale_separation_with_input(
                attempt_states
            )
            if r_sys is None or f_sys is None:
                continue
            # Get metrics for this reduced system
            e = self.get_error_metric_with_input(r_sys)
            Se, R = self.get_robustness_metric_with_input(r_sys)
            results_dict[r_sys] = [e, Se, R]
        self.results_dict = results_dict
        return self.results_dict

    def reduce_general(self):
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print("No possible reduced models found.")
            print(" Try increasing tolerance for number of states.")
            return

        self.results_dict = results_dict
        return self.results_dict

    def compute_reduced_model(self):
        if self.C is not None and self.g is None:
            # Call y = Cx based model reduction
            print("Using model reduction algorithm with y = Cx")
            print(" linear output relationship and no inputs (g = 0).")
            self.results_dict = self.reduce_simple()
            return self.results_dict
        else:
            print("Using general model reduction algorithm")
            print(" with inputs and nonlinear output relationship")
            self.results_dict = self.reduce_general()
            return self.results_dict

    def get_system(self):
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
        )


def sympy_variables_exist(ode_function, variables_to_check, **kwargs):
    """
    To check whether any variable in variables_to_check
    appears in any Sympy equation
    in the ode_function list of functions.
    Arguments:
    * ode_function: A list of Sympy functions
                    that need to be checked.
    * variables_to_check: A list of variables that
                          need to be checked for in the ode_function
    * kwargs:
    Returns a tuple:
    True, variables_that_appear:
        If any variable found, True is returned
        along with a list of variables that appear
    False, []:
        If no variable found, False is returned
        with a None (since no variables are found in ode_function).
    """
    flag = False
    all_free_symbols = []
    if "debug" in kwargs:
        debug = kwargs.get("debug")
    else:
        debug = False
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
    """A function to solve using sympy and substitute into the equation"""
    for s in collapsed_states:
        index = collapsed_states.index(s)
        f = collapsed_dynamics[index]
        if debug:
            print("In sympy_solve_and_substitute, solving for ", s)
            print("From ", f)
        solution_dict = sympy_get_steady_state_solutions(
            collapsed_variables=[s],
            collapsed_dynamics=[f],
            solution_dict=solution_dict,
            debug=debug,
        )
        if debug:
            print("Solution found: ", solution_dict)
            print("current state", s)
        if solution_dict[s] is None or len(solution_dict[s]) == 0:
            continue
        for func in ode_function:
            func = sympy.sympify(func)
            func_index = ode_function.index(func)
            updated_func = func.subs(s, solution_dict[s][0])
            ode_function[func_index] = updated_func
        for func in collapsed_dynamics:
            if func == f:
                continue
            func_index = collapsed_dynamics.index(func)
            updated_func = func.subs(s, solution_dict[s][0])
            collapsed_dynamics[func_index] = updated_func
        if debug:
            print("Updated f_hat now is ", ode_function)
        # print('returning', ode_function)
        # print('returning', solution_dict)
        # print('returning', collapsed_dynamics)
        # returned_tuple = (ode_function, solution_dict, collapsed_dynamics)
        # if len(returned_tuple) == 3:
        # print('WTF is happening')
    # return returned_tuple
    return (ode_function, solution_dict, collapsed_dynamics)


def sympy_get_steady_state_solutions(
    collapsed_variables, collapsed_dynamics, solution_dict=None, debug=False
):
    """
    Solve for each collapsed_variable from
    corresponding collapsed_dynamics one by one.
    Return the solutions as a lookup dictionary
    as a map for variables and their solutions.
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
                print("Chooosing one non-zero solution, check consistency. ")
                print(f"The solutions are {x_c_sub}.")
                print(" Highly recommend manuallly solving for this")
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
        # Search for variables existing in x_c_sub that might have been solved for before:
        # flag, solved_variables = sympy_variables_exist(ode_function = x_c_sub,
        #                                               variables_to_check = list(solution_dict.keys()),
        #                                               debug = debug)
        # if debug and flag:
        #     print('Found variables while solving that have already been solved:', solved_variables)
        # if flag:
        #     for solved_variable in solved_variables:
        #         for sub_func in x_c_sub:
        #             i = x_c_sub.index(sub_func)
        #             sub_func = sub_func.subs(solved_variable, solution_dict[solved_variable][0])
        #             x_c_sub[i] = sub_func
        # Store solution in a lookup dictionary:
        solution_dict[x_c[i]] = x_c_sub
    return solution_dict


class ReduceUtils(Reduce):
    """
    For various utility methods developed
    on top of Reduce class and other utility functions
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
        timepoints_ode=None,
        timepoints_ssm=None,
        error_tol=None,
        nstates_tol=None,
    ):
        super().__init__(
            x,
            f,
            params,
            C,
            g,
            h,
            params_values,
            x_init,
            timepoints_ode,
            timepoints_ssm,
            error_tol,
            nstates_tol,
        )

    def write_results(self, filename):
        """
        Write the model reduction results in a file given by filename.
        The symbolic data is written in LaTeX.
        """
        from sympy.printing import latex

        f1 = open(filename, "w")
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
            f1.write("\n$h = ")
            f1.write(str(key.h))
            f1.write("$\n Solutions : \n")
            f1.write(str(key.x_sol))
            f1.write("\n\n\n\n")
            f1.write("\n Sensitivity Solutions : \n")
            f1.write(str(key.S))
            f1.write("\n\n\n\n")
        f1.close()

    def get_valid_reduced_models(self, nstates_tol=None, error_tol=None):
        """
        Returns the reduced models obtained and
        stored in results_dict that satisfy the given
        tolerances for number of states and the
        error tolerance. Among the return reduced
        model objects you may access robustness metric
        for each by looking at reduced_sys.Se.
        Choose the reduced model with lowest robustness metric.
        """
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
):
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
    )
