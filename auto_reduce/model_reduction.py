
from .system import System
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve, Eq
import sympy
import numpy as np
import warnings
from scipy.linalg import solve_lyapunov, block_diag, eigvals, norm
from auto_reduce import utils
class Reduce(System):
    '''
    The class can be used to compute the various possible reduced models for the System object 
    and then find out the best reduced model choice using doi : https://doi.org/10.1101/640276 
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, u = None,
                params_values = [], x_init = [], timepoints_ode = None, timepoints_ssm = None,
                error_tol = None, nstates_tol = None, nstates_tol_min = None):
        super().__init__(x, f, params, C, g, h, u, params_values, x_init)
        self.f_hat = [] # Should be a list of Sympy objects
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
            self.timepoints_ode = np.linspace(0,100,100)
        else:
            self.timepoints_ode = timepoints_ode
        if timepoints_ssm is None:
            self.timepoints_ssm = np.linspace(0,100,10)
        else:
            self.timepoints_ssm = timepoints_ssm
        self.results_dict = {}
        self.x_c = []
        self.x_sol = None
        self.x_sol2 = None
        self.full_ssm = None
        return

    def get_output_states(self):
        x = self.x
        outputs = list(np.dot(np.array(self.C), np.array(x))) # Get y = C*x
        output_symbols = [list(i.free_symbols) for i in outputs]
        output_states = [item for sublist in output_symbols for item in sublist]
        return output_states

    def get_all_combinations(self):
        '''
        Combinatorially create sets of all states that can be reduced in self.all_reductions.
        In addition, returns the possible_reductions list after removing the sets that
        contain states involved in the outputs.
        '''
        from itertools import combinations
        possible_reductions = []
        n = self.n
        for i in range(n):
            if i != n-1:
                comb = combinations(list(range(n)), i+1)
                possible_reductions.append(list(comb))
        possible_reductions = [list(item) for sublist in possible_reductions for item in sublist]
        self.all_combinations = [i for i in possible_reductions]
        output_states = self.get_output_states()
        restart = False
        x = self.x
        for attempt in self.all_combinations:
            states_attempt = [x[i] for i in attempt]
            if not len(set(states_attempt).intersection(set(output_states))) == len(output_states) or len(attempt) > self.nstates_tol:
                restart = True
            if restart:
                possible_reductions.remove(attempt)
                restart = False

        # Remove state(s) that consist of input(s)
        if self.u is not None:
            for i in range(len(self.g)):
                if self.g[i] != 0:
                    if i in possible_reductions:
                        # This index state variable should not be in possible_reductions list
                        possible_reductions.remove(i)
        return possible_reductions

    def get_T(self, attempt):
        non_attempt = [i for i in range(self.n) if i not in attempt]
        T = np.zeros( (self.n, self.n) )
        n_hat = len(attempt)
        n = self.n
        n_c = n - n_hat
        T1 = np.zeros( (self.n, n_hat) )
        T2 = np.zeros( (self.n, n_c) )
        # For x_hat
        for ni in range(0, n_hat):
            set_T = False
            for i in range(n):
                if i in attempt and not set_T:
                    T[ni,i] = 1
                    attempt.remove(i)
                    set_T = True
        # For x_c
        for ni in range(n_hat, n):
            set_T = False
            for i in range(n):
                if i in non_attempt and not set_T:
                    T[ni,i] = 1
                    non_attempt.remove(i)
                    set_T = True
        T1 = T[0:n,0:n_hat]
        T2 = T[0:n,n_hat:n + 1]
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
        y = self.C@x_sol
        x_sols_hat = reduced_ode.solve_system().T
        reduced_sys.x_sol = x_sols_hat
        y_hat = np.array(reduced_sys.C)@np.array(x_sols_hat)
        if np.shape(y) == np.shape(y_hat):
                e = np.linalg.norm(y - y_hat)
        else:
            raise ValueError('The output dimensions must be the same for reduced and full model. Choose C and C_hat accordingly')
        if np.isnan(e):
            print('The error is NaN, something wrong...continuing.')
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
        y = self.C@x_sol
        x_sols_hat = reduced_ode.solve_system().T
        reduced_sys.x_sol = x_sols_hat
        y_hat = np.array(reduced_sys.C)@np.array(x_sols_hat)
        if np.shape(y) == np.shape(y_hat):
                e = np.linalg.norm(y - y_hat)
        else:
            raise ValueError('The output dimensions must be the same for reduced and full model. Choose C and C_hat accordingly')
        if np.isnan(e):
            print('The error is NaN, something wrong...continuing.')
        return e

    def get_robustness_metric_with_input(self, reduced_sys):
        return

    def get_robustness_metric(self, reduced_sys):
        # Create an option so the default way this is done is given two systems compute robustness metric. 
        # Implementing Theorem 2
        timepoints_ssm = self.timepoints_ssm
        _, x_sols, full_ssm = self.get_solutions()
        S = full_ssm.compute_SSM()
        self.S = S
        reduced_ssm = utils.get_SSM(reduced_sys, timepoints_ssm)
        x_sols_hat = utils.get_ODE(reduced_sys, timepoints_ssm).solve_system().T
        x_sols = np.reshape(x_sols, (len(timepoints_ssm), self.n))
        x_sols_hat = np.reshape(x_sols_hat, (len(timepoints_ssm), reduced_sys.n))
        Se = np.zeros(len(self.params_values))
        S_hat = reduced_ssm.compute_SSM()
        reduced_sys.S = S_hat
        S_bar = np.concatenate((S, S_hat), axis = 2)
        S_bar = np.reshape(S_bar, (len(timepoints_ssm), self.n + reduced_sys.n, len(self.params_values)))
        for j in range(len(self.params_values)):
            S_metric_max = 0
            for k in range(len(self.timepoints_ssm)):
                J = full_ssm.compute_J(x_sols[k,:])
                J_hat = reduced_ssm.compute_J(x_sols_hat[k,:])
                J_bar = block_diag(J, J_hat)
                # print(J)
                # print(J_bar)
                C_bar = np.concatenate((self.C, -1*reduced_sys.C), axis = 1)
                C_bar = np.reshape(C_bar, (np.shape(self.C)[0], (self.n + reduced_sys.n)))
                # if np.isnan(J).any() or np.isnan(J_hat).any() or np.isfinite(J).all() or np.isfinite(J_hat).all():
                #     warnings.warn('NaN or inf found in Jacobians, continuing')
                #     continue
                P = solve_lyapunov(J_bar, -1 * C_bar.T@C_bar)
                if k == 0:
                    max_eig_P = max(eigvals(P))
                Z = full_ssm.compute_Zj(x_sols[k,:], j)
                Z_hat = reduced_ssm.compute_Zj(x_sols_hat[k,:], j)
                Z_bar = np.concatenate((Z,Z_hat), axis = 0)
                Z_bar = np.reshape(Z_bar, ( (self.n + reduced_sys.n), 1 ) )
                S_metric = norm(Z_bar.T@P@S_bar[k,:,j])
                if  S_metric > S_metric_max:
                    S_metric_max = S_metric
            Se[j] = max_eig_P + 2*len(reduced_ssm.timepoints)*S_metric_max
            utils.printProgressBar(int(j + k*len(self.params_values)), len(timepoints_ssm)*len(self.params_values) - 1, prefix = 'Robustness Metric Progress:', suffix = 'Complete', length = 50)
        reduced_sys.Se = Se
        return Se

    def get_invariant_manifold(self, reduced_sys):
        timepoints_ode = self.timepoints_ode
        x_c = reduced_sys.x_c
        fast_states = reduced_sys.fast_states
        x_hat = reduced_sys.x
        x_sols_hat = reduced_sys.get_ODE().solve_system().T
        x_sol_c = np.zeros((len(timepoints_ode),np.shape(x_c)[0]))
        # Get the collapsed states by substituting the solutions 
        # into the algebraic relationships obtained
        for i in range(np.shape(x_sols_hat)[0]):
            # for each reduced variable (because collapsed variables are only 
            # functions of reduced variables, algebraically)
            for k in range(len(x_sols_hat[:,i])):
                for j in range(len(x_c)):
                    subs_result = fast_states[j].subs(x_hat[i], x_sols_hat[:,i][k])
                    if  subs_result == fast_states[j]:
                        continue
                    elif isinstance(subs_result, sympy.Expr):
                        # continue substituting other variables, until you get a float
                        fast_states[j] = subs_result
                    else:
                        x_sol_c[k][j] = fast_states[j].subs(x_hat[i],x_sols_hat[:,i][k])
        return x_sol_c


    def solve_timescale_separation(self, attempt_states):
        attempt = []
        for state in attempt_states:
            attempt.append(self.x.index(state))
        print('attempting reduced states index:', attempt)
        x_c = []
        fast_states = []
        f_c = []
        f_hat = []
        x_hat_init = []
        x_c_init = []
        x_hat = []
        x, f, x_init = self.x, self.f, self.x_init
        for i in range(self.n):
            if i not in attempt:
                x_c.append(x[i])
                f_c.append(f[i])
                x_c_init.append(x_init[i])
            else:
                f_hat.append(f[i])
                x_hat.append(x[i])
                x_hat_init.append(x_init[i])
        # print('Reduced set of variables is', x_hat)
        # print('f_hat = ',hat)
        # print('Collapsed set of variables is', x_c)
        lookup_collapsed = {}
        for i in range(len(x_c)):
            # print('Solving for ', x_c[i])
            x_c_sub = solve(Eq(f_c[i]), x_c[i])
            lookup_collapsed[x_c[i]] = x_c_sub
            if len(x_c_sub) == 0:
                print('Could not find solution for this collapsed variable : {0} from {1}'.format(x_c[i], f_c[i]))
                fast_states.append([])
                continue
            elif len(x_c_sub) > 1:
                # print('Multiple solutions obtained. Chooosing non-zero solution, check consistency. The solutions are ', x_c_sub)
                for sub in x_c_sub:
                    if sub == 0:
                        x_c_sub.remove(0)
            count = 0
            # print(x_c_sub[0])
            while count >= 0:
                # print(count)
                if count == len(list(x_c_sub[0].free_symbols)):
                    count = -1
                    continue
                sym = list(x_c_sub[0].free_symbols)[count]
                if sym not in x:
                    count = count + 1
                    continue
                if sym in lookup_collapsed.keys():
                    # print('The state {0} has been solved for but appears in the solution for the next variable, making the sub with {1} into the corresponding f_c and solving again should fix this.'.format(sym, lookup_collapsed[sym][0]))
                    if lookup_collapsed[sym][0] is None:
                        raise ValueError('Something went wrong...Check reduced/full/collapsed state descriptions.')
                    f_c[i] = f_c[i].subs(sym, lookup_collapsed[sym][0])
                    # print('Updating old x_c_sub then')
                    x_c_sub = solve(Eq(f_c[i]), x_c[i])
                    if len(x_c_sub) > 1:
                        # print('Multiple solutions obtained. Chooosing non-zero solution, check consistency. The solutions are ', x_c_sub)
                        for sub in x_c_sub:
                            if sub == 0:
                                x_c_sub.remove(0)
                    # print('with ',x_c_sub)
                    lookup_collapsed[x_c[i]] = x_c_sub
                    count = 0
                elif count != len(list(x_c_sub[0].free_symbols)) - 1:
                    count = count + 1
            # print('Solved for {0} to get {1}'.format(x_c[i], x_c_sub[0]))
            # This x_c_sub should not contain previously eliminated variables otherwise circles continue
            fast_states.append(x_c_sub[0])

        for i in range(len(fast_states)):
            if fast_states[i] == []:
                continue
            for j in range(len(f_hat)):
                # print('Substituting {0} for variable {1} into f_hat{2}'.format(fast_states[i], x_c[i], j))
                f_hat[j] = f_hat[j].subs(x_c[i], fast_states[i])
                # print('f_hat = ',f_hat[j])
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
            for sym in fi.free_symbols:
                if sym not in free_symbols_all:
                    free_symbols_all.append(sym)
        bugged_states = []
        for syms in free_symbols_all:
            if syms not in x_hat + self.params:
                bugged_states.append(syms)
                flag = True
        if flag:
            warnings.warn('Check model consistency')
            print('The time-scale separation that retains states {0} does not work because the state-variables {1} appear in the reduced model'.format(attempt, bugged_states))
            return None, None
        else:
            print('Successful time-scale separation solution obtained.')

        reduced_sys = create_system(x_hat, f_hat, params = self.params, C = C_hat,
                            params_values = self.params_values, x_init = x_hat_init)
        fast_subsystem = create_system(x_c, f_c, params = self.params,
                            params_values = self.params_values, x_init = x_c_init)
        reduced_sys.x_c = x_c
        reduced_sys.fast_states = fast_states
        return reduced_sys, fast_subsystem

    def solve_timescale_separation_with_input(self, attempt_states):
        attempt = []
        for state in attempt_states:
            attempt.append(self.x.index(state))
        print('attempting to retain:', attempt)
        x_c = []
        fast_states = []
        f_c = []
        f_hat = []
        x_hat_init = []
        x_c_init = []
        x_hat = []
        x, f, g, u, x_init = self.x, self.f, self.g, self.u, self.x_init
        f_g = [fi + gi for fi, gi in zip(f,g)]
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
        for i in range(len(x_c)):
            x_c_sub = solve(Eq(f_c[i]), x_c[i])
            lookup_collapsed[x_c[i]] = x_c_sub
            if len(x_c_sub) == 0:
                # print('Could not find solution for this collapsed variable : {0} from {1}'.format(x_c[i], f_c[i]))
                fast_states.append([])
                continue
            elif len(x_c_sub) > 1:
                # print('Multiple solutions obtained. Chooosing non-zero solution, check consistency. The solutions are ', x_c_sub)
                for sub in x_c_sub:
                    if sub == 0:
                        x_c_sub.remove(0)
            else:
                for sym in x_c_sub[0].free_symbols:
                    if sym in solved_states and sym in x:
                        # print('The state {0} has been solved for but appears in the solution for the next variable, making the sub with {1} into the corresponding f_c and solving again should fix this.'.format(sym, lookup_collapsed[sym][0]))
                        f_c[i] = f_c[i].subs(sym, lookup_collapsed[sym][0])
                        # print('Updating old x_c_sub then')
                        x_c_sub = solve(Eq(f_c[i]), x_c[i])
                        if len(x_c_sub) > 1:
                            print('Multiple solutions obtained. Chooosing non-zero solution, check consistency. The solutions are ', x_c_sub)
                            for sub in x_c_sub:
                                if sub == 0:
                                    x_c_sub.remove(0)
                        # print('with ',x_c_sub)
                        lookup_collapsed[x_c[i]] = x_c_sub
                    else:
                        solved_states.append(x_c[i])
                # print('Solved for {0} to get {1}'.format(x_c[i], x_c_sub[0]))
                # This x_c_sub should not contain previously eliminated variables otherwise circles continue
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
            for sym in fi.free_symbols:
                if sym not in free_symbols_all:
                    free_symbols_all.append(sym)
        bugged_states = []
        for syms in free_symbols_all:
            if syms not in x_hat + u + self.params:
                bugged_states.append(syms)
                flag = True
        if flag:
            warnings.warn('Check model consistency')
            print('The time-scale separation that retains states {0} does not work because the state-variables {1} appear in the reduced model'.format(attempt, bugged_states))
            # return None, None

        reduced_sys = create_system(x_hat, f_hat, params = self.params, C = C_hat,
                            params_values = self.params_values, x_init = x_hat_init)
        fast_subsystem = create_system(x_c, f_c, params = self.params,
                            params_values = self.params_values, x_init = x_c_init)
        reduced_sys.x_c = x_c
        reduced_sys.bugged_states = bugged_states
        reduced_sys.fast_states = fast_states
        return reduced_sys, fast_subsystem


    def set_conservation_laws(self, conserved_quantities, sym_states_to_eliminate):
        '''
        From the conserved_quantities list, this method computes the expressions for each of the 
        state indices in states_to_eliminate, and substitutes into the full model dynamics. 
        Both lists should contain symbolic variables referencing states in self.f.
        Returns the dynamics self.f. 
        '''
        states_to_eliminate = [self.x.index(state) for state in sym_states_to_eliminate]
        for i in range(len(states_to_eliminate)):
            state = self.x[states_to_eliminate[i]]
            state_sub = solve(Eq(conserved_quantities[i]), state)
            for j in range(len(self.f)):
                self.f[j] = self.f[j].subs(state, state_sub[0])

        arr_x = np.array(self.x)
        self.x = np.delete(arr_x, states_to_eliminate).tolist()
        arr_f = np.array(self.f)
        self.f = np.delete(arr_f, states_to_eliminate).tolist()
        self.x_init = np.delete(self.x_init, states_to_eliminate).tolist()
        self.C = np.delete(np.array(self.C), states_to_eliminate, axis=1)
        self.n = self.n - len(states_to_eliminate)
        return self.f

    def solve_approximations(self):
        pass

    def get_solutions(self):
        if self.x_sol is None:
            x_sol = utils.get_ode_solutions(self.get_system(), self.timepoints_ode)
            self.x_sol = x_sol
        if self.x_sol2 is None:
            x_sol2 = utils.get_ode_solutions(self.get_system(), self.timepoints_ssm)
            self.x_sol2 = x_sol2
        if self.full_ssm is None:
            full_ssm = utils.get_SSM(self.get_system(), self.timepoints_ssm)
            self.full_ssm = full_ssm
        return self.x_sol, self.x_sol2, self.full_ssm


    def reduce_simple(self):
        if self.u is not None:
            raise ValueError('For models with inputs use reduce_with_input method.')
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print('No possible reduced models found. Try increasing tolerance for number of states.')
            return
        for attempt in possible_reductions:
            if len(attempt) < self.nstates_tol_min:
                continue
            elif len(attempt) > self.nstates_tol:
                continue
            attempt_states = [self.x[i] for i in attempt]
            # Create reduced systems
            reduced_sys, fast_subsystem = self.solve_timescale_separation(attempt_states)
            if reduced_sys is None or fast_subsystem is None:
                continue
            # Get metrics for this reduced system
            e = self.get_error_metric(reduced_sys)
            Se = self.get_robustness_metric(reduced_sys)
            results_dict[reduced_sys] = [e, Se]
        self.results_dict = results_dict
        return self.results_dict

    def reduce_with_input(self):
        if self.u is None:
            raise ValueError('For models with no inputs use reduce_simple')
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print('No possible reduced models found. Try increasing tolerance for number of states.')
            return
        for attempt in possible_reductions:
            attempt_states = [self.x[i] for i in attempt]
            # Create reduced systems
            reduced_sys, fast_subsystem = self.solve_timescale_separation_with_input(attempt_states)
            if reduced_sys is None or fast_subsystem is None:
                continue
            # Get metrics for this reduced system
            e = self.get_error_metric_with_input(reduced_sys)
            Se = self.get_robustness_metric_with_input(reduced_sys)
            results_dict[reduced_sys] = [e, Se]
        self.results_dict = results_dict
        return self.results_dict

    def reduce_general(self):
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        if not len(possible_reductions):
            print('No possible reduced models found. Try increasing tolerance for number of states.')
            return

        self.results_dict = results_dict
        return self.results_dict

    def compute_reduced_model(self):
        if self.C is not None and self.g is None:
            # Call y = Cx based model reduction
            print('Using model reduction algorithm with y = Cx, linear output relationship and no inputs (g = 0).')
            self.results_dict = self.reduce_simple()
            return self.results_dict
        else:
            print('Using general model reduction algorithm with inputs and nonlinear output relationship')
            self.results_dict = self.reduce_general()
            return self.results_dict

    def get_system(self):
        return System(self.x, self.f, params = self.params, C = self.C, g = self.g,
                    h = self.h, u = self.u, params_values = self.params_values, x_init = self.x_init)


class ReduceUtils(Reduce):
    '''
    For various utility methods developed on top of Reduce class and other utility functions
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, u = None,
                params_values = [], x_init = [], timepoints_ode = None, timepoints_ssm = None,
                error_tol = None, nstates_tol = None):
        super().__init__(x, f, params, C, g, h, params_values, x_init, timepoints_ode, timepoints_ssm,
                        error_tol, nstates_tol)


    def write_results(self, filename):
        '''
        Write the model reduction results in a file given by filename. 
        The symbolic data is written in LaTeX.
        '''
        from sympy.printing import latex
        f1 = open(filename, 'w')
        f1.write('Model reduction results.\n')
        for key,value in self.results_dict.items():
            f1.write('A possible reduced model: \n \n')
            f1.write('\n$x_{hat} = ')
            f1.write(str(key.x))
            f1.write('$\n\n\n\n')
            for k in range(len(key.f)):
                f1.write('\n$f_{hat}('+ str(k+1) + ') = ')
                f1.write(latex(key.f[k]))
                f1.write('$\n\n')
            f1.write('\n\n\n')
            f1.write('\nError metric:')
            f1.write(str(value[0]))
            f1.write('\n\n\n')
            f1.write('\nRobustness metric:')
            f1.write(str(value[1]))
            f1.write('\n\n\n')
            f1.write('Other properties') 
            f1.write('\n\n\n')
            f1.write('\n C = ')
            f1.write(str(key.C))
            f1.write('\n$ g = ')
            f1.write(str(key.g))
            f1.write('$\n h = ')
            f1.write(str(key.h))
            f1.write('\n$h = ')
            f1.write(str(key.h))
            f1.write('$\n Solutions : \n')
            f1.write(str(key.x_sol))
            f1.write('\n\n\n\n')
            f1.write('\n Sensitivity Solutions : \n')
            f1.write(str(key.S))
            f1.write('\n\n\n\n')
        f1.close()

    def get_valid_reduced_models(self, nstates_tol = None, error_tol = None):
        '''
        Returns the reduced models obtained and stored in results_dict that satisfy the given 
        tolerances for number of states and the error tolerance. Among the return reduced model objects
        you may access robustness metric for each by looking at reduced_sys.Se. Choose the reduced model 
        with lowest robustness metric.
        '''
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


def create_system(x, f, params = None, C = None, g = None, h = None, u = None,
                params_values = [], x_init = []):
    return System(x, f = f, params = params, C = C, g = g, h = h, u = u,
                params_values = params_values, x_init = x_init)

