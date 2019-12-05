
from .system import System
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve, Eq
import sympy
import numpy as np
from scipy.linalg import solve_lyapunov, block_diag, eigvals, norm
from auto_reduce import utils
class Reduce(System):
    '''
    The class can be used to compute the various possible reduced models for the System object 
    and then find out the best reduced model choice using doi : https://doi.org/10.1101/640276 
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = [], timepoints_ode = None, timepoints_ssm = None,
                error_tol = None, nstates_tol = None):
        super().__init__(x, f, params, C, g, h, params_values, x_init)
        self.f_hat = [] # Should be a list of Sympy objects
        if nstates_tol is None:
            self.nstates_tol = self.n - 1
        else:
            self.nstates_tol = nstates_tol
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

    def get_error_metric(self, reduced_sys, mode = 'Cx'):
        '''
        Returns the error defined as the 2-norm of y - y_hat.
        y = Cx and y_hat = C_hat x_hat when mode = 'Cx'.
        y = h(x, P), y_hat = h_hat(x_hat, P) when mode = 'general'
        '''
        reduced_ode = utils.get_ODE(reduced_sys, self.timepoints_ode)
        x_sol = self.x_sol
        y = self.C@x_sol
        x_sols_hat = reduced_ode.solve_system().T
        reduced_sys.x_sol = x_sols_hat
        y_hat = np.array(reduced_sys.C)@np.array(x_sols_hat)
        if np.shape(y) == np.shape(y_hat):
                e = np.linalg.norm(y - y_hat)
        else:
            raise ValueError('The output dimensions must be the same for reduced and full model. Choose C and C_hat accordingly')
        if e == 0 or np.isnan(e):
            print('The error is zero or NaN, something wrong...continuing.')
        return e

    def get_robustness_metric(self, reduced_sys, fast_subsystem, attempt):
        _, T1, T2 = self.get_T(attempt)
        T1 = np.reshape(T1, (self.n, reduced_sys.n))
        T2 = np.reshape(T2, (self.n, self.n - reduced_sys.n))
        timepoints_ssm = self.timepoints_ssm
        x_sols = self.x_sol
        full_ssm = self.full_ssm
        reduced_ssm = utils.get_SSM(reduced_sys, timepoints_ssm)
        x_sols_hat = utils.get_ODE(reduced_sys, timepoints_ssm).solve_system().T
        x_sols = np.reshape(x_sols, (len(timepoints_ssm), self.get_system().n))
        x_sols_hat = np.reshape(x_sols_hat, (len(timepoints_ssm), reduced_sys.n))
        collapsed_ssm = utils.get_SSM(fast_subsystem, timepoints_ssm)
        Se = np.zeros(len(self.params_values))
        max_eigP = 0
        S_c = collapsed_ssm.compute_SSM()
        S_hat = reduced_ssm.compute_SSM()
        reduced_sys.S = S_hat
        S_bar_c = np.concatenate((S_hat, S_c), axis = 2)
        S_bar_c = np.reshape(S_bar_c, (len(timepoints_ssm), self.n, len(self.params_values)))
        for k in range(len(timepoints_ssm)):
            J = full_ssm.compute_J(x_sols[k,:])
            J_hat = reduced_ssm.compute_J(x_sols_hat[k,:])
            J_bar = block_diag(J, J_hat)
            C_bar = np.concatenate((self.C, -1*reduced_sys.C), axis = 1)
            C_bar = np.reshape(C_bar, (np.shape(self.C)[0], (self.n + reduced_sys.n)))
            P = solve_lyapunov(J_bar, -1 * C_bar.T@C_bar)
            P11 = P[0:self.n, 0:self.n]
            P12 = P[0:self.n, self.n:self.n + reduced_sys.n + 1]
            P21 = P[self.n:self.n + reduced_sys.n + 1, 0:self.n]
            P22 = P[self.n:self.n + reduced_sys.n + 1, self.n:self.n + reduced_sys.n + 1]
            P11 = np.reshape(P11, (self.n, self.n))
            P12 = np.reshape(P12, (self.n, reduced_sys.n))
            P21 = np.reshape(P21, (reduced_sys.n, self.n))
            P22 = np.reshape(P22, (reduced_sys.n, reduced_sys.n))
            eig_P = max(eigvals(P))
            if max_eigP < eig_P:
                max_eigP = eig_P
            S_metric_max = 0
            for j in range(len(self.params_values)):
                Z = full_ssm.compute_Zj(x_sols[k,:], j)
                Z_hat = reduced_ssm.compute_Zj(x_sols_hat[k,:], j)
                Z_bar = np.concatenate((Z,Z_hat), axis = 0)
                Z_bar = np.reshape(Z_bar, ( (self.n + reduced_sys.n), 1 ) )
                q11 = np.array(P11@T1 + P12)
                q12 = np.array(P11@T2)
                q21 = np.array(P21@T1 + P22)
                q22 = np.array(P21@T2)
                Q_s = np.zeros( (self.n + reduced_sys.n, self.n))
                Q_s[0:self.n,0:reduced_sys.n] = q11
                Q_s[0:self.n,reduced_sys.n:self.n + 1] = q12
                Q_s[self.n:self.n + reduced_sys.n + 1,0:reduced_sys.n] = q21
                Q_s[self.n:self.n + reduced_sys.n + 1,reduced_sys.n:self.n + 1] = q22
                S_metric = norm(Z_bar.T@Q_s@S_bar_c[k,:,j])
                if  S_metric > S_metric_max:
                    S_metric_max = S_metric
                # TODO : Check if this is correct
                Se[j] = max_eigP + 2*len(reduced_ssm.timepoints)*S_metric_max
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

    def solve_timescale_separation(self, attempt):
        print('attempting :', attempt)
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
        for i in range(len(x_c)):
            x_c_sub = solve(Eq(f_c[i]), x_c[i])
            if len(x_c_sub) == 0:
                print('Could not find solution for this collapsed variable : {0} from {1}'.format(x_c[i], f_c[i]))
                fast_states.append([])
                continue
            else:
                fast_states.append(x_c_sub[0])
        for i in range(len(fast_states)):
            if fast_states[i] == []:
                continue
            for j in range(len(f_hat)):
                f_hat[j] = f_hat[j].subs(x_c[i], fast_states[i])
                
        for i in range(len(fast_states)):
            if fast_states[i] == []:
                continue
            for j in range(len(f_hat)):
                f_hat[j] = f_hat[j].subs(x_c[i], fast_states[i])

        for i in range(len(x_hat)):
            for j in range(len(f_c)):
                # The slow variables stay at steady state in the fast subsystem
                f_c[j] = f_c[j].subs(x_hat[i], x_hat_init[i])
                
        # Create C_hat TODO : Check 
        output_states = self.get_output_states()
        C_hat = np.zeros((np.shape(self.C)[0], np.shape(x_hat)[0]))
        is_output = 0
        for i in range(len(x_hat)):
            if x_hat[i] in output_states:
                is_output = 1 
            for row_ind in range(np.shape(C_hat)[0]):
                C_hat[row_ind][i] = 1 * is_output


        reduced_sys = create_system(x_hat, f_hat, params = self.params, C = C_hat, 
                            params_values = self.params_values, x_init = x_hat_init)
        fast_subsystem = create_system(x_c, f_c, params = self.params, 
                            params_values = self.params_values, x_init = x_c_init)
        reduced_sys.x_c = x_c
        reduced_sys.fast_states = fast_states
        return reduced_sys, fast_subsystem

    def solve_conservation_laws(self, laws):
        pass
    
    def solve_approximations(self):
        pass

    def reduce_Cx(self):
        results_dict = {}
        possible_reductions = self.get_all_combinations()
        x_sol = utils.get_ode_solutions(self.get_system(), self.timepoints_ode)
        self.x_sol = x_sol
        full_ssm = utils.get_SSM(self.get_system(), self.timepoints_ssm)
        self.full_ssm = full_ssm
        if not len(possible_reductions):
            print('No possible reduced models found. Try increasing tolerance for number of states.')
            return
        for attempt in possible_reductions: 
            # Create reduced systems
            reduced_sys, fast_subsystem = self.solve_timescale_separation(attempt)
            # Get metrics for this reduced system
            try:
                e = self.get_error_metric(reduced_sys)
            except:
                e = np.NaN
                continue
            try:
                Se = self.get_robustness_metric(reduced_sys, fast_subsystem, attempt)
            except:
                Se = np.NaN
                continue
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
            self.results_dict = self.reduce_Cx()
            return self.results_dict
        else:
            print('Using general model reduction algorithm with inputs and nonlinear output relationship')
            self.results_dict = self.reduce_general()
            return self.results_dict

    def get_system(self):
        return System(self.x, self.f, self.params, self.C, self.g,
                    self.h, self.params_values, self.x_init)


class ReduceUtils(Reduce):
    '''
    For various utility methods developed on top of Reduce class and other utility functions
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
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
        return f1

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


def create_system(x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = []):
    return System(x, f, params, C, g, h, 
                params_values, x_init)

