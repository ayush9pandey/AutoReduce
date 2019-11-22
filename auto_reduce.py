from converters import *  
from local_sensitivity_analysis import *

# Import required libraries and dependencies
from sympy import *
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import sympy
from scipy.integrate import odeint, solve_ivp

import scipy

def load_ODE_model(n_states, n_params = 0):
    x, f, P = ode_to_sympy(n_states, n_params)
    return x, f, P

class System(object):
    '''
    Class that stores the system model in this form:  x_dot = f(x, theta), y = Cx.
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = []):
        '''
        The general system dynamics : x_dot = f(x, P) + g(x, P)u, y = h(x,P)
        Use the utility function ode_to_sympy to write these.

        x : (Symbolic) state variable vector

        f : The system model dynamics. Writted symbolically with symbols x = [x_0, x_1, ...]
        for states and P = [P_0, P_1, ...] for parameters. 

        params : (Symbolic) parameters used to define f, g, h. None if no symbolic parameters.

        g : The actuator / input dynamics. None by default if the system is autonomous. 
        
        C : The output matrix for y = Cx, size of C must be #outputs times #states. If None,
        the argument h is expected. Cannot set C and h both.

        h : The output description y = h(x, P) where x are states and P are parameters. 
        params_values : Values for model parameters

        x_init : Model initial condition 
        '''
        self.x = x
        self.n = len(x)
        self.f = f
        self.params = params
        self.C = C
        self.g = g
        self.h = h
        self.params_values = params_values
        self.x_init = x_init
        return

    def set_dynamics(self, f = None, g = None, h = None, C = None, P = []):
        '''
        Set either f, g, h, or C to the System object or parameter values using P.
        '''
        if f:
            self.f = f
        if g:
            self.g = g
        if h:
            self.h = h
        if C:
            self.C = C
        return self

    def evaluate(self, f, x, P):
        '''
        Evaluate the given symbolic function (f) that is part of the System
        at the values given by x for self.x and P for self.params
        '''
        fs = []
        for i in range(len(f)):
            fi = f[i]
            fi = fi.subs(list(zip(self.x, x)))
            fi = fi.subs(list(zip(self.params, P)))
            fs.append(fi)
        return fs

    def set_parameters(self, P = [], x_init = []):
        '''
        Set model parameters and initial conditions
        '''
        if P:
            self.params_values = [pi for pi in P]
            if self.params:
                for fi in self.f:
                    fi = fi.subs(list(zip(self.params, self.params_values)))
        if x_init:
            self.x_init = [pi for pi in x_init]

    def load_SBML_model(self, filename):
        raise NotImplementedError


    def load_Sympy_model(self, sympy_model):
        raise NotImplementedError

class ODE(System):
    '''
    To solve the Model using scipy.solve_ivp
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = [], timepoints = None):
        super().__init__(x, f, params, C, g, h, params_values, x_init)
        if timepoints is None:
            timepoints = []
        else:
            self.timepoints = timepoints
        return
    def solve_system(self, method = 'RK45', dense_output = False):
        # self.timepoints = timepoints
        fun = lambdify((self.x, self.params), self.f)
        def fun_ode(t, x, params):
            y = fun(x, params)
            return np.array(y)

        t_min = self.timepoints[0]
        t_max = self.timepoints[-1]
        sol = solve_ivp(lambda t, x :fun_ode(t, x, self.params_values), (t_min, t_max), self.x_init,
                        method = method, dense_output = dense_output, t_eval = self.timepoints)
        self.sol = sol
        return sol


class SSM(System):
    '''
    Class that computes local sensitivity analysis coefficients for the given Model using a numerical 
    approximation method discussed in doi: https://doi.org/10.1016/0021-9991(76)90007-3
    Uses numerical method to find sensitivity analysis matrix (SSM).
    Both the Jacobian matrix and the Z matrix are estimated using 4th
    order central difference as given in the paper.
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, 
                params_values = [], x_init = [], timepoints = None):
        super().__init__(x, f, params, C, g, h, params_values, x_init)
        if timepoints is None:
            timepoints = []
        else:
            self.timepoints = timepoints
        return

    def compute_Zj(self, x, j):
        '''
        Compute Z_j, i.e. df/dp_j at a particular timepoint k for the parameter p_j. 
        Returns a vector of size n x 1. 
        '''
        # initialize Z
        Z = np.zeros(self.n)    
        P_holder = self.params_values
        # For each state
        for i in range(self.n):
            P = P_holder
            F = np.zeros( (4,1) ) # For 4th order difference
            h = P[j]*0.01 # Small parameter for this parameter
            # Gets O(4) central difference on dfi/dpj
            if h != 0:
                P[j] = P_holder[j] + 2*h
                f = self.evaluate(self.f, x, P)
                F[0] = f[i]
                P[j] = P_holder[j] + h
                f = self.evaluate(self.f, x, P)
                F[1] = f[i]
                P[j] = P_holder[j] - h
                f = self.evaluate(self.f, x, P)
                F[2] = f[i]
                P[j] = P_holder[j] - 2*h
                f = self.evaluate(self.f, x, P)
                F[3] = f[i]
                #Store approx. dfi/dpj into Z
                Z[i] = (-F[0] + 8*F[1] - 8*F[2] + F[3])/(12*h)   
        return Z

    def compute_J(self, x):
        '''
        Compute the Jacobian J = df/dx at a timepoint k.
        Returns a matrix of size n x n.
        '''
        # initialize J
        J = np.zeros( (self.n, self.n) )   
        P = self.params_values 
        # store x
        X = x 
        for i in range(self.n):
            for j in range(self.n):
                F = np.zeros( (4,1) )
                h = X[j]*0.01
                # Gets O(4) central difference on dfi/dxj
                if h != 0:
                    x = X
                    x[j] = X[j] + 2*h
                    f = self.evaluate(self.f, x, P)
                    F[0] = f[i]
                    x[j] = X[j] + h
                    f = self.evaluate(self.f, x, P)
                    F[1] = f[i]
                    x[j] = X[j] - h
                    f = self.evaluate(self.f, x, P)
                    F[2] = f[i]
                    x[j] = X[j] - 2*h
                    f = self.evaluate(self.f, x, P)
                    F[3] = f[i]
                    #Store approx. dfi/dxj into J
                    J[i,j]= (-F[0] + 8*F[1] - 8*F[2] + F[3])/(12*h)   
        return J

    def compute_SSM(self, normalize = False):
        '''
        Returns the sensitivity coefficients S_j for each parameter p_j. 
        The sensitivity coefficients are written in a sensitivity matrix SSM of size len(timepoints) x len(params) x n
        If normalize argument is true, the coefficients are normalized by the nominal value of each paramneter.
        '''
        def sens_func(t, x, J, Z):
            # forms ODE to solve for sensitivity coefficient S
            dsdt = J@x + Z
            return dsdt
        P = self.params_values
        S0 = np.zeros(self.n) # Initial value for S_i  
        SSM = np.zeros( (len(self.timepoints), len(P), self.n) )
        # solve for all x's in timeframe set by timepoints
        sol = ODE(self.x, self.f, params = self.params, params_values = self.params_values,
                C = self.C, g = self.g, h = self.h, x_init = self.x_init, timepoints = self.timepoints).solve_system()
        xs = sol.y
        xs = np.reshape(xs,(len(self.timepoints), self.n))
        self.xs = xs
        # Solve for SSM at each time point 
        for k in range(len(self.timepoints)): 
            timepoints = self.timepoints[0:k+1]
            if len(timepoints) == 1:
                continue
            t_span = (timepoints[0], timepoints[-1])
            # get the jacobian matrix
            J = self.compute_J(xs[k,:])
            #Solve for S = dx/dp for all x and all P (or theta, the parameters) at time point k
            for j in range(len(P)): 
                # get the pmatrix
                Zj = self.compute_Zj(xs[k,:], j)
                # solve for S
                sens_func_ode = lambda t, x : sens_func(t, x, J, Zj)
                sol = solve_ivp(sens_func_ode, t_span, S0, t_eval = timepoints)
                S = sol.y
                S = np.reshape(S, (len(timepoints), self.n))
                SSM[k,j,:] = S[k,:]
        self.SSM = SSM
        if normalize:
            SSM = self.normalize_SSM() #Identifiablity was estimated using an normalized SSM
        return SSM

    def normalize_SSM(self):
        SSM_normalized = np.zeros(np.shape(self.SSM))
        for j in range(len(self.params_values)):
            for i in range(self.n):
                SSM_normalized[:,j,i] = np.divide( SSM[:,j,i]*self.params_values[j], self.xs[:,i]) 
        self.SSM_normalized = SSM_normalized
        return SSM_normalized

class Reduce(System):
    '''
    The class can be used to compute the various possible reduced models for the System object 
    and then find out the best reduced model choice using doi : https://doi.org/10.1101/640276 
    '''
    def __init__(self):
        self.f_hat = [] # Should be a list of Sympy objects
        self.nstates_tol = len(self.f) - 1
        self.error_tol = 1e6
        self.results_dict = {}
        self.timepoints = []
        return

    def compute_full_model(self):
        raise NotImplementedError

    def compute_reduced_model(self):
        x_sol, y, S_final = self.compute_full_model()
        C = self.C
        x = self.x
        f = self.f
        params = self.params
        x_init = self.x_init
        timepoints = self.timepoints
        params_values = self.params_values
        n = len(x)
        nstates_tol = self.nstates_tol
        outputs = list(np.dot(np.array(C), np.transpose(np.array(x)))) # Get y = C*x
        possible_reductions = []
        results_dict = {}
        from itertools import combinations 
        for i in range(n):
            if i != n-1:
                comb = combinations(list(range(n)), i+1) 
                possible_reductions.append(list(comb))
        possible_reductions = [list(item) for sublist in possible_reductions for item in sublist]
        for attempt in possible_reductions: 
            states_attempt = [x[i] for i in attempt]
            restart = False
            for i in outputs:
                if i not in states_attempt or len(attempt) > nstates_tol:
                    restart = True
            if restart:
                continue
            x_c = []
            collapsed_states = []
            f_c = []
            f_hat = []
            x_hat_init = []
            x_c_init = []
            x_hat = []
            for i in range(n):
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
                collapsed_states.append(x_c_sub[0])
                for j in range(len(f_hat)):
                    f_hat[j] = f_hat[j].subs(x_c[i], x_c_sub[0])
                    
            for i in range(len(x_c)):
                for j in range(len(f_hat)):
                    f_hat[j] = f_hat[j].subs(x_c[i], collapsed_states[i])
            # Solve ODEs
            fun_hat = lambdify((x_hat, params),f_hat)
            
            def fun_hat_ode(t, x_hat, params):
                y = fun_hat(x_hat, params)
                return np.array(y)

        #     x_sol = odeint(fun_ode, x_init, timepoints)
            try:
                x_sol_hat, S_hat = solve_system(fun_hat_ode, params_values, timepoints, x_hat_init)
            except:
                continue
                
        #     try:
        #         x_sol_hat = odeint(fun_hat_ode, x_hat_init, timepoints)
        #     except:
        #         continue
            x_sols_hat = x_sol_hat(timepoints)   
            x_sol_c = np.zeros((len(timepoints),np.shape(x_c)[0]))
            # Get the collapsed states by substituting the solutions into the algebraic relationships obtained
            for i in range(np.shape(x_sols_hat)[0]): 
                # for each reduced variable (because collapsed variables are only 
                # functions of reduced variables, algebraically)
                for k in range(len(x_sols_hat[:,i])):
                    for j in range(len(x_c)):
                        subs_result = collapsed_states[j].subs(x_hat[i],x_sols_hat[:,i][k])
                        if  subs_result == collapsed_states[j]:
                            continue
                        elif isinstance(subs_result, sympy.Expr):
                            collapsed_states[j] = subs_result # continue substituting other variables, until you get a float
                        else:
                            x_sol_c[k][j] = collapsed_states[j].subs(x_hat[i],x_sols_hat[:,i][k])
                    
            # construct C_hat
            C_hat = np.zeros((np.shape(C)[0], np.shape(x_hat)[0]))
            li = []
            for i in range(len(x_hat)):
                if x_hat[i] in outputs:
                    li.append(i)
            for row_ind in range(np.shape(C_hat)[0]):
                C_hat[row_ind][li.pop(0)] = 1
                
            
            y_hat = np.matmul(np.array(C_hat), np.array(x_sols_hat))
                
            if np.shape(y) == np.shape(y_hat):
                e = np.linalg.norm(y - y_hat)
            else:
                raise ValueError('The output dimensions must be the same for reduced and full model. Choose C and C_hat accordingly')
            if e == 0 or np.isnan(e):
                print('The error is zero or NaN, something wrong...continuing.')
                continue
                
        #     error_norm.append(e)
        #     reduced_models.append(f_hat)
        #     retained_states.append(attempt)
        # Sensitivity of error for each reduced model
        #     S_hat = solve_system(fun_hat_ode)
            # Normalize all sensitivities
                
            S_hat_final = []
            for i in range(len(params_values)): 
                # for sensitivity of all states wrt each parameter
                sen_sol = S_hat[i]
                sens_i = np.abs(np.array(sen_sol(timepoints)))
                sens_i = sens_i.transpose()
                normed_sens = sens_i.copy()
                for j in range(len(timepoints)):
                    for k in range(np.shape(x_sols_hat)[0]):
                        normed_sens[j,k] = normed_sens[j,k] * params_values[i] / x_sols_hat[k,j] 
                S_hat_final.append(normed_sens)

            all_Ses = []
        #     Se = []
            total_sensitivity = np.zeros(len(outputs))
            S_bar = np.concatenate( (S_final,S_hat_final), axis = 2 )
            C_bar = np.concatenate( (C, -1*C_hat), axis = 1)
            for i in range(np.shape(S_bar)[0]):
                S_bar_p = S_bar[i,:,:]
            #     print(np.shape(S_bar_p))
                Se = np.matmul(C_bar, np.transpose(S_bar_p))
                Se = np.ma.array(Se, mask = np.isnan(Se))
            #     print(np.shape(Se))
                for l in range(np.shape(Se)[0]):
                    total_sensitivity[l] += np.sqrt(np.sum(Se[l,:]**2))
                all_Ses.append(Se)
        #     Jbar = 
        #     Cbar = 
        #     P = solve_lyapunov(Jbar, Cbar)
            print('Successful reduction by retaining states - {0}'.format(attempt))
            print('The norm of the error for the reduced model is {0:0.3f}'.format(e))
            print('The norm of the sensitivity of the error for the reduced model is {0}'.format(total_sensitivity))
            results_dict[str(attempt)] = []
            results_dict[str(attempt)].append(f_hat)
            results_dict[str(attempt)].append(e)
            results_dict[str(attempt)].append(total_sensitivity)
            results_dict[str(attempt)].append(x_sol)
            results_dict[str(attempt)].append(x_sol_hat)
            results_dict[str(attempt)].append(all_Ses)
            self.results_dict = results_dict
        return


class Utils(Reduce):
    '''
    For future methods developed on top of Reduce class and other utility functions
    '''
    def __init__(self):
        return 

    def get_reduced_model(self, mode):
        results_dict = self.results_dict
        error_tol = self.error_tol

        if mode == 'Sympy':
            # Post reduction
            count = 0
            reduced_models_good = []
            error_norm = []
            retained_states = []
            reduced_models = []
            for i in range(len(results_dict.keys())):
                key = list(results_dict.keys())[i]
                errors = results_dict[key][1]
                error_norm.append(errors)
                reduced_models.append(results_dict[key][0])
                flag = len(np.where(errors <= error_tol)[0].tolist())
                count += flag
                if flag:
                    reduced_models_good.append(results_dict[key][0])
                    retained_states.append(key)
            if count == 0:
                print('None of the reduced models could achieve the desired error tolerance. You could try with higher tolerance or an improved model or parameters. The model with minimum error is -')
                index_reduced = np.where(error_norm == np.min(error_norm))[0].tolist()[0]
                final_reduced = reduced_models[index_reduced]
            elif count > 1: # multiple reduced models satisfy the tolerance desired, choose one with least states
                final_reduced = reduced_models_good[0]
                index_reduced = 0
                for i in reduced_models_good:
                    if len(i) < len(final_reduced):
                        final_reduced = reduced_models[i]
                        index_reduced = i
            else:
                final_reduced = reduced_models_good[0]
            #     min_norm_ind = np.where(error_norm == np.min(error_norm))
            #     min_index_list = min_norm_ind[0].tolist()
            #     final_reduced = reduced_models[min_index_list[0]]
            #     if len(min_index_list) > 1:
            #         for i in min_index_list:
            #             if len(reduced_models[i]) < len(final_reduced):
            #                 final_reduced = reduced_models[i]
            final_retained = retained_states[index_reduced]
            print('The states retained for the final reduced model are {0}'.format(final_retained))
            print('The final reduced model is {0}'.format(final_reduced))
            return  final_reduced
        if mode == 'SBML':
            sbml_doc = sympy_to_sbml(self.f_hat)
            return sbml_doc
        if mode == 'ODE':
            fun_hat = lambdify((self.x,self.P), self.f_hat)
            return fun_hat

    def compute_full_model(self):
        x = self.x
        f = self.f
        P = self.params
        C = self.C
        x_init = self.x_init
        params_values = self.params_values
        timepoints = self.timepoints
        fun = lambdify((x, P),f)
        def fun_ode(t, x, P):
            y = fun(x, P)
            return np.array(y)
        x_sol, S = solve_system(fun_ode, params_values, timepoints, x_init)
        y = np.matmul(np.array(C), np.array(x_sol(timepoints)))
        S_final = []
        for i in range(len(params_values)): 
            # for sensitivity of all states wrt each parameter
            sen_sol = S[i]
            sens_i = np.abs(np.array(sen_sol(timepoints)))
            sens_i = sens_i.transpose()
            normed_sens = sens_i.copy()
            x_sols = x_sol(timepoints)
            for j in range(len(timepoints)):
                for k in range(np.shape(x_sols)[0]):
                    normed_sens[j,k] = normed_sens[j,k] * params_values[i] / x_sols[k,j] 
            S_final.append(normed_sens)
        return x_sol, y, S_final

  