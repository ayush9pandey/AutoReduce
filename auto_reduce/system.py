
# Import required libraries and dependencies
from .converters import ode_to_sympy
from sympy import *
import numpy as np

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

    
### TRASH ####
      #     error_norm.append(e)
        #     reduced_models.append(f_hat)
        #     retained_states.append(attempt)
        # Sensitivity of error for each reduced model
        #     S_hat = solve_system(fun_hat_ode)
            # Normalize all sensitivities
            # Compute total sensitivity
        #     S_hat_final = []
        #     for i in range(len(params_values)): 
        #         # for sensitivity of all states wrt each parameter
        #         sen_sol = S_hat[i]
        #         sens_i = np.abs(np.array(sen_sol(timepoints)))
        #         sens_i = sens_i.transpose()
        #         normed_sens = sens_i.copy()
        #         for j in range(len(timepoints)):
        #             for k in range(np.shape(x_sols_hat)[0]):
        #                 normed_sens[j,k] = normed_sens[j,k] * params_values[i] / x_sols_hat[k,j] 
        #         S_hat_final.append(normed_sens)

        #     all_Ses = []
        # #     Se = []
        #     total_sensitivity = np.zeros(len(outputs))
        #     S_bar = np.concatenate( (S_final,S_hat_final), axis = 2 )
        #     C_bar = np.concatenate( (C, -1*C_hat), axis = 1)
        #     for i in range(np.shape(S_bar)[0]):
        #         S_bar_p = S_bar[i,:,:]
        #     #     print(np.shape(S_bar_p))
        #         Se = np.matmul(C_bar, np.transpose(S_bar_p))
        #         Se = np.ma.array(Se, mask = np.isnan(Se))
        #     #     print(np.shape(Se))
        #         for l in range(np.shape(Se)[0]):
        #             total_sensitivity[l] += np.sqrt(np.sum(Se[l,:]**2))
        #         all_Ses.append(Se)
        #     print('Successful reduction by retaining states - {0}'.format(attempt))
        #     print('The norm of the error for the reduced model is {0:0.3f}'.format(e))
        #     print('The norm of the sensitivity of the error for the reduced model is {0}'.format(total_sensitivity))
        #     results_dict[str(attempt)] = []
        #     results_dict[str(attempt)].append(f_hat)
        #     results_dict[str(attempt)].append(e)
        #     results_dict[str(attempt)].append(total_sensitivity)
        #     results_dict[str(attempt)].append(x_sol)
        #     results_dict[str(attempt)].append(x_sol_hat)
        #     results_dict[str(attempt)].append(all_Ses)
   