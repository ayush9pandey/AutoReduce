from converters import *  
from local_sensitivity_analysis import *

# Import required libraries and dependencies
from sympy import *
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import sympy
from scipy.integrate import odeint, solve_ivp

import scipy

def solve_system(ode_model, params, timepoints, init):
    guesses = np.array(params)
    t_min = timepoints[0]
    t_max = timepoints[-1]
    x_sol, sensitivity_sol = solve_extended_ode(ode_model, guesses,
                                                t_min = t_min, t_max = t_max,
                                                init = init,
                                                method = "RK45")
    return x_sol, sensitivity_sol


class SBMLReduce(object):
    def __init__(self):
        self.sbml_doc = None
        self.sympy_model = None
        self.ode_model = None
        self.x = [] # Should be a list of Sympy objects
        n = len(self.x)
        self.f = [] # Should be a list of Sympy objects
        self.f_hat = [] # Should be a list of Sympy objects
        self.P = [] # Should be a list of Sympy objects
        self.params_values = np.zeros(len(self.P))
        self.timepoints = np.linspace(0, 10, 100)
        self.nstates_tol = n - 1
        self.error_tol = 1e6
        self.x_init = np.zeros(n)
        self.C = np.zeros((1,n), dtype = int).tolist()
        self.results_dict = {}
        return
    
    def load_SBML_model(self, filename):
        return

    def load_ODE_model(self, n_states, n_params = 0):
        x, f, P = ode_to_sympy(n_states, n_params)
        self.x = x
        self.f = f
        self.P = P
        return self.x, self.f, self.P

    def load_Sympy_model(self, sympy_model):
        return

    def compute_reduced_model(self):
        x_sol, y, S_final = self.compute_full_model()
        C = self.C
        x = self.x
        f = self.f
        params = self.P
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
        P = self.P
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