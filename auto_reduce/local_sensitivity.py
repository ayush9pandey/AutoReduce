from .system import System
from auto_reduce import utils
import numpy as np
# from .ode import ODE
from scipy.integrate import solve_ivp, odeint
from sympy import lambdify

class SSM(System):
    '''
    Class that computes local sensitivity analysis coefficients for the given Model using a numerical 
    approximation method discussed in doi: https://doi.org/10.1016/0021-9991(76)90007-3
    Uses numerical method to find sensitivity analysis matrix (SSM).
    Both the Jacobian matrix and the Z matrix are estimated using 4th
    order central difference as given in the paper.
    '''
    def __init__(self, x, f, params = None, C = None, g = None, h = None, u = None,
                params_values = [], x_init = [], timepoints = None):
        super().__init__(x, f, params, C, g, h, u, params_values, x_init)
        if timepoints is None:
            timepoints = []
        else:
            self.timepoints = timepoints

        return

    def compute_Zj(self, x, j, **kwargs):
        '''
        Compute Z_j, i.e. df/dp_j at a particular timepoint k for the parameter p_j. 
        Returns a vector of size n x 1. 
        Use mode = 'accurate' for this object attribute to use accurate computations using numdifftools.
        '''
        # if 'mode' in kwargs:
        #     if kwargs.get('mode') == 'accurate':
        #         del kwargs['mode']
        #         return self.sensitivity_to_parameter(x, j, **kwargs)
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
                if Z[i] == np.Inf:
                    Z[i] = 1
                elif Z[i] == np.NaN:
                    Z[i] = 0

        return Z

    def compute_J(self, x, **kwargs):
        '''
        Compute the Jacobian J = df/dx at a timepoint k.
        Returns a matrix of size n x n.
        Use mode = 'accurate' for this object attribute to use accurate computations using numdifftools.
        '''
        if 'fun' in kwargs:
            fun = kwargs.get('fun')
        else:
            fun = self.f
        if 'var' in kwargs:
            var = kwargs.get('var')
        else:
            var = x
        # initialize J
        J = np.zeros( (self.n, len(var)) )   
        P = self.params_values 
        u = self.u
        if 'mode' in kwargs:
            if kwargs.get('mode') == 'accurate':
                del kwargs['mode']
                try:
                    import numdifftools as nd
                except:
                    raise ValueError('The package numdifftools is not installed for this method to work.')
                fun_l = lambdify((self.x, self.params), fun)
                def fun_ode(t, x, params):
                    y = fun_l(x, params)
                    return np.array(y)
                jfun = nd.Jacobian(lambda x: fun_ode(0, x, P), **kwargs)
                return jfun(x)
        # store the variable with respect to which we approximate the differentiation (df/dvar)
        X = var 
        for i in range(self.n):
            for j in range(len(var)):
                F = np.zeros( (4,1) )
                h = X[j]*0.01
                # Gets O(4) central difference on dfi/dvarj
                if h != 0:
                    var = X
                    var[j] = X[j] + 2*h
                    f = self.evaluate(fun, var, P, u)
                    F[0] = f[i]
                    var[j] = X[j] + h
                    f = self.evaluate(fun, var, P, u)
                    F[1] = f[i]
                    var[j] = X[j] - h
                    f = self.evaluate(fun, var, P, u)
                    F[2] = f[i]
                    var[j] = X[j] - 2*h
                    f = self.evaluate(fun, var, P, u)
                    F[3] = f[i]
                    #Store approvar. dfi/dvarj into J
                    J[i,j]= (-F[0] + 8*F[1] - 8*F[2] + F[3])/(12*h)   
                    # print(J[i,j])
                    # if J[i,j] == np.Inf:
                    #     J[i,j] = 1
                    # elif J[i,j] == np.NaN:
                    #     J[i,j] = 0
        return J

    def compute_SSM(self, normalize = False, **kwargs):
        '''
        Returns the sensitivity coefficients S_j for each parameter p_j. 
        The sensitivity coefficients are written in a sensitivity matrix SSM of size len(timepoints) x len(params) x n
        If normalize argument is true, the coefficients are normalized by the nominal value of each paramneter.
        Use mode = 'accurate' for this object attribute to use accurate computations using numdifftools.
        '''
        if 'mode' in kwargs:
            if kwargs.get('mode') == 'accurate_SSM':
                return self.solve_extended_ode(**kwargs)

        def sens_func(t, x, J, Z):
            # forms ODE to solve for sensitivity coefficient S
            dsdt = J@x + Z
            return dsdt
        P = self.params_values
        S0 = np.zeros(self.n) # Initial value for S_i  
        SSM = np.zeros( (len(self.timepoints), len(P), self.n) )
        # solve for all x's in timeframe set by timepoints
        system_obj = self.get_system()
        sol = utils.get_ODE(system_obj, self.timepoints).solve_system().T
        xs = sol
        xs = np.reshape(xs,(len(self.timepoints), self.n))
        self.xs = xs
        # Solve for SSM at each time point 
        for k in range(len(self.timepoints)): 
            # print('for timepoint',self.timepoints[k])
            timepoints = self.timepoints[0:k+1]
            if len(timepoints) == 1:
                continue
            # get the jacobian matrix
            J = self.compute_J(xs[k,:], **kwargs)
            #Solve for S = dx/dp for all x and all P (or theta, the parameters) at time point k
            for j in range(len(P)): 
                utils.printProgressBar(int(j + k*len(P)), len(self.timepoints)*len(P) - 1, prefix = 'SSM Progress:', suffix = 'Complete', length = 50)
                # print('for parameter',P[j])
                # get the pmatrix
                Zj = self.compute_Zj(xs[k,:], j, **kwargs)
                # solve for S
                sens_func_ode = lambda t, x : sens_func(t, x, J, Zj)
                sol = odeint(sens_func_ode, S0, timepoints, tfirst = True)
                S = sol
                S = np.reshape(S, (len(timepoints), self.n))
                SSM[k,j,:] = S[k,:]
        self.SSM = SSM
        if normalize:
            SSM = self.normalize_SSM() #Identifiablity was estimated using an normalized SSM
        return SSM

    def normalize_SSM(self):
        '''
        Returns normalized sensitivity coefficients. 
        Multiplies each sensitivity coefficient with the corresponding parameter p_j
        Divides the result by the corresponding state to obtain the normalized coefficient that is returned.
        '''
        SSM_normalized = np.zeros(np.shape(self.SSM))
        for j in range(len(self.params_values)):
            for i in range(self.n):
                SSM_normalized[:,j,i] = np.divide(self.SSM[:,j,i]*self.params_values[j], self.xs[:,i]) 
        self.SSM_normalized = SSM_normalized
        return SSM_normalized

    def get_system(self):
        return System(self.x, self.f, self.params, self.C, self.g,
                    self.h, self.u, self.params_values, self.x_init)
                    


    ############## Sam Clamons ###########

    def sensitivity_to_parameter(self, x, j, **kwargs):
        '''
        Calculates the response of each derivative (defined by ode) to changes
        in a single parameter (the jth one) at point x.

        keyword argument options?:
            ode_sol - An OdeSolution object holding a continuously-interpolated
                        solution for ode.
            ode_jac - Jacobian of the ode, as calculated by numdifftools.Jacobian.
            ode - The ODE for the system, of the form ode(t, x, params)
            params - A list of parameters to feed to ode.
            p - The index of the parameter to calculate sensitivities to.
            t_min - Starting time.
            t_max - Ending time.
        returns: An OdeSolution object representing the (continously-interpolated)
                    sensitivity of each variable to the specified parameter over
                    time.
        '''
        # Build a scipy-integratable derivative-of-sensitivity function.
        import numdifftools as nd
        import copy
        def dS_dt(t, s):
            xs = ode_sol(t)
            # Wrapper to let numdifftools calculate df/dp.
            def ode_as_parameter_call(param):
                call_params = copy.deepcopy(self.params_values)
                call_params[j] = self.params_values
                return ode(t, xs, call_params)
            df_dp = lambda xs: nd.Jacobian(ode_as_parameter_call)(xs).transpose()[:,0]
            return df_dp(params[j]) + np.matmul(ode_jac(xs), s)
        sol = odeint(dS_dt, np.zeros(n_vars), self.timepoints, **kwargs)
        return sol


    ############## Sam Clamons ###########
    def solve_extended_ode(self, ode = None, params = None, t_min = None, t_max = None, init = None, method = "RK45"):
        '''
        Augments an ODE system (as a scipy-integratable function) into an ODE
        representing the original ODE plus sensitivities, then solves them all.

        The key equation here is, for a system dx/dt = f(x, p, t),

        dS_j/dt = f_j + J*S_j

        where S_j is the vector of sensitivities of xs to parameter j, f_j is the
        vector of derivatives df/dp_j, and J is the Jacobian of f w.r.t. xs.

        params:
            ode - An ode to solve, with a signature ode(t, xs, parameters).
            params - a vector of parameters around which to calculate sensitivity.
            t_min - Starting time.
            t_max - Ending time.
            init - Initial conditions for the ode.
            method - ODE solving method, passed directly to
                        scipy.integrate.odeint.

        Returns: (x_sols, sensitivities)
            x_sols - An OdeSolution object with the solution to the original ODE.
                        Shape of a solution is (n_variables, n_times)
            sensitivities - An array of OdeSolution objects of  size (n_params)
                            holding sensitivities of each variable to each
                            parameter over time, as a continuous interpolation.
                            Shape of a solution is (n_variables, n_times)
        '''
        import numdifftools as nd
        n_variables = len(init)
        n_params    = len(params)

        # Solve ODE.
        ode_func = lambda t, xs: ode(t, xs, params)
        ode_jac  = nd.Jacobian(lambda x: ode_func(0, x))
        sol = solve_ivp(ode_func, (t_min, t_max), init,
                                        method = method, dense_output = True,
                                        jac = lambda t, x: ode_jac(x))

        if sol.status != 0:
            raise ValueError("In solve_extended_ode, solve_ivp failed with "
                                " error message: " + sol.message)

        sensitivities = [None] * n_params
        for p in range(n_params):
            print("\rSolving sensitivity for parameter %d/%d       " \
                % (p+1, n_params))
            sensitivities[p] = self.sensitivity_to_parameter(sol.sol, ode_jac, ode,
                                                            params, n_variables, p,
                                                            t_min, t_max)

        return (sol.sol, sensitivities)