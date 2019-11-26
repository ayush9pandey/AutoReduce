from system import System
import numpy as np
from sympy import lambdify
from scipy.integrate import solve_ivp

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

    def solve_system(self, **kwargs):
        '''
        Solve the System ODE for the given timepoints and initial conditions set for System. 
        Other options passed to scipy.integrate.solve_ivp.
        '''
        fun = lambdify((self.x, self.params), self.f)
        def fun_ode(t, x, params):
            y = fun(x, params)
            return np.array(y)

        t_min = self.timepoints[0]
        t_max = self.timepoints[-1]
        sol = solve_ivp(lambda t, x :fun_ode(t, x, self.params_values), (t_min, t_max), self.x_init,
                        t_eval = self.timepoints, **kwargs)
        self.sol = sol
        return sol

