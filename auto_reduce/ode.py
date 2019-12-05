from .system import System
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint 
import time

class ODE(System):
    '''
    To solve the Model using scipy.odeint
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
        Other options passed to scipy.integrate.odeint.
        '''
        fun = lambdify((self.x, self.params), self.f)
        def fun_ode(t, x, params):
            y = fun(x, params)
            return np.array(y)

        if 'timing' in kwargs:
            timing = kwargs.get('timing')
            del kwargs['timing']
        else:
            timing = False
        start_time = time.time()
        sol = odeint(lambda t, x : fun_ode(t, x, self.params_values), self.x_init, self.timepoints,
                       tfirst = True, **kwargs)

        end_time = time.time()
        if timing:
            print('Time taken for this ODE call : ', end_time - start_time)
        self.sol = sol
        return sol

    def get_system(self):
        return System(self.x, self.f, self.params, self.C, self.g,
                    self.h, self.params_values, self.x_init)
