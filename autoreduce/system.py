
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
    def __init__(self, x, f, params = None, C = None, g = None, h = None, u = None,
                params_values = [], x_init = [], input_values = []):
        """
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

        u : List of inputs

        x_init : Model initial condition
        """

        self.x = x
        self.n = len(x)
        self.f = f
        self.params = params
        self.C = C
        self.g = g
        self.h = h
        self.u = u
        self.params_values = params_values
        self.input_values = input_values
        self.x_init = x_init
        return

    def set_dynamics(self, f = None, g = None, h = None, C = None, u = None, params = []):
        """
        Set either f, g, h, or C to the System object or parameter values using P.
        """
        if f:
            self.f = f
        if g:
            self.g = g
        if h:
            self.h = h
        if C:
            self.C = C
        if u:
            self.u = u
        if params:
            self.params = params
        return self

    def evaluate(self, f, x, P, u = None):
        """
        Evaluate the given symbolic function (f) that is part of the System
        at the values given by x for self.x and P for self.params
        """
        fs = []
        for i in range(len(f)):
            fi = f[i]
            fi = fi.subs(list(zip(self.x, x)))
            if self.u is not None:
                fi = fi.subs(list(zip(self.u, u)))
            fi = fi.subs(list(zip(self.params, P)))
            fs.append(fi)
        return fs

    def set_parameters(self, params_values = [], x_init = []):
        """
        Set model parameters and initial conditions
        """
        f_new = []
        if params_values:
            self.params_values = [pi for pi in params_values]
            if self.params:
                for fi in self.f:
                    f_new.append(fi.subs(list(zip(self.params, self.params_values))))
        if x_init:
            self.x_init = [pi for pi in x_init]
        self.f = f_new
        return f_new

    def load_SBML_model(self, filename):
        raise NotImplementedError


    def load_Sympy_model(self, sympy_model):
        raise NotImplementedError