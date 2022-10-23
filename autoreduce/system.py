# Import required libraries and dependencies
from sympy import *
import libsbml
import numpy as np
from .sbmlutil import create_sbml_model, add_species, add_reaction
from .sbmlutil import add_parameters
from warnings import warn


class System(object):
    """
    Class that stores the system model in this form: 
    x_dot = f(x, theta), y = Cx.
    """
    def __init__(self, x, f, params=None,
                 C=None, g=None, h=None, u=None,
                 params_values=None,
                 x_init=None, input_values=None, **kwargs):
        """
        The general system dynamics : 
        x_dot = f(x, P) + g(x, P)u, y = h(x,P)
        Use the utility function ode_to_sympy to write these.

        x : (Symbolic) state variable vector

        f : The system model dynamics. 
            Written symbolically with symbols x = [x_0, x_1, ...]
            for states and P = [P_0, P_1, ...] for parameters.

        params : (Symbolic) parameters used to 
                define f, g, h. None if no symbolic parameters.

        g : The actuator / input dynamics. 
            None by default if the system is autonomous.

        C : The output matrix for y = Cx, 
            size of C must be #outputs times #states. If None,
            the argument h is expected. Cannot set C and h both.

        h : The output description y = h(x, P) 
            where x are states and P are parameters.

        params_values : Values for model parameters

        u : List of inputs

        x_init : Model initial conditions
        """

        self.x = x
        self.n = len(x)
        self.f = f
        self.params = params
        self.C = C
        self.g = g
        self.h = h
        self.u = u
        if params_values is not None:
            self.params_values = params_values
        else:
            self.params_values = []
        if input_values is not None:
            self.input_values = input_values
        else:
            self.input_values = []
        if x_init is not None:
            self.x_init = x_init
        else:
            self.x_init = []
        if 'parameter_dependent_ic' in kwargs:
            self.parameter_dependent_ic = kwargs.get('parameter_dependent_ic')
        else:
            self.parameter_dependent_ic = False
        if 'ic_parameters' in kwargs and kwargs.get('ic_parameters'):
            self.ic_parameters = kwargs.get('ic_parameters')
            if self.parameter_dependent_ic is False:
                raise ValueError('Make sure to set parameter_dependent_ic \
                    argument to True to use parameters as initial conditions')
            if self.x_init == []:
                self.x_init = []*len(self.ic_parameters)
            elif isinstance(self.x_init, np.ndarray):
                self.x_init = list(self.x_init)
            for ic_p, ic_i in zip(self.ic_parameters, range(len(self.x_init))):
                self.set_ic_from_params(self.x_init, ic_p, ic_i)
        else:
            self.ic_parameters = None
        return

    def set_dynamics(self, f=None, g=None,
                     h=None, C=None, u=None,
                     params=None):
        """
        Set either f, g, h, or C to the
        System object or parameter values using P.
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
        if params is not None:
            self.params = params
        else:
            self.params = []
        return self

    # Change this variable name P to params_values or something
    def evaluate(self, f, x, P, u=None, **kwargs):
        """
        Evaluate the given symbolic 
        function (f) that is part of the System
        at the values given by x for self.x
        and P for self.params
        """
        if 'set_params_as' in kwargs:
            set_params_as = kwargs['set_params_as']
        else:
            set_params_as = None
        fs = []
        for i in range(len(f)):
            fi = f[i]
            fi = fi.subs(list(zip(self.x, x)))
            if self.u is not None:
                fi = fi.subs(list(zip(self.u, u)))
            if not set_params_as:
                fi = fi.subs(list(zip(self.params, P)))
            fs.append(fi)
        return fs

    def set_parameters(self, params_values=None, x_init=None):
        """
        Set model parameters and initial conditions
        """
        f_new = []
        if params_values is not None:
            self.params_values = [pi for pi in params_values]
            if self.params:
                for fi in self.f:
                    f_new.append(fi.subs(list(zip(self.params,
                                 self.params_values))))
        else:
            self.params_values = []
        if x_init is not None:
            self.x_init = [pi for pi in x_init]
        else:
            self.x_init = []
        self.f = f_new
        return f_new

    def set_ic_from_params(self, x_init, ic_param, ic_index):
        """
        Set System initial conditions using parameter values
        """
        if isinstance(ic_param, Symbol):
            param_index = self.params.index(ic_param)
            value_from_params = self.params_values[param_index]
            x_init[ic_index] = value_from_params
            return value_from_params
        elif isinstance(ic_param, (int, float)):
            x_init[ic_index] = ic_param
            return ic_param
        else:
            raise ValueError('Sympy Symbol or float expected in ic_parameters\
                             argument.')

    def generate_sbml_model(self, show_warnings=True, **kwargs):
        """Creates an new SBML model and populates with the species and
        their ODE in the System object
        :param show_warnings: bool, to display warnings
        :param kwargs: extra keywords pass onto create_sbml_model()
        :return: tuple: (document,model) SBML objects
        """
        document, model = create_sbml_model(**kwargs)
        states = [str(i) for i in self.x]
        states_ic = [float(i) for i in self.x_init]
        all_rxn_ids = [f"r{i}" for i in range(len(self.f))]
        params = [str(i) for i in self.params]
        for species, ic, ode_i, r_id in zip(states, states_ic,
                                            self.f, all_rxn_ids):
            # Create species and initial conditions
            model = add_species(model, species, ic)
            # Reactions, for all species, s_i, --> s_i with
            # rate equal to the ODE term
            model = add_reaction(model, species=species,
                                 kinetic_law=ode_i,
                                 reaction_id=r_id,
                                 all_species=states)
        model = add_parameters(model, all_parameters=params,
                               all_values=self.params_values)
        if document.getNumErrors() and show_warnings:
            warn("SBML model generated has errors."
                 "Use document.getErrorLog() to print all errors.")
        return document, model

    def write_sbml(self, filename: str, **kwargs):
        """
        Writes an SBML file for AutoReduce System model.
        :param filename: String of file name to write SBML model to
        returns:
        :return SBMLDocument: libSBML SBMLDocument object
        """
        document, model = create_sbml_model(**kwargs)
        if 'show_warnings' in kwargs:
            show_warnings = kwargs.get('show_warnings')
        else:
            show_warnings = True
        document, _ = self.generate_sbml_model(show_warnings=show_warnings,
                                               **kwargs)
        sbml_string = libsbml.writeSBMLToString(document)
        with open(filename, 'w') as f:
            f.write(sbml_string)
        return document
