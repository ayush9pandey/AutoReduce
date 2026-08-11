"""Core symbolic representation of nonlinear dynamical systems."""

from warnings import warn

import libsbml
import numpy as np
from sympy import Symbol
from sympy.printing import latex

from autoreduce.utils.sbml import (
    add_parameters,
    add_reaction,
    add_species,
    create_sbml_model,
)


def _values_equal(left, right):
    """Return value equality for scalars, lists, tuples, and arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return left == right


class System(object):
    """
    Class that stores the system model in this form:
    x_dot = f(x, theta), y = Cx.
    """

    def __init__(
        self,
        x,
        f,
        params=None,
        C=None,
        g=None,
        h=None,
        u=None,
        params_values=None,
        x_init=None,
        input_values=None,
        timepoints_ode=None,
        timepoints_ssm=None,
        params_dict=None,
        **kwargs,
    ):
        """
        The general system dynamics :
        x_dot = f(x, P) + g(x, P)u, y = h(x,P)
        Use the utility function ode_to_sympy to write these.

        x : (Symbolic) state variable vector

        f : The system model dynamics.
            Written symbolically with symbols x = [x_0, x_1, ...]
            for states and P = [P_0, P_1, ...] for parameters.

        params_dict : Dictionary mapping symbolic parameters to numerical
            values. Use params_dict to set, get, and update parameters.

        g : The actuator / input dynamics.
            None by default if the system is autonomous.

        C : The output matrix for y = Cx,
            size of C must be #outputs times #states. If None,
            the argument h is expected. Cannot set C and h both.

        h : The output description y = h(x, P)
            where x are states and P are parameters.

        u : List of inputs

        x_init : Model initial conditions

        Parameter values can be read and changed with get_param,
        set_param, set_param_dict, and update_param_dict.
        """

        if C is not None and h is not None:
            raise ValueError("Set either C or h, not both.")
        if len(x) != len(f):
            raise ValueError("x and f must have the same length.")
        if params_dict is not None and (
            params is not None or params_values is not None
        ):
            raise ValueError(
                "Use either params_dict or params/params_values, not both."
            )
        if params is not None and params_values is not None:
            if len(params) != len(params_values):
                raise ValueError(
                    "params and params_values must have the same length."
                )
        if x_init is not None and len(x_init) != len(x):
            raise ValueError("x_init must have the same length as x.")

        self.x = x
        self.n = len(x)
        self.f = f
        self.C = C
        self.g = g
        self.h = h
        self.u = u
        if params_dict is not None:
            self.params_dict = dict(params_dict)
            self.params = list(self.params_dict.keys())
            self.params_values = list(self.params_dict.values())
        else:
            self.params = params
            if params_values is not None:
                self.params_values = params_values
            else:
                self.params_values = []
            if self.params is None:
                self.params_dict = {}
            else:
                self.params_dict = dict(zip(self.params, self.params_values))
        if input_values is not None:
            self.input_values = input_values
        else:
            self.input_values = []
        self.timepoints_ode = timepoints_ode
        self.timepoints_ssm = timepoints_ssm
        if x_init is not None:
            self.x_init = x_init
        else:
            self.x_init = []
        if "parameter_dependent_ic" in kwargs:
            self.parameter_dependent_ic = kwargs.get("parameter_dependent_ic")
        else:
            self.parameter_dependent_ic = False
        if "ic_parameters" in kwargs and kwargs.get("ic_parameters"):
            self.ic_parameters = kwargs.get("ic_parameters")
            if self.parameter_dependent_ic is False:
                raise ValueError(
                    "Make sure to set parameter_dependent_ic \
                    argument to True to use parameters as initial conditions"
                )
            if self.x_init == []:
                self.x_init = [] * len(self.ic_parameters)
            elif isinstance(self.x_init, np.ndarray):
                self.x_init = list(self.x_init)
            for ic_p, ic_i in zip(self.ic_parameters, range(len(self.x_init))):
                self.set_ic_from_params(self.x_init, ic_p, ic_i)
        else:
            self.ic_parameters = None
        return

    def get_param(self, param_name):
        """Get one parameter value from params_dict."""
        if param_name not in self.params_dict:
            raise ValueError(
                f"Parameter {param_name!r} was not found. "
                f"Available parameters are: {list(self.params_dict.keys())}."
            )
        return self.params_dict[param_name]

    def set_param(self, param_name, param_value):
        """Set one parameter value in params_dict and params_values."""
        if param_name not in self.params_dict:
            raise ValueError(
                f"Parameter {param_name!r} was not found. "
                f"Available parameters are: {list(self.params_dict.keys())}."
            )
        self.params_dict[param_name] = param_value
        param_index = self.params.index(param_name)
        self.params_values[param_index] = param_value
        return param_value

    def set_param_dict(self, params_dict):
        """Set parameter values from a dictionary."""
        unknown_params = [
            param_name
            for param_name in params_dict
            if param_name not in self.params_dict
        ]
        if unknown_params:
            raise ValueError(
                f"Parameters {unknown_params!r} were not found. "
                f"Available parameters are: {list(self.params_dict.keys())}."
            )
        self.params_dict.update(params_dict)
        self.update_param_dict()
        return self.params_dict

    def update_param_dict(self):
        """Update params_values from params_dict."""
        params_dict_keys = set(self.params_dict.keys())
        params_keys = set([] if self.params is None else self.params)
        if params_dict_keys != params_keys:
            raise ValueError(
                "params_dict keys must exactly match params. "
                f"params_dict keys are {list(self.params_dict.keys())}; "
                f"params are {self.params}."
            )
        self.params_values = [self.params_dict[param] for param in self.params]
        return self.params_dict

    def _count_inputs(self):
        """Return the number of declared inputs, if any."""
        if self.u is None:
            return 0
        if isinstance(self.u, (list, tuple, np.ndarray)):
            return len(self.u)
        return 1

    def _count_outputs(self):
        """Return the number of declared outputs, if any."""
        if self.C is not None:
            shape = np.shape(self.C)
            if len(shape) == 0:
                return 1
            if len(shape) == 1:
                return 1
            return shape[0]
        if self.h is not None:
            if isinstance(self.h, (list, tuple, np.ndarray)):
                return len(self.h)
            return 1
        return 0

    @staticmethod
    def _pluralize(count, singular, plural=None):
        """Return a count-aware label."""
        if plural is None:
            plural = singular + "s"
        label = singular if count == 1 else plural
        return f"{count} {label}"

    def _pretty_print_text(self):
        """Build the human-readable system summary used by pretty_print."""
        parts = [
            "AutoReduce System object with "
            + self._pluralize(self.n, "state variable")
        ]
        input_count = self._count_inputs()
        if input_count:
            parts.append(self._pluralize(input_count, "input"))
        output_count = self._count_outputs()
        if output_count:
            parts.append(self._pluralize(output_count, "output"))
        summary = ", ".join(parts) + "."
        return f"{summary}\nsystem equations:\n{latex(self.f)}"

    def pretty_print(self):
        """Print a concise model summary with LaTeX system equations."""
        print(self._pretty_print_text())

    def __str__(self):
        """Return the symbolic dynamics for concise string display."""
        return str(self.f)

    def set_dynamics(
        self, f=None, g=None, h=None, C=None, u=None, params=None
    ):
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
        if "set_params_as" in kwargs:
            set_params_as = kwargs["set_params_as"]
        else:
            set_params_as = None
        fs = []
        for i, _ in enumerate(f):
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
                    f_new.append(
                        fi.subs(list(zip(self.params, self.params_values)))
                    )
        else:
            self.params_values = []
        if self.params is None:
            self.params_dict = {}
        else:
            self.params_dict = dict(zip(self.params, self.params_values))
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
            raise ValueError(
                "Sympy Symbol or float expected in ic_parameters\
                             argument."
            )

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
        for species, ic, ode_i, r_id in zip(
            states, states_ic, self.f, all_rxn_ids
        ):
            # Create species and initial conditions
            model = add_species(model, species, ic)
            # Reactions, for all species, s_i, --> s_i with
            # rate equal to the ODE term
            model = add_reaction(
                model,
                species=species,
                kinetic_law=ode_i,
                reaction_id=r_id,
                all_species=states,
            )
        model = add_parameters(
            model, all_parameters=params, all_values=self.params_values
        )
        if document.getNumErrors() and show_warnings:
            warn(
                "SBML model generated has errors."
                "Use document.getErrorLog() to print all errors."
            )
        return document, model

    def write_sbml(self, filename: str, **kwargs):
        """
        Writes an SBML file for AutoReduce System model.
        :param filename: String of file name to write SBML model to
        returns:
        :return SBMLDocument: libSBML SBMLDocument object
        """
        document, model = create_sbml_model(**kwargs)
        if "show_warnings" in kwargs:
            show_warnings = kwargs.get("show_warnings")
        else:
            show_warnings = True
        document, _ = self.generate_sbml_model(
            show_warnings=show_warnings, **kwargs
        )
        sbml_string = libsbml.writeSBMLToString(document)
        with open(filename, "w") as f:
            f.write(sbml_string)
        return document

    def __eq__(self, other):
        """
        Compare two System objects for equality.
        Two systems are equal if they have the same:
        - states (x)
        - dynamics (f)
        - parameters (params)
        - parameter values (params_values)
        - initial conditions (x_init)
        - output matrix (C)
        - input dynamics (g)
        - output description (h)
        - inputs (u)
        - input values (input_values)
        """
        if not isinstance(other, System):
            return False

        # Compare basic attributes
        attributes = (
            "x",
            "f",
            "params",
            "params_values",
            "x_init",
            "C",
            "g",
            "h",
            "u",
            "input_values",
        )
        return all(
            _values_equal(getattr(self, attr), getattr(other, attr))
            for attr in attributes
        )

    def __hash__(self):
        """Hash for System to use as dict key

        Returns:
            Object id: System object ID.
        """
        return id(self)
