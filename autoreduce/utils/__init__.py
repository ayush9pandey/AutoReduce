"""Utility modules for conversion, SBML export, and reduction setup."""

_EXPORTS = {
    "get_ODE": ("autoreduce.utils.reduction", "get_ODE"),
    "get_SSM": ("autoreduce.utils.reduction", "get_SSM"),
    "get_ode_solutions": (
        "autoreduce.utils.reduction",
        "get_ode_solutions",
    ),
    "get_reducible": ("autoreduce.utils.reduction", "get_reducible"),
    "load_ODE_model": ("autoreduce.utils.converters", "load_ODE_model"),
    "load_sbml": ("autoreduce.utils.converters", "load_sbml"),
    "ode_to_sympy": ("autoreduce.utils.converters", "ode_to_sympy"),
    "reduce_utils": ("autoreduce.utils.reduction", "reduce_utils"),
    "solve_ODE_SSM": ("autoreduce.utils.reduction", "solve_ODE_SSM"),
    "solve_ode": ("autoreduce.utils.reduction", "solve_ode"),
    "solve_sensitivity": (
        "autoreduce.utils.reduction",
        "solve_sensitivity",
    ),
    "solve_ssm": ("autoreduce.utils.reduction", "solve_ssm"),
    "solve_timescale_separation": (
        "autoreduce.utils.reduction",
        "solve_timescale_separation",
    ),
    "sympy_to_sbml": ("autoreduce.utils.converters", "sympy_to_sbml"),
}

__all__ = [
    "get_ODE",
    "get_SSM",
    "get_ode_solutions",
    "get_reducible",
    "load_ODE_model",
    "load_sbml",
    "ode_to_sympy",
    "reduce_utils",
    "solve_ODE_SSM",
    "solve_ode",
    "solve_sensitivity",
    "solve_ssm",
    "solve_timescale_separation",
    "sympy_to_sbml",
]


def __getattr__(name):
    """Lazily load utility exports to avoid import cycles."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'autoreduce.utils' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
