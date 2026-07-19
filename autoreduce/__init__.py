"""AutoReduce public API and package metadata."""

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

from autoreduce.reductions.abundance import solve_approximations
from autoreduce.reductions.conservation import (
    apply_conservation_laws,
    find_conserved_sets,
    get_conservation_laws,
    setup_conservation_laws,
    solve_conservation_laws,
)
from autoreduce.reductions.timescale import (
    Reduce,
    ReduceUtils,
    create_system,
    solve_timescale_separation,
)
from autoreduce.solvers.ode import ODE
from autoreduce.solvers.ssm import SSM
from autoreduce.system.system import System
from autoreduce.utils.converters import (
    load_ODE_model,
    load_sbml,
    ode_to_sympy,
    sympy_to_sbml,
)
from autoreduce.utils.reduction import (
    get_ODE,
    get_SSM,
    get_ode_solutions,
    get_reducible,
    reduce_utils,
    solve_ODE_SSM,
    solve_ode,
    solve_sensitivity,
    solve_ssm,
)

__all__ = [
    "ODE",
    "Reduce",
    "ReduceUtils",
    "SSM",
    "System",
    "__version__",
    "apply_conservation_laws",
    "create_system",
    "find_conserved_sets",
    "get_ODE",
    "get_SSM",
    "get_conservation_laws",
    "get_ode_solutions",
    "get_reducible",
    "load_ODE_model",
    "load_sbml",
    "ode_to_sympy",
    "reduce_utils",
    "setup_conservation_laws",
    "solve_ODE_SSM",
    "solve_approximations",
    "solve_conservation_laws",
    "solve_ode",
    "solve_sensitivity",
    "solve_ssm",
    "solve_timescale_separation",
    "sympy_to_sbml",
]
