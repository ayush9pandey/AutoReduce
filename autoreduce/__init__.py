"""AutoReduce public API and package metadata."""

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

from autoreduce.reductions.abundance import solve_approximations
from autoreduce.reductions.conservation import (
    apply_conservation_laws,
    find_conserved_sets,
    setup_conservation_laws,
    solve_conservation_laws,
)
from autoreduce.reductions.core import (
    Reduce,
    ReduceUtils,
    create_system,
    get_error_metric,
    get_robustness_metric,
)
from autoreduce.reductions.timescale import (
    explore_all_QSS_models,
    reduce_with_input,
    solve_timescale_separation,
)
from autoreduce.solvers.ode import ODE
from autoreduce.solvers.ssm import SSM
from autoreduce.solvers.utils import (
    get_ODE,
    get_ode_solutions,
    get_SSM,
    solve_ode,
    solve_ODE_SSM,
    solve_sensitivity,
    solve_ssm,
)
from autoreduce.system.system import System
from autoreduce.utils.converters import (
    load_ode_model,
    load_sbml,
    ode_to_sympy,
    sympy_to_sbml,
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
    "explore_all_QSS_models",
    "find_conserved_sets",
    "get_ODE",
    "get_SSM",
    "get_error_metric",
    "get_ode_solutions",
    "get_robustness_metric",
    "load_ode_model",
    "load_sbml",
    "ode_to_sympy",
    "reduce_with_input",
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
