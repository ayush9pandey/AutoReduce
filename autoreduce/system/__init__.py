"""System model representations and import adapters."""

from autoreduce.system.pydmd import from_dmd_model, from_linear_operator
from autoreduce.system.system import System

__all__ = ["System", "from_dmd_model", "from_linear_operator"]
