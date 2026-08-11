"""Conversion from PyDMD-style linear operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from sympy import Matrix, symbols

try:
    import pydmd as _pydmd  # noqa: F401
except ImportError:
    _pydmd = None

from autoreduce.system.system import System

__all__ = ["from_dmd_model", "from_linear_operator"]


def from_linear_operator(
    operator,
    state_symbols: Optional[Sequence] = None,
    *,
    params: Optional[Sequence] = None,
    params_values: Optional[Sequence] = None,
    x_init: Optional[Sequence] = None,
    output_matrix=None,
    timepoints_ode=None,
    timepoints_ssm=None,
    discrete_time=True,
) -> System:
    """Create a `System` from a linear DMD operator matrix.

    PyDMD operators are discrete-time maps by default. With
    `discrete_time=True`, the returned dynamics are `(A - I) x`, which is the
    unit-step state increment. Set `discrete_time=False` when the operator is
    already a continuous-time linear dynamics matrix.
    """
    operator_array = np.asarray(operator, dtype=object)
    if operator_array.ndim != 2 or operator_array.shape[0] != operator_array.shape[1]:
        raise ValueError("operator must be a square matrix.")

    n_states = operator_array.shape[0]
    if state_symbols is None:
        states = list(symbols(f"x0:{n_states}"))
    else:
        states = list(state_symbols)
    if len(states) != n_states:
        raise ValueError("state_symbols length must match operator dimension.")

    if discrete_time:
        operator_array = operator_array - np.eye(n_states, dtype=object)

    dynamics = list(Matrix(operator_array) * Matrix(states))
    return System(
        states,
        dynamics,
        params=[] if params is None else list(params),
        params_values=[] if params_values is None else list(params_values),
        C=output_matrix,
        x_init=None if x_init is None else list(x_init),
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
    )


def from_dmd_model(
    dmd_model,
    state_symbols: Optional[Sequence] = None,
    *,
    params: Optional[Sequence] = None,
    params_values: Optional[Sequence] = None,
    x_init: Optional[Sequence] = None,
    output_matrix=None,
    timepoints_ode=None,
    timepoints_ssm=None,
    discrete_time=True,
) -> System:
    """Create a `System` from a fitted PyDMD model."""
    if _pydmd is None:
        raise ImportError(
            "PyDMD is required for autoreduce.system.pydmd.from_dmd_model. "
            "Install it with `pip install autoreduce[dmd]`."
        )
    operator = _extract_operator(dmd_model)
    return from_linear_operator(
        operator,
        state_symbols,
        params=params,
        params_values=params_values,
        x_init=x_init,
        output_matrix=output_matrix,
        timepoints_ode=timepoints_ode,
        timepoints_ssm=timepoints_ssm,
        discrete_time=discrete_time,
    )


def _extract_operator(dmd_model):
    """Return a matrix-like operator from a fitted PyDMD object."""
    operator = getattr(dmd_model, "operator", None)
    if operator is not None:
        as_numpy_array = getattr(operator, "as_numpy_array", None)
        if as_numpy_array is not None:
            return as_numpy_array() if callable(as_numpy_array) else as_numpy_array
        for attr in ("A", "atilde", "_Atilde"):
            value = getattr(operator, attr, None)
            if value is not None:
                return value() if callable(value) else value

    for attr in ("A", "atilde", "_Atilde"):
        value = getattr(dmd_model, attr, None)
        if value is not None:
            return value() if callable(value) else value

    raise ValueError(
        "Could not find a linear operator on the PyDMD model. Fit the model "
        "first or pass an explicit operator to from_linear_operator."
    )
