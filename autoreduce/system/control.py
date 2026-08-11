"""Conversion from python-control input/output systems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

try:
    import control as ct
except ImportError as exc:
    raise ImportError(
        "python-control is required for autoreduce.system.control. "
        "Install it with `pip install autoreduce[control]`."
    ) from exc

from autoreduce.system.system import System


def from_nonlinear_io_system(
    io_system: ct.NonlinearIOSystem,
    state_symbols: Sequence,
    *,
    params: Optional[Sequence] = None,
    params_values: Optional[Sequence] = None,
    input_symbols: Optional[Sequence] = None,
    input_values: Optional[Sequence] = None,
    x_init: Optional[Sequence] = None,
    output_matrix=None,
    time=0,
) -> System:
    """Create an AutoReduce `System` from a `NonlinearIOSystem`.

    Parameters
    ----------
    io_system
        python-control nonlinear input/output system.
    state_symbols
        Symbolic states used by AutoReduce.
    params
        Symbolic parameters passed to the python-control update and output
        functions by string name.
    params_values
        Numeric parameter values in the same order as `params`.
    input_symbols
        Symbolic input variables. Required when `io_system` has inputs.
    input_values
        Numeric input values in the same order as `input_symbols`.
    x_init
        Initial conditions in the same order as `state_symbols`.
    output_matrix
        Linear output matrix. Cannot be supplied when `io_system` defines a
        nonlinear output function.
    time
        Time value used when evaluating the symbolic update and output
        functions.

    Returns
    -------
    System
        AutoReduce system with symbolic dynamics from `io_system`.
    """
    if not isinstance(io_system, ct.NonlinearIOSystem):
        raise TypeError("io_system must be a control.NonlinearIOSystem.")
    if io_system.updfcn is None:
        raise ValueError("io_system must define an update function.")

    states = list(state_symbols)
    if len(states) != io_system.nstates:
        raise ValueError("state_symbols length must equal io_system.nstates.")

    params = [] if params is None else list(params)
    params_values = [] if params_values is None else list(params_values)
    if len(params) != len(params_values):
        raise ValueError("params and params_values must have the same length.")

    if io_system.ninputs:
        if input_symbols is None:
            raise ValueError("input_symbols are required for systems with inputs.")
        inputs = list(input_symbols)
        if len(inputs) != io_system.ninputs:
            raise ValueError("input_symbols length must equal io_system.ninputs.")
    else:
        inputs = None

    if input_values is not None and inputs is None:
        raise ValueError("input_values require input_symbols.")
    if input_values is not None and len(input_values) != len(inputs):
        raise ValueError("input_values length must equal input_symbols length.")

    if x_init is not None and len(x_init) != io_system.nstates:
        raise ValueError("x_init length must equal io_system.nstates.")

    if output_matrix is not None and io_system.outfcn is not None:
        raise ValueError(
            "output_matrix cannot be supplied when io_system defines outfcn."
        )

    control_params = {str(param): param for param in params}
    state_vector = np.asarray(states, dtype=object)
    input_vector = np.asarray([] if inputs is None else inputs, dtype=object)
    dynamics = list(io_system.updfcn(time, state_vector, input_vector, control_params))

    output_function = None
    if io_system.outfcn is not None:
        output_function = list(
            io_system.outfcn(time, state_vector, input_vector, control_params)
        )

    return System(
        states,
        dynamics,
        params=params,
        params_values=params_values,
        C=output_matrix,
        g=None,
        h=output_function,
        u=inputs,
        input_values=input_values,
        x_init=x_init,
    )
