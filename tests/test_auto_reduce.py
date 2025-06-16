#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from sympy import Symbol  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
from pathlib import Path

from autoreduce.system import System
from autoreduce.converters import load_sbml


def test_get_reduced_model(reducible_system_1):
    """
    This function creates a reducible System object
    that can be used to create reduced models given
    the x_hat (the list of states in reduced model).
    All other states are collapsed to be at quasi-steady
    state and both the reduced and the collapsed models
    are returned.
    """
    x_hat = []
    assert isinstance(reducible_system_1, System)
    reduced_system, collapsed_system = (
        reducible_system_1.solve_timescale_separation(x_hat)
    )
    if reduced_system is not None:
        assert isinstance(reduced_system, System)
    if collapsed_system is not None:
        assert isinstance(collapsed_system, System)


def test_biocrnplyer_model(system_2):
    """
    This function tests the biocrnpyler model
    """
    assert isinstance(system_2, System)
    assert len(system_2.x) == 3
    assert len(system_2.f) == 3
    assert len(system_2.params) == 1
