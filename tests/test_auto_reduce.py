#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

import warnings

import pytest

from autoreduce import explore_all_QSS_models, solve_timescale_separation
from autoreduce.reductions.core import Reduce
from autoreduce.system.system import System


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


def test_direct_solve_timescale_separation(system_1):
    """Solve a QSSA reduction without creating a reducible object first."""
    A, _, C, D = system_1.x
    assert system_1.timepoints_ode is None
    reduced_system, collapsed_system = solve_timescale_separation(
        system_1, [A, C, D], timepoints_ode=[0.0, 1.0]
    )

    assert system_1.timepoints_ode is None
    assert isinstance(reduced_system, System)
    assert isinstance(collapsed_system, System)
    assert reduced_system.x == [A, C, D]


def test_direct_reduction_imports_are_public():
    """The simplified reduction helpers are exported from public packages."""
    from autoreduce import explore_all_QSS_models as package_explore
    from autoreduce.reductions import (
        explore_all_QSS_models as reductions_explore,
    )

    assert package_explore is reductions_explore


def test_direct_explore_all_QSS_models(system_1):
    """Explore candidate reductions without creating a reducible object first."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Solve time-scale separation failed"
        )
        results = explore_all_QSS_models(
            system_1,
            nstates_tol=3,
            nstates_tol_min=2,
            skip_numerical_computations=True,
        )

    assert isinstance(results, dict)
    assert results
    assert all(isinstance(reduced_system, System) for reduced_system in results)
    assert all(result is None for result in results.values())


def test_explore_all_QSS_models_requires_timepoints_for_metrics(system_1):
    """Numerical metrics require ODE and SSM timepoints."""
    with pytest.raises(
        ValueError,
        match="timepoints_ode is used for accuracy",
    ):
        explore_all_QSS_models(system_1)


def test_biocrnplyer_model(system_2):
    """
    This function tests the biocrnpyler model
    """
    assert isinstance(system_2, System)
    assert not isinstance(system_2, Reduce)
    assert len(system_2.x) == 3
    assert len(system_2.f) == 3
    assert len(system_2.params) == 1
