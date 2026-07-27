import numpy as np
import pytest
from sympy import Symbol

from autoreduce import System, find_conserved_sets, solve_conservation_laws
from autoreduce.reductions.core import Reduce


def test_search_depth_and_direct_solver():
    """Find and apply enzyme conservation from a plain System object."""
    S = Symbol("S")
    E = Symbol("E")
    C = Symbol("C")
    P = Symbol("P")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    k3 = Symbol("k3")

    system = System(
        [S, E, C, P],
        [
            -k1 * S * E + k2 * C,
            -k1 * S * E + (k2 + k3) * C,
            k1 * S * E - (k2 + k3) * C,
            k3 * C,
        ],
        params=[k1, k2, k3],
        params_values=[1.0, 0.5, 0.25],
        x_init=[10.0, 1.0, 0.0, 0.0],
        C=np.array([[0.0, 0.0, 0.0, 1.0]]),
    )

    with pytest.raises(ValueError, match="search_depth=1"):
        find_conserved_sets(system, search_depth=1)

    assert find_conserved_sets(system, search_depth=2) == [[E, C]]

    conserved_system = solve_conservation_laws(
        system,
        total_quantities={"E_total": 1.0},
        conserved_sets=[[E, C]],
        states_to_eliminate=[E],
    )
    assert isinstance(conserved_system, System)
    assert not isinstance(conserved_system, Reduce)
    assert conserved_system.x == [S, C, P]
    assert len(conserved_system.x_init) == len(conserved_system.x)
    assert system.x == [S, E, C, P]
    assert len(system.x_init) == len(system.x)
    assert Symbol("E_total") in system.params
    assert Symbol("E_total") in conserved_system.params


def test_conservation_rejects_eliminated_output_state():
    """A state used directly in y=Cx cannot be eliminated algebraically."""
    S = Symbol("S")
    E = Symbol("E")
    C = Symbol("C")
    k = Symbol("k")

    system = System(
        [S, E, C],
        [-k * S * E, -k * S * E, k * S * E],
        params=[k],
        params_values=[1.0],
        x_init=[10.0, 1.0, 0.0],
        C=np.array([[0.0, 1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="linear output C@x"):
        solve_conservation_laws(
            system,
            total_quantities={"E_total": 1.0},
            conserved_sets=[[E, C]],
            states_to_eliminate=[E],
        )
