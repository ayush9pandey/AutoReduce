import numpy as np
import pytest
from sympy import Symbol

from autoreduce import System, find_conserved_sets, solve_conservation_laws


def test_conservation_search_depth_and_direct_solver():
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
        C=np.eye(4),
    )

    with pytest.raises(ValueError, match="conservation_search_depth=1"):
        find_conserved_sets(system, conservation_search_depth=1)

    assert find_conserved_sets(system, conservation_search_depth=2) == [[E, C]]

    conserved_system = solve_conservation_laws(
        system,
        total_quantities={"E_total": 1.0},
        conserved_sets=[[E, C]],
        states_to_eliminate=[E],
    )
    assert conserved_system.x == [S, C, P]
    assert Symbol("E_total") in conserved_system.params
