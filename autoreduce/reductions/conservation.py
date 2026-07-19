"""Conservation-law reduction methods."""

from itertools import combinations

import numpy as np
from sympy import Eq, Symbol, simplify, solve

from autoreduce.system.system import System


def _as_reducible(system_obj):
    """Return a `Reduce` object, creating one for plain `System` inputs."""
    from autoreduce.reductions.timescale import Reduce

    if isinstance(system_obj, Reduce):
        return system_obj
    if not isinstance(system_obj, System):
        raise TypeError("system_obj must be an AutoReduce System object.")
    return Reduce(
        system_obj.x,
        system_obj.f,
        params=system_obj.params,
        C=system_obj.C,
        g=system_obj.g,
        h=system_obj.h,
        u=system_obj.u,
        params_values=system_obj.params_values,
        x_init=system_obj.x_init,
        input_values=getattr(system_obj, "input_values", None),
        parameter_dependent_ic=getattr(
            system_obj, "parameter_dependent_ic", False
        ),
        ic_parameters=getattr(system_obj, "ic_parameters", None),
    )


def _coalesce_search_depth(
    conservation_search_depth=None, num_conservation_laws=None
):
    """Resolve the new search-depth name and the legacy alias."""
    if conservation_search_depth is not None:
        return conservation_search_depth
    return num_conservation_laws


def find_conserved_sets(
    system_obj,
    conservation_search_depth=None,
    num_conservation_laws=None,
    **kwargs,
):
    """Find conserved species sets by summing ODE terms.

    Only linear combinations with coefficient 1 are currently supported.
    `conservation_search_depth` is the maximum number of ODE terms to add
    while searching for sums that cancel to zero.

    `num_conservation_laws` is retained as a legacy alias for
    `conservation_search_depth`; it was never strictly the number of laws.
    """
    _ = kwargs
    search_depth = _coalesce_search_depth(
        conservation_search_depth, num_conservation_laws
    )
    if search_depth is None or search_depth <= 0:
        raise ValueError(
            "Pass conservation_search_depth when conserved_sets or "
            "conservation_laws are not provided."
        )

    all_conserved_sets = []
    seen = set()
    ode_terms = [
        (index, ode)
        for index, ode in enumerate(system_obj.f)
        if simplify(ode) != 0
    ]
    for depth in range(1, search_depth + 1):
        for candidate in combinations(ode_terms, depth):
            candidate_indices = [index for index, _ in candidate]
            sum_terms = sum(ode for _, ode in candidate)
            if simplify(sum_terms) != 0:
                continue
            conserved_species = [system_obj.x[index] for index in candidate_indices]
            if len(conserved_species) <= 1:
                continue
            key = tuple(conserved_species)
            if key in seen:
                continue
            seen.add(key)
            all_conserved_sets.append(conserved_species)

    if not all_conserved_sets:
        raise ValueError(
            "No conserved sets found up to "
            f"conservation_search_depth={search_depth}. This value is a "
            "search depth, not the number of conservation laws. Increase it "
            "when a conserved quantity requires summing more ODE terms, or "
            "pass conserved_sets/conservation_laws explicitly."
        )
    return all_conserved_sets


def get_conservation_laws(*args, **kwargs):
    """Backward-compatible alias for `find_conserved_sets`."""
    return find_conserved_sets(*args, **kwargs)


def setup_conservation_laws(system_obj, total_quantities, conserved_sets):
    """Create conservation-law expressions from conserved species sets."""
    if total_quantities is None:
        raise ValueError("total_quantities must be provided.")
    if conserved_sets is None or len(conserved_sets) == 0:
        raise ValueError("conserved_sets must not be empty.")
    if len(total_quantities) != len(conserved_sets):
        raise ValueError(
            "total_quantities must have one entry for each conserved set."
        )

    params = [] if system_obj.params is None else list(system_obj.params)
    params_values = (
        [] if system_obj.params_values is None else list(system_obj.params_values)
    )
    conservation_laws = []
    for conserved_set, total_name in zip(conserved_sets, total_quantities):
        total_symbol = Symbol(total_name)
        if total_symbol not in params:
            params.append(total_symbol)
            params_values.append(total_quantities[total_name])
        law = sum(conserved_set) - total_symbol
        conservation_laws.append(law)

    system_obj.params = params
    system_obj.params_values = params_values
    return conservation_laws


def _unique_laws(conservation_laws, debug=False):
    """Return conservation laws with symbolic duplicates removed."""
    unique = []
    for law in conservation_laws:
        duplicate = any(simplify(law - existing) == 0 for existing in unique)
        if duplicate:
            if debug:
                print(
                    "Found a duplicate conservation law. "
                    "It will be removed."
                )
            continue
        unique.append(law)
    return unique


def _choose_states_to_eliminate(system_obj, conservation_laws):
    """Choose one state from each conservation law, in system order."""
    states_to_eliminate = []
    for law in conservation_laws:
        chosen_var = None
        for state in system_obj.x:
            if state in law.free_symbols:
                chosen_var = state
                break
        if chosen_var is None:
            raise ValueError(
                f"No state variable in conservation law {law} can be eliminated."
            )
        states_to_eliminate.append(chosen_var)
    return states_to_eliminate


def _delete_values(values, indices):
    """Delete state-indexed values while tolerating absent values."""
    if values is None:
        return values
    if len(values) == 0:
        return []
    return np.delete(np.asarray(values, dtype=object), indices).tolist()


def _substitute_collection(values, substitutions):
    """Substitute eliminated states in a scalar or collection of expressions."""
    def substitute_expr(expr):
        result = expr
        for old, new in substitutions:
            if hasattr(result, "subs"):
                result = result.subs(old, new)
        return result

    if values is None:
        return None
    if isinstance(values, np.ndarray):
        result = values.copy()
        return np.vectorize(substitute_expr)(result)
    if isinstance(values, (list, tuple)):
        return [substitute_expr(expr) for expr in values]
    return substitute_expr(values)


def apply_conservation_laws(
    system_obj, conservation_laws, states_to_eliminate
):
    """Apply conservation laws to a system in place."""
    if conservation_laws is None or len(conservation_laws) == 0:
        raise ValueError("conservation_laws must not be empty.")
    if states_to_eliminate is None or len(states_to_eliminate) == 0:
        raise ValueError("states_to_eliminate must not be empty.")
    if len(conservation_laws) != len(states_to_eliminate):
        raise ValueError(
            "conservation_laws and states_to_eliminate must have the same length."
        )

    state_indices = [system_obj.x.index(state) for state in states_to_eliminate]
    substitutions = []
    for law, state in zip(conservation_laws, states_to_eliminate):
        state_solutions = solve(Eq(law, 0), state)
        if not state_solutions:
            raise ValueError(
                f"Could not solve conservation law {law} for state {state}."
            )
        substitutions.append((state, state_solutions[0]))

    for old_state, replacement in substitutions:
        system_obj.f = [ode.subs(old_state, replacement) for ode in system_obj.f]
        system_obj.h = _substitute_collection(
            system_obj.h, [(old_state, replacement)]
        )

    system_obj.x = _delete_values(system_obj.x, state_indices)
    system_obj.f = _delete_values(system_obj.f, state_indices)
    system_obj.x_init = _delete_values(system_obj.x_init, state_indices)
    if getattr(system_obj, "parameter_dependent_ic", False):
        system_obj.ic_parameters = _delete_values(
            system_obj.ic_parameters, state_indices
        )
    if system_obj.C is not None:
        c_array = np.asarray(system_obj.C)
        if c_array.ndim == 1:
            system_obj.C = np.delete(c_array, state_indices).tolist()
        else:
            system_obj.C = np.delete(c_array, state_indices, axis=1)
    system_obj.n = len(system_obj.x)
    return system_obj.f


def solve_conservation_laws(
    system_obj,
    conservation_laws=None,
    total_quantities=None,
    conserved_sets=None,
    states_to_eliminate=None,
    conservation_search_depth=None,
    num_conservation_laws=None,
    **kwargs,
):
    """Find and apply conservation laws to a `System`.

    Plain `System` inputs are converted to a `Reduce` working object
    internally, so callers do not need to create `get_reducible(system)` first.
    `num_conservation_laws` remains available as a compatibility alias for
    `conservation_search_depth`.
    """
    debug = kwargs.get("debug", False)
    reducible_system = _as_reducible(system_obj)
    search_depth = _coalesce_search_depth(
        conservation_search_depth, num_conservation_laws
    )
    reducible_system.conservation_search_depth = search_depth
    reducible_system.num_conservation_laws = search_depth

    if (
        search_depth is None
        and conserved_sets is None
        and conservation_laws is None
    ):
        raise ValueError(
            "Pass conservation_laws, conserved_sets, or "
            "conservation_search_depth."
        )

    if conservation_laws is None:
        if conserved_sets is None:
            conserved_sets = find_conserved_sets(
                reducible_system,
                conservation_search_depth=search_depth,
                debug=debug,
            )
        elif not conserved_sets:
            raise ValueError("conserved_sets must not be empty.")
        reducible_system.conserved_sets = conserved_sets

        if total_quantities is None:
            total_quantities = {
                str(conserved_set[0]) + "_total": 0
                for conserved_set in conserved_sets
            }
        reducible_system.total_quantities = total_quantities
        conservation_laws = setup_conservation_laws(
            reducible_system, total_quantities, conserved_sets
        )
        if debug:
            print("Found conservation laws:", conservation_laws)

    reducible_system.conservation_laws = _unique_laws(
        conservation_laws, debug=debug
    )
    if states_to_eliminate is None:
        states_to_eliminate = _choose_states_to_eliminate(
            reducible_system, reducible_system.conservation_laws
        )
        if debug:
            print("Choosing states to eliminate:", states_to_eliminate)
    reducible_system.states_to_eliminate = states_to_eliminate
    reducible_system.f = apply_conservation_laws(
        reducible_system,
        conservation_laws=reducible_system.conservation_laws,
        states_to_eliminate=reducible_system.states_to_eliminate,
    )
    return reducible_system
