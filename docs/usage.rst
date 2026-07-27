Usage
=====

QSSA Example
------------

Quasi-steady-state approximation (QSSA) assumes that selected fast states
relax quickly compared with the states retained in the reduced model. For a
Michaelis-Menten-style mechanism with enzyme conservation,

.. math::

   E = E_T - C,

the full symbolic system can be written as

.. math::

   \begin{aligned}
   \dot{S} &= -k_1(E_T - C)S + k_2C, \\
   \dot{C} &= k_1(E_T - C)S - (k_2 + k_3)C, \\
   \dot{P} &= k_3C.
   \end{aligned}

If ``C`` is treated as the fast state, AutoReduce solves
``dot(C) = 0`` and substitutes the resulting algebraic expression into the
slow dynamics for ``S`` and ``P``.

.. code-block:: python

    import numpy as np
    from sympy import Symbol, simplify

    from autoreduce import System, solve_timescale_separation

    S = Symbol("S")
    C = Symbol("C")
    P = Symbol("P")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    k3 = Symbol("k3")
    E_total = Symbol("E_total")

    E = E_total - C
    x = [S, C, P]
    f = [
        -k1 * E * S + k2 * C,
        k1 * E * S - (k2 + k3) * C,
        k3 * C,
    ]

    system = System(
        x,
        f,
        params=[k1, k2, k3, E_total],
        params_values=[1.0, 0.5, 0.25, 1.0],
        x_init=[10.0, 0.0, 0.0],
        C=np.array([[1, 0, 0], [0, 0, 1]]),
    )

    reduced_system, collapsed_system = (
        solve_timescale_separation(
            system,
            [S, P],
            fast_states=[C],
        )
    )

    reduced_dynamics = [simplify(expr) for expr in reduced_system.f]

The reduced dynamics are

.. math::

   \begin{aligned}
   \dot{S} &= -\frac{E_T S k_1 k_3}{S k_1 + k_2 + k_3}, \\
   \dot{P} &= \frac{E_T S k_1 k_3}{S k_1 + k_2 + k_3}.
   \end{aligned}

Conservation-Law Example
------------------------

Conservation laws remove states whose values are determined by invariant
total quantities. For the enzyme-substrate mechanism,

.. math::

   E + C = E_T,

AutoReduce can eliminate ``E`` before applying other reductions.

.. code-block:: python

    import numpy as np
    from sympy import Symbol, simplify

    from autoreduce import System, solve_conservation_laws

    S = Symbol("S")
    E = Symbol("E")
    C = Symbol("C")
    P = Symbol("P")
    k1 = Symbol("k1")
    k2 = Symbol("k2")
    k3 = Symbol("k3")

    x = [S, E, C, P]
    f = [
        -k1 * S * E + k2 * C,
        -k1 * S * E + (k2 + k3) * C,
        k1 * S * E - (k2 + k3) * C,
        k3 * C,
    ]

    system = System(
        x,
        f,
        params=[k1, k2, k3],
        params_values=[1.0, 0.5, 0.25],
        x_init=[10.0, 1.0, 0.0, 0.0],
        C=np.eye(4),
    )

    conserved_system = solve_conservation_laws(
        system,
        total_quantities={"E_total": 1.0},
        conserved_sets=[[E, C]],
        states_to_eliminate=[E],
    )

    conserved_dynamics = [simplify(expr) for expr in conserved_system.f]

For more worked examples, see the :doc:`examples` section.
