#  Copyright (c) 2025, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

import pytest  # type: ignore
from autoreduce import System


def test_reduced_models(system, reducible_system):
    A, B, C, D = system.x
    k1, k2, k3 = system.params
    possible_reductions = reducible_system.get_all_combinations()
    for attempt in possible_reductions:
        answer_AB = [
            A**2 * B * k1 * k2 / (k2 + k3) - A**2 * B * k1,
            A**2 * B * k1 * k2 / (k2 + k3) - A**2 * B * k1,
        ]
        # On simplification, the answer is:
        answer_AB_eq = [
            -(A**2) * B * k1 * k3 / (k2 + k3),
            -(A**2) * B * k1 * k3 / (k2 + k3),
        ]
        try:
            solve_timescale_test(
                system,
                reducible_system,
                attempt_states=[A, B],
                mode="fail",
                answer=answer_AB,
            )
        except Exception:
            solve_timescale_test(
                system,
                reducible_system,
                attempt_states=[A, B],
                mode="fail",
                answer=answer_AB_eq,
            )

        solve_timescale_test(
            system,
            reducible_system,
            attempt_states=[A, D],
            mode="fail",
            answer=[0, 0],
        )
        solve_timescale_test(
            system,
            reducible_system,
            attempt_states=[A, C],
            mode="fail",
            answer=[0, -C * k3],
        )
        solve_timescale_test(
            system,
            reducible_system,
            attempt_states=[B, C, D],
            mode="success",
            answer=[0, -k3 * C, k3 * C],
        )
        solve_timescale_test(
            system,
            reducible_system,
            attempt_states=[A, C, D],
            mode="success",
            answer=[0, -k3 * C, k3 * C],
        )
        # solve_timescale_test(attempt_states = [A, B, C],
        #                          mode = 'success',
        #                          answer = [-k1 * A**2 * B,
        #                                        -k1 * A**2 * B,
        #                                        k1 * A**2 * B])
        solve_timescale_test(
            system,
            reducible_system,
            attempt_states=[C, D],
            mode="success",
            answer=[-k3 * C, k3 * C],
        )
        answer_ABD = [
            k1 * k2 * A**2 * B / (k2 + k3) - k1 * A**2 * B,
            k1 * k2 * A**2 * B / (k2 + k3) - k1 * A**2 * B,
            k1 * k3 * A**2 * B / (k2 + k3),
        ]
        # On simplification, the answer is:
        answer_ABD_eq = [
            -(A**2) * B * k1 * k3 / (k2 + k3),
            -(A**2) * B * k1 * k3 / (k2 + k3),
            A**2 * B * k1 * k3 / (k2 + k3),
        ]
        try:
            solve_timescale_test(
                system,
                reducible_system,
                attempt_states=[A, B, D],
                mode="success",
                answer=answer_ABD,
            )
        except Exception:
            solve_timescale_test(
                system,
                reducible_system,
                attempt_states=[A, B, D],
                mode="success",
                answer=answer_ABD_eq,
            )


def solve_timescale_test(
    system, reducible_system, attempt_states=None, mode=None, answer=None
):
    if mode == "fail":
        with pytest.warns(Warning, match="Solve time-scale separation failed"):
            reduced_system, collapsed_system = get_reduced_model_helper(
                system, reducible_system, x_hat=attempt_states
            )
            if reduced_system is not None:
                print(reduced_system.f)
                assert reduced_system.f == answer
                assert collapsed_system.x == [
                    x for x in system.x if x not in attempt_states
                ]
    elif mode == "success":
        reduced_system, collapsed_system = get_reduced_model_helper(
            system, reducible_system, x_hat=attempt_states
        )
        assert answer is not None
        assert reduced_system is not None
        assert reduced_system.f == answer
        # Test slow (retained) states
        assert reduced_system.x == attempt_states
        # Test fast (collapsed) states
        assert collapsed_system.x == [
            x for x in system.x if x not in attempt_states
        ]


def get_reduced_model_helper(reducible_system, x_hat=None):
    if x_hat is None:
        x_hat = []
    assert isinstance(reducible_system, System)
    reduced_system, collapsed_system = (
        reducible_system.solve_timescale_separation(x_hat)
    )
    if reduced_system is not None:
        assert isinstance(reduced_system, System)
    if collapsed_system is not None:
        assert isinstance(collapsed_system, System)
    return reduced_system, collapsed_system
