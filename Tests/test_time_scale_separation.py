
#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from test_auto_reduce import TestAutoReduce
import warnings


class TestTimeScaleSeparation(TestAutoReduce):
    def test_reduced_models(self):
        A, B, C, D = self.x
        k1, k2, k3 = self.params
        possible_reductions = self.reducible_system.get_all_combinations()
        for attempt in possible_reductions:
            attempt_states = [self.x[i] for i in attempt]
            answer_AB = [A**2*B*k1*k2/(k2+k3) - A**2*B*k1,
                         A**2*B*k1*k2/(k2+k3) - A**2*B*k1]
            self.test_solve_timescale_separation(attempt_states = [A, B], 
                                                mode = 'fail', answer = answer_AB)
            self.test_solve_timescale_separation(attempt_states = [A, D], 
                                                mode = 'fail', answer = [0, 0])
            self.test_solve_timescale_separation(attempt_states = [A, C], 
                                                mode = 'fail', answer = [0, -C*k3])
            self.test_solve_timescale_separation(attempt_states = [B, C, D], 
                                                mode = 'success', 
                                                answer = [0, -k3 * C, k3 * C])
            self.test_solve_timescale_separation(attempt_states = [A, C, D], 
                                                mode = 'success', 
                                                answer = [0, -k3 * C, k3 * C])
            # self.test_solve_timescale_separation(attempt_states = [A, B, C], 
        #                                       mode = 'success', 
        #                                       answer = [-k1 * A**2 * B, 
    #                                                   -k1 * A**2 * B, 
    #                                                   k1 * A**2 * B]) 
            self.test_solve_timescale_separation(attempt_states = [C, D], 
                                                mode = 'success', 
                                                answer = [-k3 * C, k3 * C])
            answer_ABD = [k1 * k2 * A**2 * B / (k2 + k3) - k1 * A**2 * B, 
                        k1 * k2 * A**2 * B / (k2 + k3) - k1 * A**2 * B, 
                        k1 * k3 * A**2 * B / (k2 + k3)]
            self.test_solve_timescale_separation(attempt_states = [A, B, D], 
                                                mode = 'success',
                                                answer = answer_ABD)

    def test_solve_timescale_separation(self, attempt_states = None, mode = None, answer = None):
        if mode == 'fail':
            with self.assertWarnsRegex(Warning, 'Solve time-scale separation failed. Check model consistency.'):
                reduced_system, collapsed_system = self.test_get_reduced_model(x_hat=attempt_states)
                print(reduced_system.f)
                self.assertEqual(reduced_system.f, answer)
                self.assertEqual(collapsed_system.x, [x for x in self.x if x not in attempt_states])
        elif mode == 'success':
            reduced_system, collapsed_system = self.test_get_reduced_model(x_hat = attempt_states)
            assert answer is not None
            self.assertEqual(reduced_system.f, answer)
            # Test slow (retained) states
            self.assertEqual(reduced_system.x, attempt_states)
            # Test fast (collapsed) states
            self.assertEqual(collapsed_system.x, [x for x in self.x if x not in attempt_states])