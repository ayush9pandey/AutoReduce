
#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from test_auto_reduce import TestAutoReduce

import libsbml
import warnings


class TestSystem(TestCase):
    def setUp(self):
        self.system1 = None
        self.system2 = None

    def test_system_equality(self, system1 = None, system2 = None):
        """
        Test all properties of two systems for equality
        """
        self.system1 = system1
        self.system2 = system2

        if system1 is None and system2 is None:
            return 
        elif system1 is None or system2 is None:
            warnings.warn('One of the System objects is None.')
        else:
            self.test_states()
            self.test_f()
            self.test_g()
            self.test_h()
            self.test_params()
            self.test_initial_conditions()
            self.test_params_values()
            self.test_C()

    def test_states(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.x, self.system2.x)

    def test_f(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.f, self.system2.f)
    
    def test_g(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.g, self.system2.g)

    def test_h(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.h, self.system2.h)

    def test_params(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.params, self.system2.params)
    
    def test_params_values(self):
        if self.system1 is not None and self.system2 is not None:
            self.assertEqual(self.system1.params_values, 
                            self.system2.params_values)

    def test_initial_conditions(self):
        if self.system1 is not None and self.system2 is not None:
            if self.system1.x_init is not None and self.system2.x_init is not None:
                self.assertTrue((self.system1.x_init == self.system2.x_init).all())

    def test_C(self):
        if self.system1 is not None and self.system2 is not None:
            if self.system1.C is not None and self.system2.C is not None:
                self.assertTrue((self.system1.C == self.system2.C).all())
    
