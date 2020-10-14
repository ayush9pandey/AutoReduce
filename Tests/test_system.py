
#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from test_auto_reduce import TestAutoReduce

import libsbml
import warnings


class TestSystem(TestAutoReduce):
    def test_states(self):
        self.assertEqual(self.system.x, self.x)
        self.assertEqual(len(self.system.x), len(self.x))

    def test_f(self):
        self.assertEqual(self.system.f, self.f)
        self.assertEqual(len(self.system.f), len(self.f))
    
    def test_params(self):
        self.assertEqual(self.system.params, self.params)
        self.assertEqual(len(self.system.params), len(self.params))
