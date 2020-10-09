#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from sympy import Symbol
from autoreduce.system import System
import libsbml
import warnings


class TestAutoReduce(TestCase):

    def setUp(self) -> None:
        """
        This method gets executed before every test. It sets up a test CRN:
        2A + B (k1)<->(k2) C, C (k3)--> D
        """
        A = Symbol("A")
        B = Symbol("B")
        C = Symbol("C")
        D = Symbol("D")
        k1 = Symbol("k1")
        k2 = Symbol("k2")
        k3 = Symbol("k3")
        
        self.params = [k1, k2, k3]
        # States:
        self.x = [A, B, C, D]

        # ODE in Sympy for the given test CRN with mass-action kinetics
        self.f = [-k1 * A**2 * B + k2 * C,
            -k1 * A**2 * B + k2 * C,
            k1 * A**2 * B - k2 * C - k3 * C,
            k3 * C]
        
        self.system = System(self.x, self.f)   
        
class TestTimeScaleSeparation(TestAutoReduce):

    def test_states(self):
        assert self.system.x
        assert len(self.system.x) == 4

    def test_fast_states(self):
        pass

    def test_slow_states(self):
        pass