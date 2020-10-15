#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from sympy import Symbol
import numpy as np
from autoreduce.system import System
from autoreduce.utils import get_reducible
from autoreduce.model_reduction import Reduce
import libsbml
import warnings


class TestAutoReduce(TestCase):
    """
    Super class of all testing in AutoReduce as it sets up a System with 
    a simple test CRN
    """
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
        init_cond = np.ones(len(self.x))
        self.C = None
        self.g = None
        self.h = None
        self.u = None
        self.input_values = None
        self.params_values = [2, 4, 6]
        self.system = System(self.x, self.f, params = self.params, x_init = init_cond, params_values = self.params_values,
                            C = self.C, g = self.g, h = self.h, u = self.u, input_values = self.input_values)   
        self.reducible_system = get_reducible(self.system)
    
    def test_get_reduced_model(self, x_hat = None):
        """
        This function creates a reducible System object that can be used 
        to create reduced models given the x_hat (the list of states in reduced model). 
        All other states are collapsed to be at quasi-steady state and both the reduced and 
        the collapsed models are returned.
        """
        if x_hat is None:
            x_hat = []
        self.assertIsInstance(self.reducible_system, System)
        reduced_system, collapsed_system = self.reducible_system.solve_timescale_separation(x_hat)
        if reduced_system is not None:
            self.assertIsInstance(reduced_system, System)
        if collapsed_system is not None:
            self.assertIsInstance(collapsed_system, System)
        return reduced_system, collapsed_system
        

