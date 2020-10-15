
#  Copyright (c) 2020, Build-A-Cell. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
import libsbml
import warnings
import numpy as np
from scipy.integrate import odeint

from test_auto_reduce import TestAutoReduce
from test_system import TestSystem

from autoreduce.ode import ODE
from autoreduce.system import System 
from autoreduce.utils import get_ODE


class TestODE(TestAutoReduce):
    def test_ode_objects(self):
        import numpy as np
        timepoints = np.linspace(0,10,100)
        ode_object = ODE(self.x, self.f, self.params, self.C, self.g, self.h, self.u,
                        params_values = [2, 4, 6], x_init = np.ones(4), input_values = self.input_values, timepoints = timepoints)
        ode_object_same = get_ODE(self.system, timepoints = timepoints)
        self.assertIsInstance(ode_object, ODE)
        self.assertIsInstance(ode_object, System)
        self.assertEqual(ode_object.f, ode_object_same.f)
        self.assertIsInstance(ode_object.get_system(), System)
        test_sys = TestSystem()
        test_sys.test_system_equality(ode_object.get_system(), self.system)
        test_sys.test_system_equality(ode_object.get_system(), self.system)
        
    def test_ode_solutions(self):
        """
        Solve the ODE manually and solve using ODE class to compare the two solutions
        """
        timepoints = np.linspace(0,10,100)
        params_values = [2, 4, 6]
        ode_object = ODE(self.x, self.f, self.params, self.C, self.g, self.h, self.u,
                        params_values = params_values , x_init = np.ones(4), input_values = self.input_values, timepoints = timepoints)
        solutions = ode_object.solve_system()

        self.assertIsInstance(solutions, np.ndarray)

        # Solve manually:
        def scipy_odeint_func(x, t):            
            k1, k2, k3 = params_values
            A, B, C, D = x
            return np.array([-k1 * A**2 * B + k2 * C,
                    -k1 * A**2 * B + k2 * C,
                    k1 * A**2 * B - k2 * C - k3 * C,
                    k3 * C])

        solutions_manual = odeint(scipy_odeint_func, y0 = np.ones(4), t = timepoints)
        self.assertTrue((solutions == solutions_manual).all())
