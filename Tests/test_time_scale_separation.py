#  Copyright (c) 2020, Ayush Pandey. All rights reserved.
#  See LICENSE file in the project root directory for details.

from unittest import TestCase
from unittest.mock import mock_open, patch
from sympy import Symbol
import libsbml
import warnings


class TestTimeScaleSeparation(TestCase):

    def setUp(self) -> None:
        """
        This method gets executed before every test
        """
        A = Symbol("A")
        A = Symbol("A")
        A = Symbol("A")
        A = Symbol("A")
        pass

    def test_check_crn_validity(self):
        pass
