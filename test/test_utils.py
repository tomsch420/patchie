import unittest

import probabilistic_model.learning.jpt.variables
import random_events.variables

import patchie.utils
from orm import ORMMixin, Point, Color
from patchie.utils import variable_type_of_column, binary_expression_to_continuous_constraint
import operator
import portion


class UtilsTestCase(ORMMixin, unittest.TestCase):

    def test_jpt_variable_type_of_variable_type(self):
        self.assertTrue(patchie.utils.jpt_variable_type_of_variable_type(
            random_events.variables.Continuous) is probabilistic_model.learning.jpt.variables.Continuous)
        self.assertTrue(patchie.utils.jpt_variable_type_of_variable_type(
            random_events.variables.Symbolic) is random_events.variables.Symbolic)
        self.assertTrue(patchie.utils.jpt_variable_type_of_variable_type(
            random_events.variables.Integer) is probabilistic_model.learning.jpt.variables.Integer)

    def test_variable_type_of_column(self):
        self.assertTrue(issubclass(variable_type_of_column(Point.x), random_events.variables.Continuous))
        self.assertTrue(issubclass(variable_type_of_column(Color.color), random_events.variables.Symbolic))

    def test_binary_expression_to_constraint(self):
        interval = portion.closed(0, float("inf"))
        self.assertEqual(binary_expression_to_continuous_constraint(operator.ge, 0), interval)

    def test_binary_expression_to_discrete_constraint(self):
        self.assertEqual(patchie.utils.binary_expression_to_discrete_constraint(operator.eq, 0), {0})


if __name__ == '__main__':
    unittest.main()
