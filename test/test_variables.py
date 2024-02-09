import unittest

from sqlalchemy.orm import aliased, Session
from sqlalchemy import create_engine
from sqlalchemy import select, func

from patchie.variables import *
from orm import Base, Point, ColoredPoint, ORMMixin, Color

import numpy as np
import random


class VariableTestCase(unittest.TestCase):

    def test_integer(self):
        variable = Integer(SQLColumn(Color.color), {0, 1}, 0.5, 1)
        self.assertEqual(variable.name, "Color.color")

    def test_symbolic(self):
        variable = Symbolic(SQLColumn(Color.color), {0, 1})
        self.assertEqual(variable.name, "Color.color")

    def test_continuous(self):
        variable = Continuous(SQLColumn(Point.x), 0.5, 1)
        self.assertEqual(variable.name, "Point.x")

    def test_column_mixin_with_alias(self):
        column = aliased(Color, name="color2").color
        variable = Continuous(SQLColumn(column), 0.5, 1)
        self.assertEqual(variable.name, "color2.color")

    def test_serialization(self):
        variable = Integer(SQLColumn(Color.color), {0, 1}, 0.5, 1)
        data = variable.to_json()
        deserialized = Integer.from_json(data)
        self.assertEqual(variable, deserialized)
        self.assertEqual(variable.column, deserialized.column)


@unittest.skip("Unsure how this should look like from a use case.")
class VariableInferenceTestCase(ORMMixin, unittest.TestCase):

    @property
    def points(self):
        return self.session.execute(select(Point)).scalars().all()

    def test_relevant_columns_from(self):
        columns = relevant_columns_from(Point.__table__)
        self.assertEqual(columns, [Point.x, Point.y, Color.color])

    def test_dataframe_from_objects(self):
        dataframe = dataframe_from_objects(self.points)
        self.assertEqual(dataframe.columns.tolist(), ["x", "y"])

    def test_variables_and_dataframe_from_objects(self):
        variables, dataframe = variables_and_dataframe_from_objects(self.points)
        self.assertEqual(dataframe.columns.tolist(), ["Point.x", "Point.y", "Color.color"])

    @unittest.skip
    def test_variables_and_dataframe_from_objects_with_alias(self):
        points = self.orm_session.execute(select(aliased(Point, name="Point2"))).scalars().all()
        variables, dataframe = variables_and_dataframe_from_objects(points)
        self.assertEqual(dataframe.columns.tolist(), ["Point2.x", "Point2.y", "Point2.color"])


if __name__ == '__main__':
    unittest.main()
