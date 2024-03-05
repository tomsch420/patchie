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
        variable = Integer(Column.from_column_element(Color.color), {0, 1}, 0.5, 1)
        self.assertEqual(variable.name, "Color.color")

    def test_symbolic(self):
        variable = Symbolic(Column.from_column_element(Color.color), {0, 1})
        self.assertEqual(variable.name, "Color.color")

    def test_continuous(self):
        variable = Continuous(Column.from_column_element(Point.x), 0.5, 1)
        self.assertEqual(variable.name, "Point.x")

    def test_column_mixin_with_alias(self):
        column = aliased(Color, name="color2").color
        variable = Continuous(Column.from_column_element(column), 0.5, 1)
        self.assertEqual(variable.name, "color2.color")

    def test_serialization(self):
        variable = Integer(Column.from_column_element(Color.color), {0, 1}, 0.5, 1)
        data = variable.to_json()
        deserialized = Integer.from_json(data)
        self.assertEqual(variable, deserialized)
        self.assertEqual(variable.column, deserialized.column)

    def test_relevant_columns_from_table(self):
        columns, foreign_columns, query = relevant_columns_from(ColoredPoint.__table__, 0)
        self.assertEqual([c.name for c in columns + foreign_columns],
                         ["ColoredPoint.frame"])

    def test_relevant_columns_from_table_with_relations(self):
        columns, foreign_columns, query = relevant_columns_from(ColoredPoint.__table__, 1)
        self.assertSetEqual(set([c.name for c in columns + foreign_columns]),
                            {"ColoredPoint.frame", "ColoredPoint.Point.x", "ColoredPoint.Point.y",
                             "ColoredPoint.Color.color"})

    def test_relevant_columns_from_table_with_relations_and_unknown_parent(self):
        columns, foreign_columns, query = relevant_columns_from(Point.__table__, 1)
        self.assertSetEqual(set([c.name for c in columns + foreign_columns]),
                            {"Point.x", "Point.y", "Point.ColoredPoint.frame"})

    def test_sql_column_from_label(self):
        label = Point.x.label("ColoredPoint.Point.x")
        column = Column.from_column_element(label)
        self.assertEqual(column.column_name, "ColoredPoint.Point.x")
        self.assertEqual(column.table_name, "Point")


if __name__ == '__main__':
    unittest.main()
