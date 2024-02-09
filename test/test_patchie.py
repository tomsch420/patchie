import unittest

from sqlalchemy import select, func, join

from orm import *
from patchie.patchie import Patchie
from patchie.variables import variables_and_dataframe_from_objects, Symbolic, SQLColumn
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.probabilistic_circuit.distributions import SymbolicDistribution
import networkx as nx
import matplotlib.pyplot as plt
import portion


class ContinuousPatchieTestCase(ORMMixin, unittest.TestCase):

    model: Patchie

    def setUp(self):
        super().setUp()
        self.model = Patchie()
        points = self.session.query(Point).limit(500).all()
        variables, dataframe = variables_and_dataframe_from_objects(points)
        model = JPT(variables, min_samples_leaf=0.2)
        model.fit(dataframe)
        self.model.add_edges_from(model.probabilistic_circuit.unweighted_edges)
        self.model.add_weighted_edges_from(model.probabilistic_circuit.weighted_edges)
        self.model.add_nodes_from(model.probabilistic_circuit.nodes)

    def test_setup(self):
        self.assertIsNotNone(self.session.connection())
        query = select(func.count(Point.id))
        self.assertEqual(self.session.scalar(query), 2000)

    def test_orm(self):
        query = select(ColoredPoint)
        result = self.session.scalar(query)
        self.assertIsNotNone(result.point)
        self.assertIsNotNone(result.id)
        self.assertIsNotNone(result.color)

    def show(self):
        nx.draw(self.model, with_labels=True)
        plt.show()

    def test_query_creation_boolean_clause(self):
        query = (Point.x > 0)
        event = self.model.event_from_query(query)
        self.assertEqual(event["Point.x"], portion.open(0, float("inf")))

    def test_query_creation_boolean_clause_list_and(self):
        query = (Point.x > 0) & (Point.y > 1)
        event = self.model.event_from_query(query)
        self.assertEqual(event["Point.x"], portion.open(0, float("inf")))
        self.assertEqual(event["Point.y"], portion.open(1, float("inf")))

    def test_query_creation_boolean_clause_list_or(self):
        query = (Point.x > 0) | (Point.x < -1)
        event = self.model.event_from_query(query)
        self.assertEqual(event["Point.x"],
                         portion.open(0, float("inf")) | portion.open(float("-inf"), -1))

    def test_query_creation_complex(self):
        query = (Point.x > 0) | (Point.x < -1) & (Point.y > 1) & (Point.y < 2)
        event = self.model.event_from_query(query)
        self.assertEqual(event["Point.x"],
                         portion.open(0, float("inf")) | portion.open(float("-inf"), -1))
        self.assertEqual(event["Point.y"], portion.open(1, 2))

    def test_model_creation_from_join(self):
        query = join(Point, ColoredPoint).join(Color)
        self.model.load_models_from_join(query)


class DiscretePatchieTestCase(ORMMixin, unittest.TestCase):

    model: Patchie
    color = Symbolic(SQLColumn(Color.color), {"red", "blue", "green"})

    def setUp(self):
        super().setUp()
        distribution = SymbolicDistribution(self.color, [0.5, 0.3, 0.2])
        self.model = Patchie()
        self.model.add_node(distribution)

    def test_equality_query(self):
        query = (Color.color == "red")
        event = self.model.event_from_query(query)
        self.assertEqual(event[self.color], ("red", ))

    def test_inequality_query(self):
        query = (Color.color != "red")
        event = self.model.event_from_query(query)
        self.assertEqual(event[self.color], ("blue", "green"))

    def test_containment(self):
        query = (Color.color.in_(["red", "blue"]))
        event = self.model.event_from_query(query)
        self.assertEqual(event[self.color], ("blue", "red"))

    def test_not_containment(self):
        query = (Color.color.notin_(["red", "blue"]))
        event = self.model.event_from_query(query)
        self.assertEqual(event[self.color], ("green", ))


if __name__ == '__main__':
    unittest.main()
