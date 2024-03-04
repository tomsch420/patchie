import os
import unittest

from probabilistic_model.bayesian_network.distributions import ConditionalProbabilisticCircuit
from sqlalchemy import select, func, join

from orm import *
from patchie.model_loader import FolderModelLoader
from patchie.patchie import Patchie
from patchie.variables import (variables_and_dataframe_from_objects, Symbolic,
                               variables_and_dataframe_from_columns_and_query, relevant_columns_from, Column)
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.probabilistic_circuit.distributions import SymbolicDistribution
import networkx as nx
import matplotlib.pyplot as plt
import portion
import tempfile


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
    color = Symbolic(Column.from_column_element(Color.color), {"red", "blue", "green"})

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


class FittingTestCase(ORMMixin, unittest.TestCase):

    model: Patchie
    folder: str

    def setUp(self):
        super().setUp()
        self.model = Patchie()
        self.folder = tempfile.mkdtemp()
        model_loader = FolderModelLoader(self.folder)
        self.model.model_loader = model_loader

    def test_fitting(self):
        self.model.fit_to_tables(self.session, [Point, ColoredPoint, Color])
        self.assertEqual(set(os.listdir(self.folder)), {"Color.json", "ColoredPoint.json", "Point.json"})

    def test_fitting_to_table(self):
        model = self.model.fit_to_table(self.session, ColoredPoint)
        self.assertEqual(len(model.variables), 1)
        self.assertEqual(model.variables[0].name, "ColoredPoint.frame")

    def test_fitting_to_table_with_child_join_table(self):
        model = self.model.fit_to_table(self.session, Point)
        self.assertEqual(len(model.variables), 2)
        self.assertEqual(model.variables[0].name, "Point.x")
        self.assertEqual(model.variables[1].name, "Point.y")

    def test_fitting_of_interaction_model(self):
        self.model.fit_to_tables(self.session, [Point, ColoredPoint, Color])
        interaction_model = self.model.fit_interaction_model(self.session, ColoredPoint, Point)
        self.assertEqual([v.name for v in interaction_model.variables], ["ColoredPoint.latent", "Point.latent"])

    def test_variables_and_dataframe_from_query_result_and_columns(self):
        columns, foreign_columns, query = relevant_columns_from(Point.__table__, 1)
        variables, foreign_variables, dataframe = variables_and_dataframe_from_columns_and_query(
            columns, foreign_columns, query, self.session)
        self.assertEqual(len(variables), 2)


class QueryFittingTestCase(ORMMixin, unittest.TestCase):

    model: Patchie
    folder: str

    def setUp(self):
        super().setUp()
        self.model = Patchie()
        self.folder = tempfile.mkdtemp()
        model_loader = FolderModelLoader(self.folder)
        self.model.model_loader = model_loader
        self.model.fit_to_tables(self.session, [Point, ColoredPoint, Color])
        interaction_model = self.model.fit_interaction_model(self.session, ColoredPoint, Point)
        self.model.model_loader.save_interaction_model(interaction_model, [ColoredPoint, Point])
        interaction_model = self.model.fit_interaction_model(self.session, ColoredPoint, Color)
        self.model.model_loader.save_interaction_model(interaction_model, [ColoredPoint, Color])

    def test_query_construction(self):
        query = join(Point, ColoredPoint).join(Color)
        model = self.model.load_models_from_join(query)
        self.assertEqual(len(model.variables), 7)

        for node, out_degree in model.out_degree:
            if out_degree == 0:
                self.assertIsInstance(node, ConditionalProbabilisticCircuit)
            else:
                self.assertEqual(len(node.variables), 1)
                variable = node.variable
                self.assertTrue(variable.name.endswith("latent"))

        model = model.as_probabilistic_circuit()
        print(model)

if __name__ == '__main__':
    unittest.main()
