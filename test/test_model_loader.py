import json
import os
import unittest

from sqlalchemy import select, func

from orm import *
from patchie.model_loader import FolderModelLoader
from patchie.variables import variables_and_dataframe_from_objects
from probabilistic_model.learning.jpt.jpt import JPT
import tempfile


class ModelLoaderTestCase(ORMMixin, unittest.TestCase):

    model_loader: FolderModelLoader
    temporary_directory = tempfile.TemporaryDirectory()

    def setUp(self):
        super().setUp()

        self.model_loader = FolderModelLoader(self.temporary_directory.name)

        colored_points = self.session.query(ColoredPoint).limit(500).all()
        points = [colored_point.point for colored_point in colored_points]
        colors = [colored_point.color for colored_point in colored_points]

        variables, dataframe = variables_and_dataframe_from_objects(points)
        model = JPT(variables, min_samples_leaf=0.4, min_impurity_improvement=0.05)
        model.fit(dataframe)
        self.model_loader.save_model(model.probabilistic_circuit, Point)

        variables, dataframe = variables_and_dataframe_from_objects(colors)
        model = JPT(variables, min_samples_leaf=0.4, min_impurity_improvement=0.05)
        model.fit(dataframe)
        self.model_loader.save_model(model.probabilistic_circuit, Color)

    def test_setup(self):
        self.assertEqual(self.temporary_directory.name, self.model_loader.folder_path)

    def test_load_model(self):
        model = self.model_loader.load_model(Point)
        self.assertIsInstance(model.root, JPT)

    def test_load_interaction_terms(self):
        model = self.model_loader.load_interaction_model([Point, Color])

    def test_save_model(self):
        self.assertSetEqual(set(os.listdir(self.temporary_directory.name)), {"Point.json", "Color.json"})
