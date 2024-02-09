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

        file_1_name = points[0].__tablename__ + ".json"
        file_1_path = os.path.join(self.temporary_directory.name, file_1_name)
        with open(file_1_path, "w") as f:
            json.dump(model.probabilistic_circuit.to_json(), f)

        variables, dataframe = variables_and_dataframe_from_objects(colors)
        model = JPT(variables, min_samples_leaf=0.4, min_impurity_improvement=0.05)
        model.fit(dataframe)

        file_2_name = colors[0].__tablename__ + ".json"
        file_2_path = os.path.join(self.temporary_directory.name, file_2_name)

        with open(file_2_path, "w") as f:
            json.dump(model.probabilistic_circuit.to_json(), f)

    def test_setup(self):
        self.assertEqual(self.temporary_directory.name, self.model_loader.folder_path)

    def test_load_model(self):
        model = self.model_loader.load_model(Point)
        print(model)
