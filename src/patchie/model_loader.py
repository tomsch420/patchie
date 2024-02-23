from typing_extensions import Type, List
from sqlalchemy.orm import DeclarativeBase
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
import os
import json


class ModelLoader:
    """
    Class that serves as an interface for loading models that describe sql tables.
    """

    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        """
        Load a model from a table.
        :param table: The table to load the model from.
        :return: The model that is described by the table.
        """
        raise NotImplementedError

    def load_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> ProbabilisticModel:
        """
        Load an interaction model that describes how to connect the latent variables.
        :param tables: The tables that are involved in the interaction
        :return: The model that is described by the table interaction.
        """
        raise NotImplementedError


class FolderModelLoader(ModelLoader):
    """
    Class that loads models from a folder.
    """

    folder_path: str

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        for file in os.listdir(self.folder_path):
            if file.startswith(table.__tablename__):
                filename = os.path.join(self.folder_path, file)
                with open(filename, "r") as f:
                    model = json.load(f)
                model = ProbabilisticCircuit.from_json(model)
                return model
        raise ValueError(f"No model found for table {table.__tablename__}")

    def load_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> ProbabilisticModel:
        variables = [f"{table.__tablename__}.latent" for table in tables]
        print(variables)
