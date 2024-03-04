from typing_extensions import Type, List
from sqlalchemy.orm import DeclarativeBase
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.distributions.multinomial import MultinomialDistribution
import os
import json
from sqlalchemy.orm.decl_api import DCTransformDeclarative
from sqlalchemy.sql.annotation import AnnotatedTable

class ModelLoader:
    """
    Class that serves as an interface for loading models that describe sql tables.
    """

    def name_of_table(self, table: Type[DeclarativeBase]) -> str:
        if isinstance(table, (DCTransformDeclarative, DeclarativeBase)):
            return f"{table.__tablename__}"
        elif isinstance(table, AnnotatedTable):
            return f"{table.name}"
        else:
            raise ValueError(f"Table {table} is not a valid table type. Type is {type(table)}.")

    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        """
        Load a model from a table.
        :param table: The table to load the model from.
        :return: The model that is described by the table.
        """
        raise NotImplementedError

    def save_model(self, model: ProbabilisticCircuit, table: Type[DeclarativeBase]):
        """
        Save a model of a table.
        :param model: The model to save.
        :param table: The table that describes the model.
        """
        raise NotImplementedError

    def load_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> MultinomialDistribution:
        """
        Load an interaction model that describes how to connect the latent variables.
        :param tables: The tables that are involved in the interaction
        :return: The model that is described by the table interaction.
        """
        raise NotImplementedError

    def save_interaction_model(self, model: ProbabilisticModel, tables: List[Type[DeclarativeBase]]):
        """
        Load an interaction model that describes how to connect the latent variables.

        :param model: The model to save.
        :param tables: The tables that are involved in the interaction
        :return: The model that is described by the table interaction.
        """
        raise NotImplementedError


class FolderModelLoader(ModelLoader):
    """
    Class that loads models from a folder.
    """

    folder_path: str
    file_extension: str

    def __init__(self, folder_path: str, file_extension: str = ".json"):
        self.folder_path = folder_path
        self.file_extension = file_extension

    def filename_of_table(self, table: Type[DeclarativeBase]) -> str:
        return f"{self.name_of_table(table)}.{self.file_extension}"

    def path_of_table(self, table: Type[DeclarativeBase]) -> str:
        return os.path.join(self.folder_path, f"{self.filename_of_table(table)}")

    def filename_of_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> str:
        table_names = sorted([self.name_of_table(table) for table in tables])
        table_names = "_".join(table_names)
        return f"interaction_{table_names}.{self.file_extension}"

    def path_of_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> str:
        return os.path.join(self.folder_path, self.filename_of_interaction_model(tables))

    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        with open(self.path_of_table(table), "r") as f:
            model = json.load(f)
        model = ProbabilisticCircuit.from_json(model)
        return model

    def save_model(self, model: ProbabilisticCircuit, table: Type[DeclarativeBase]):
        with open(self.path_of_table(table), "w") as f:
            json.dump(model.to_json(), f)

    def load_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> MultinomialDistribution:
        with open(self.path_of_interaction_model(tables), "r") as f:
            model = json.load(f)
        model = MultinomialDistribution.from_json(model)
        return model

    def save_interaction_model(self, model: MultinomialDistribution, tables: List[Type[DeclarativeBase]]):
        with open(self.path_of_interaction_model(tables), "w") as f:
            json.dump(model.to_json(), f)