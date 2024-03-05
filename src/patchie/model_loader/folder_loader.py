import json
import os
from typing import Type, List

from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
from sqlalchemy.orm import DeclarativeBase

from patchie.model_loader.model_loader import ModelLoader


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
        return f"{self.name_of_table(table)}{self.file_extension}"

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
