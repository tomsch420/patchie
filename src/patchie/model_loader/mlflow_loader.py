import os
from typing import Type

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
from sqlalchemy.orm import DeclarativeBase

from typing_extensions import Optional

from patchie.model_loader.model_loader import ModelLoader
import mlflow
import mlflow.pyfunc


class MLFlowModelLoader(ModelLoader):
    """
    Class that loads models from a MLFlow tracking instance.
    """

    def __init__(self, tracking_uri: Optional[str] = None):

        # if no uri is provided
        if tracking_uri is None:

            # read from environment variable
            # tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

            # if environment variable is not set
            if tracking_uri is None:

                # default to localhost
                tracking_uri = "http://localhost:5000"

        # set tracking uri
        mlflow.set_tracking_uri(tracking_uri)

    def save_model(self, model: ProbabilisticCircuit, table: Type[DeclarativeBase]):
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(model.to_json(), self.name_of_table(table))


    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        ...

    def save_interaction_model(self, model: ProbabilisticCircuit, tables: Type[DeclarativeBase]):
        ...

    def load_interaction_model(self, tables: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        ...
