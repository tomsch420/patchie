import json
import os
from typing import Type

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
from sqlalchemy.orm import DeclarativeBase

from typing_extensions import Optional, List

from patchie.model_loader.model_loader import ModelLoader
import mlflow
import mlflow.pyfunc


class MLFlowWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for the MLFlow model persistence.
    """

    model: ProbabilisticCircuit

    def load_context(self, context):
        with open(context.artifacts["path"], "r") as file:
            self.model = ProbabilisticCircuit.from_json(json.load(file))


class MLFlowModelLoader(ModelLoader):
    """
    Class that loads models from a MLFlow tracking instance.
    """

    environment_variable_tracking_uri = "MLFLOW_TRACKING_URI"
    """
    Environment variable that contains the tracking uri.
    """

    default_tracking_uri = "http://localhost:5000"
    """
    Default tracking uri.
    """

    def __init__(self, tracking_uri: Optional[str] = None):

        # if no uri is provided
        if tracking_uri is None:

            # read from environment variable
            tracking_uri = os.environ.get(self.environment_variable_tracking_uri)
            # if environment variable is not set
            if tracking_uri is None:

                # default to localhost
                tracking_uri = self.default_tracking_uri

        # set tracking uri
        mlflow.set_tracking_uri(tracking_uri)

    def save_model(self, model: ProbabilisticCircuit, table: Type[DeclarativeBase]):
        table_name = self.name_of_table(table)
        with mlflow.start_run(run_name=table_name) as run:
            model_path = os.path.join("/", "tmp", table_name)
            with open(model_path, "w") as file:
                json.dump(model.to_json(), file)
            mlflow.pyfunc.log_model(artifact_path=table_name,
                                    python_model=MLFlowWrapper(),
                                    artifacts={"path": model_path},
                                    registered_model_name=table_name)

    def interaction_model_name(self, tables: List[Type[DeclarativeBase]]) -> str:
        table_names = sorted([self.name_of_table(table) for table in tables])
        table_names = "_".join(table_names)
        return table_names

    def load_model(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{self.name_of_table(table)}/latest")
        loaded_model = loaded_model.unwrap_python_model().model
        return loaded_model

    def save_interaction_model(self, model: ProbabilisticCircuit, tables: List[Type[DeclarativeBase]]):
        table_names = self.interaction_model_name(tables)
        with mlflow.start_run(run_name=table_names) as run:
            model_path = os.path.join("/", "tmp", table_names)
            with open(model_path, "w") as file:
                json.dump(model.to_json(), file)
            mlflow.pyfunc.log_model(artifact_path=table_names,
                                    python_model=MLFlowWrapper(),
                                    artifacts={"path": model_path},
                                    registered_model_name=table_names)

    def load_interaction_model(self, tables: List[Type[DeclarativeBase]]) -> ProbabilisticCircuit:
        loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{self.interaction_model_name(tables)}/latest")
        loaded_model = loaded_model.unwrap_python_model().model
        return loaded_model
