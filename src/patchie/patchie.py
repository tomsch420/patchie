import operator
from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np
from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                                             ProbabilisticCircuitMixin)
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from random_events.events import Event, EncodedEvent
from random_events.variables import Symbolic as RESymbolic
from sqlalchemy.orm import DeclarativeBase

from sqlalchemy.sql.elements import ColumnElement, BinaryExpression, BooleanClauseList
from sqlalchemy.sql.selectable import Join
from sqlalchemy.sql.operators import in_op, not_in_op
from sqlalchemy import Table, select
from sqlalchemy.orm import Session

from typing_extensions import Type, List, Union, Optional

from .utils import binary_expression_to_continuous_constraint
from .variables import (Integer, Continuous, Symbolic, Column, Variable, relevant_columns_from,
                        variables_and_dataframe_from_columns_and_query)
from .model_loader.model_loader import ModelLoader
from .query_converter import QueryConverter
import pandas as pd
import tqdm
from .constants import tomato_red
from probabilistic_model.bayesian_network.distributions import (DiscreteDistribution, ConditionalProbabilisticCircuit,
                                                                ConditionalProbabilityTable)


@dataclass
class HyperParameters:

    min_samples_leaf: float = 0.1
    min_impurity_improvement: float = 0.
    max_leaves: float = float("inf")
    max_depth: float = float("inf")


class Patchie(ProbabilisticCircuit, QueryConverter):

    model_loader: Optional[ModelLoader]
    """
    The interface used to load and save models and interaction models.
    """

    relation_depth: int = 1
    """
    The depth of relations to consider when learning models.
    In learning, the data is gathered from the table of consideration and all tables that are reachable by a join.
    This is done recursively, until tables require more than ``relation_depth`` joins to be reached.
    """

    def __init__(self, model_loader: Optional[ModelLoader] = None, relation_depth: int = 1):
        super().__init__()
        self.model_loader = model_loader
        self.relation_depth = relation_depth

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return super().variables

    def preprocess_event(self, event: Union[Event, ColumnElement]) -> EncodedEvent:
        if isinstance(event, Event):
            return super().preprocess_event(event)
        elif isinstance(event, ColumnElement):
            return self.preprocess_event(self.event_from_query(event))
        else:
            raise ValueError(f"Event of type {type(event)} can not be processed.")

    def add_from_join_statement_to_graph(self, join_statement: Join, join_graph: nx.DiGraph):

        if isinstance(join_statement.left, Table) and isinstance(join_statement.right, Table):
            join_graph.add_edge(join_statement.right, join_statement.left)
            return join_statement.right

        if isinstance(join_statement.left, Table):
            table = join_statement.left
            print(type(join_statement.right))
            assert isinstance(join_statement.right, Join)
            recursive_join_statement = join_statement.right

        elif isinstance(join_statement.right, Table):
            table = join_statement.right
            assert isinstance(join_statement.left, Join)
            recursive_join_statement = join_statement.left

        else:
            raise ValueError(f"At least on of the join statements must be a table. "
                             f"Got {join_statement}.")

        join_graph.add_node(table)

        if recursive_join_statement is not None:
            connecting_table = self.add_from_join_statement_to_graph(recursive_join_statement, join_graph)
            join_graph.add_edge(connecting_table, table)

        return table

    def bayesian_network_from_join(self, join_statement: Join) -> Tuple[BayesianNetwork, List[Symbolic]]:
        """
        Create a bayesian network from a join statement using the model loader to access the necessary models
        and interactions. The join statement has to be a tree.

        :param join_statement: The join statement to create the bayesian network from.
        :return: The bayesian network that is described by the join statement.
        """
        join_graph = nx.DiGraph()
        self.add_from_join_statement_to_graph(join_statement, join_graph)

        assert nx.is_tree(join_graph), "The join statement must be a tree."

        bayesian_network = BayesianNetwork()

        latent_nodes = dict()
        latent_variables = list()

        for table, in_degree in join_graph.in_degree:
            model = self.model_loader.load_model(table)

            # construct latent node
            latent_variable = self.latent_variable(table, model)
            latent_variables.append(latent_variable)
            if in_degree == 0:
                latent_node = DiscreteDistribution(latent_variable, model.root.weights)
            else:
                latent_node = ConditionalProbabilityTable(latent_variable)
            bayesian_network.add_node(latent_node)
            latent_nodes[table] = latent_node

            # construct model node
            model_node = ConditionalProbabilisticCircuit(model.variables)
            model_node = model_node.from_unit(model.root)
            bayesian_network.add_node(model_node)

            # connect latent and model node
            bayesian_network.add_edge(latent_node, model_node)

        for edge in join_graph.edges:
            source, target = edge
            interaction_model = self.model_loader.load_interaction_model([source, target])
            target_node = latent_nodes[target]
            source_node = latent_nodes[source]
            bayesian_network.add_edge(source_node, target_node)

            target_node.from_multinomial_distribution(interaction_model)

        return bayesian_network, latent_variables

    def load_from_join(self, join_statement: Join):
        """
        Load the model from the join statement.
        The model is loaded as probabilistic circuit and added to this model nodes.
        All latent variables are marginalized.

        :param join_statement: The join statement to load the model from.
        """
        bayesian_network, latent_variables = self.bayesian_network_from_join(join_statement)
        pc = bayesian_network.as_probabilistic_circuit().simplify()
        marginal_variables = [variable for variable in pc.variables if variable not in latent_variables]
        pc = pc.marginal(marginal_variables)
        self.add_nodes_from(pc.nodes)
        self.add_edges_from(pc.unweighted_edges)
        self.add_weighted_edges_from(pc.weighted_edges)

    def load_table(self, table: Type[DeclarativeBase]) -> ProbabilisticCircuitMixin:
        """
        Load the model from the table.
        The model is loaded as probabilistic circuit and added to this model nodes.

        :param table: The table to load the model from.
        :return: The root of the loaded model.
        """
        pc = self.model_loader.load_model(table)
        self.add_nodes_from(pc.nodes)
        self.add_edges_from(pc.unweighted_edges)
        self.add_weighted_edges_from(pc.weighted_edges)
        return pc.root

    def fit_to_tables(self, session: Session, tables: List[Type[DeclarativeBase]]):
        """
        Fit the model to the tables and save them using the model loader.

        :param session: The session to use to fit the model.
        :param tables: The tables to fit the model to.
        """
        for table in tables:
            model = self.fit_to_table(session, table)
            self.model_loader.save_model(model, table)

    def fit_to_table(self, session, table: Type[DeclarativeBase]) -> ProbabilisticCircuit:
        """
        Fit the model to a table and save it using the model loader.

        :param session: The session used to get the data for the fitting.
        :param table: The table to fit the model to.
        """
        columns, foreign_columns, query = relevant_columns_from(table.__table__, self.relation_depth)
        variables, foreign_variables, dataframe = variables_and_dataframe_from_columns_and_query(
            columns, foreign_columns, query, session)
        model = JPT(variables + foreign_variables, targets=variables + foreign_variables, features=variables,
                    min_samples_leaf=0.2)
        model = model.fit(dataframe).marginal(variables, simplify_if_univariate=False, as_deterministic_sum=True)
        return model.probabilistic_circuit

    def latent_variable(self, table: Type[DeclarativeBase], model: ProbabilisticCircuit):
        """
        :return: A latent variable for the table with domain as index set of the models root subcircuits.
        """
        return RESymbolic(f"{self.model_loader.name_of_table(table)}.latent",
                          domain=range(len(model.root.subcircuits)))

    def fit_interaction_model(self, session: Session, table_1: Type[DeclarativeBase], table_2: Type[DeclarativeBase]):
        """
        Fit an interaction term between the two tables. The tables must be join-able.
        The models that are associated with the tables must be deterministic sum units.
        The variables in said models must be the same that are obtained by joining the tables.

        :param session: The session used to get the data for the fitting.
        :param table_1: One of the tables of the interaction term.
        :param table_2: The other table of the interaction term.

        :return: The fitted interaction term.
        """

        # load both models
        model_1 = self.model_loader.load_model(table_1)
        model_2 = self.model_loader.load_model(table_2)

        # enforce that both models are deterministic sum units
        assert isinstance(model_1.root, DeterministicSumUnit)
        assert isinstance(model_2.root, DeterministicSumUnit)

        # create the latent variables
        latent_variable_1 = self.latent_variable(table_1, model_1)
        latent_variable_2 = self.latent_variable(table_2, model_2)

        # get the relevant columns from the tables
        columns_1, _, _ = relevant_columns_from(table_1.__table__, 0)
        columns_2, _, _ = relevant_columns_from(table_2.__table__, 0)

        # get the joined data from the tables
        query = select(*columns_1, *columns_2).join(table_2)
        dataframe = pd.read_sql_query(query, session.bind)

        # check that the columns are the same as the variables
        assert set(dataframe.columns) == set([variable.name for variable in model_1.variables + model_2.variables])

        # initialize encodings
        encoded_values = np.empty((len(dataframe), 2), dtype=int)
        for index, sample in tqdm.tqdm(dataframe.iterrows(), total=len(dataframe),
                                       desc=f"Encoding samples for interaction between {table_1.__tablename__} "
                                            f"and {table_2.__tablename__}", colour=tomato_red):

            # get the relevant columns from the respective samples
            sample_for_model_1 = sample[[variable.name for variable in model_1.variables]].tolist()
            sample_for_model_2 = sample[[variable.name for variable in model_2.variables]].tolist()

            # encode the values of the dataframe into subcircuit indices
            encoded_values[index, 0] = model_1.root.sub_circuit_index_of_sample(sample_for_model_1)
            encoded_values[index, 1] = model_2.root.sub_circuit_index_of_sample(sample_for_model_2)

        # create the interaction model
        interaction_model = MultinomialDistribution([latent_variable_1, latent_variable_2])
        interaction_model._variables = [latent_variable_1, latent_variable_2]

        # fit the interaction model
        interaction_model._fit(encoded_values.tolist())
        return interaction_model