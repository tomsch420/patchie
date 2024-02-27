import operator
from dataclasses import dataclass
from typing import Tuple

import networkx as nx
from matplotlib import pyplot as plt
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit)
from random_events.events import Event
from sqlalchemy.orm import DeclarativeBase

from sqlalchemy.sql.elements import ColumnElement, BinaryExpression, BooleanClauseList
from sqlalchemy.sql.selectable import Join
from sqlalchemy.sql.operators import in_op, not_in_op
from sqlalchemy import Table, select
from sqlalchemy.orm import Session

from typing_extensions import Type, List

from .utils import binary_expression_to_continuous_constraint
from .variables import Integer, Continuous, Symbolic, Column, SQLColumn, Variable, variables_and_dataframe_from_objects
from .model_loader import ModelLoader
import pandas as pd


@dataclass
class HyperParameters:

    min_samples_leaf: float = 0.1
    min_impurity_improvement: float = 0.
    max_leaves: float = float("inf")
    max_depth: float = float("inf")


class Patchie(ProbabilisticCircuit):

    model_loader: ModelLoader

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return super().variables

    def get_variable_that_matches_column(self, column: ColumnElement) -> Variable:
        """
        Get the variable that matches the column.
        :param column: The column to match.
        :return: The variable that matches the column.
        """
        wrapped_column = SQLColumn(column)
        for variable in self.variables:
            if variable.name == wrapped_column.name:
                return variable
        raise ValueError(f"No variable matches the column {column}")

    def event_from_binary_expression(self, binary_expression: BinaryExpression) -> Event:
        """
        Create an event from a binary expression.
        :param binary_expression: The expression to create the event from.
        :return: The event that is described by the binary expression.
        """
        event = Event()
        variable = self.get_variable_that_matches_column(binary_expression.left)

        if isinstance(variable, Continuous):
            constraint = binary_expression_to_continuous_constraint(binary_expression.operator,
                                                                    binary_expression.right.effective_value)
        elif isinstance(variable, (Integer, Symbolic)):
            if binary_expression.operator == operator.eq:
                constraint = {binary_expression.right.effective_value}
            elif binary_expression.operator == operator.ne:
                constraint = set(variable.domain) - {binary_expression.right.effective_value}
            elif binary_expression.operator is in_op:
                constraint = set(binary_expression.right.effective_value)
            elif binary_expression.operator is not_in_op:
                constraint = set(variable.domain) - set(binary_expression.right.effective_value)
            else:
                raise NotImplementedError(f"Operator {binary_expression.operator} not supported.")
        else:
            raise NotImplementedError(f"Variable type {type(variable)} not supported.")
        event[variable] = constraint
        return event

    def event_from_boolean_clause_list(self, boolean_clause_list: BooleanClauseList) -> Event:
        """
        Create an event from a boolean clause list.
        :param boolean_clause_list: The clause list to create the event from.
        :return: The event that is described by the clause list.
        """
        event = Event()
        for clause in boolean_clause_list.clauses:
            event = boolean_clause_list.operator(event, self.event_from_query(clause))
        return event

    def event_from_query(self, query: ColumnElement) -> Event:
        """
        Construct an event from a sql query.
        :param query: The query to parse.
        :return: The event that is described by the query.
        """
        if isinstance(query, BinaryExpression):
            return self.event_from_binary_expression(query)
        elif isinstance(query, BooleanClauseList):
            return self.event_from_boolean_clause_list(query)
        else:
            raise NotImplementedError(f"Query of type {type(query)} not supported.")

    def add_from_join_statement_to_graph(self, join_statement: Join, join_graph: nx.Graph):

        if isinstance(join_statement.left, Table) and isinstance(join_statement.right, Table):
            join_graph.add_edge(join_statement.left, join_statement.right)
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
            join_graph.add_edge(table, connecting_table)

        return table

    def load_models_from_join(self, join_statement: Join):
        join_graph = nx.Graph()
        self.add_from_join_statement_to_graph(join_statement, join_graph)
        assert nx.is_forest(join_graph), "The join statement must be a forest."
        print([(source.name, target.name) for source, target in join_graph.edges])
        models = dict()
        for table in join_graph.nodes:
            models[table] = self.model_loader.load_model(table)
            self.add_nodes_from(models[table].nodes)
            self.add_edges_from(models[table].unweighted_edges)
            self.add_weighted_edges_from(models[table].weighted_edges)
            print(self)

        for edge in join_graph.edges:
            source, target = edge
            source_model = models[source]
            target_model = models[target]
            interaction_model = self.model_loader.load_interaction_model([source, target])
            self.nodes[source_model].add_interaction_model(interaction_model, self.nodes[target_model])

    def fit_to_tables(self, session: Session, tables: List[Type[DeclarativeBase]]):
        """
        Fit the model to the tables and save them using the model loader.

        :param session: The session to use to fit the model.
        :param tables: The tables to fit the model to.
        """
        for table in tables:
            query = select(table)
            data = session.scalars(query).all()
            variables, dataframe = variables_and_dataframe_from_objects(data)
            model = JPT(variables, min_samples_leaf=0.2)
            model.fit(dataframe)
            self.model_loader.save_model(model.probabilistic_circuit, table)
