import operator

from random_events.events import Event
from random_events.variables import Continuous, Integer, Symbolic
from sqlalchemy import ColumnElement, BinaryExpression, BooleanClauseList
from sqlalchemy.sql.operators import in_op, not_in_op
from typing_extensions import List

from .utils import binary_expression_to_continuous_constraint
from .variables import Variable, Column


class QueryConverter:

    variables: List[Variable]

    def get_variable_that_matches_column(self, column: ColumnElement) -> Variable:
        """
        Get the variable that matches the column.
        :param column: The column to match.
        :return: The variable that matches the column.
        """
        wrapped_column = Column.from_column_element(column)
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
