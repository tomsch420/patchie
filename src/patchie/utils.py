import portion
import random_events.variables
import operator

import probabilistic_model.learning.jpt.variables

from typing_extensions import Type, Any, Union, Set
from sqlalchemy import Column


def variable_type_of_column(column: Type[Column]) -> Type[random_events.variables.Variable]:

    # check if column is like a float
    if column.type.python_type == float:
        return random_events.variables.Continuous

    # check if column is like a string
    elif column.type.python_type == str:
        return random_events.variables.Symbolic

    # check if column is like an integer
    elif column.type.python_type == int:
        return random_events.variables.Integer

    else:
        raise NotImplementedError(f"Column type {column.type.python_type} not supported")


def jpt_variable_type_of_variable_type(variable_type: Type[random_events.variables.Variable],
                                       scale_continuous_types: bool = False) \
    -> Type[probabilistic_model.learning.jpt.variables.Variable]:
    """
    Convert a variable type to a jpt variable type.
    :param variable_type: The variable type to convert.
    :param scale_continuous_types: Whether to scale continuous types or not.
    :return: The jpt variable type
    """

    if variable_type is random_events.variables.Continuous:
        if scale_continuous_types:
            return probabilistic_model.learning.jpt.variables.ScaledContinuous
        else:
            return probabilistic_model.learning.jpt.variables.Continuous

    elif variable_type is random_events.variables.Symbolic:
        return variable_type

    elif variable_type is random_events.variables.Integer:
        return probabilistic_model.learning.jpt.variables.Integer

    else:
        raise NotImplementedError(f"Variable type {variable_type} not supported")


def binary_expression_to_continuous_constraint(operation: operator, value: Any) -> portion.Interval:
    """
    Converts a binary expression over a continuous variable to an interval.
    :param operation: The binary expression to convert.
    :param value: The value to convert.
    :return: The interval described by the constraint.
    """
    if operation == operator.gt:
        return portion.open(value, float("inf"))
    elif operation == operator.ge:
        return portion.closed(value, float("inf"))
    elif operation == operator.lt:
        return portion.open(float("-inf"), value)
    elif operation == operator.le:
        return portion.closed(float("-inf"), value)
    elif operation == operator.eq:
        return portion.singleton(value)
    else:
        raise ValueError(f"Operator {operator} in combination with value {value} not supported.")


def binary_expression_to_discrete_constraint(operation: operator, value: Any) -> Set:
    """
    Converts a binary expression over a discrete variable to a set.
    :param operation: The binary expression to convert.
    :param value: The value to convert.
    :return: The set described by the constraint.
    """
    if operation == operator.eq:
        return {value}
    if operation == operator.ne:
        return set(value).difference()
    else:
        raise ValueError(f"Operator {operator} in combination with value {value} not supported.")
