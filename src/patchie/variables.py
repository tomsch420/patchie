import pandas as pd
import portion
import sys

from probabilistic_model.learning.jpt.variables import (Integer as JPTInteger,
                                                        Continuous as JPTContinuous,
                                                        Variable as JPTVariable,
                                                        infer_variables_from_dataframe)
from probabilistic_model.utils import SubclassJSONSerializer

from random_events.variables import Symbolic as RESymbolic
from random_events.utils import get_full_class_name

from sqlalchemy import FromClause
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.schema import Table, ColumnElement
from typing_extensions import Union, List, Iterable, Tuple, Type, Any, Dict, Self, Optional
from sqlalchemy.ext.serializer import loads, dumps


class TableMixin:
    """
    Mixin Class for tables.
    """

    table: Table

    def __init__(self, table: Table):
        self.table = table


class Column:
    """
    Abstraction of sqlalchemy column elements that is needed for variables.
    """

    table_name: str
    column_name: str

    def __init__(self, table_name, column_name):
        self.table_name = table_name
        self.column_name = column_name

    @property
    def name(self):
        return f"{self.table_name}.{self.column_name}"

    def to_json(self) -> Dict[str, Any]:
        return {"column_name": self.column_name,
                "table_name": self.table_name}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        column_name = data["column_name"]
        table_name = data["table_name"]
        return cls(table_name, column_name)

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        return self.table_name == other.table_name and self.column_name == other.column_name


class SQLColumn(Column):

    column: ColumnElement
    """
    The column that is represented by the variable.
    """

    def __init__(self, column: ColumnElement):
        self.column = column

    @property
    def table_name(self):
        return f"{self.column.table.name}"

    @property
    def column_name(self):
        return self.column.name


class Variable(JPTVariable):

    column: Column
    """
    The column that this variable relates to.
    """

    def __init__(self, column: Column, domain: Any):
        self.column = column
        JPTVariable.__init__(self, self.column.name, domain)

    @property
    def name(self):
        return self.column.name

    @name.setter
    def name(self, value):
        ...

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "column": self.column.to_json()
        }


class Integer(Variable, JPTInteger):

    def __init__(self, column: Column, domain: Iterable, mean: float, std: float, ):
        Variable.__init__(self, column, domain)
        JPTInteger.__init__(self, self.name, domain, mean, std)

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        column = Column.from_json(data["column"])
        return cls(column, data["domain"], data["mean"], data["std"])


class Continuous(Variable, JPTContinuous):
    def __init__(self, column: Column, mean: float, std: float, minimal_distance: float = 1.,
                 min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10):
        Variable.__init__(self, column, domain=portion.open(-portion.inf, portion.inf))
        JPTContinuous.__init__(self, self.name, mean, std, minimal_distance, min_likelihood_improvement,
                               min_samples_per_quantile)

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        column = Column.from_json(data["column"])
        return cls(column, data["mean"], data["std"], data["minimal_distance"], data["min_likelihood_improvement"],
                   data["min_samples_per_quantile"])


class Symbolic(Variable, RESymbolic):

    def __init__(self, column: Column, domain: Iterable):
        Variable.__init__(self, column, domain)
        RESymbolic.__init__(self, self.name, domain)

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        column = Column.from_json(data["column"])
        return cls(column, data["domain"])


def relevant_columns_from(table: Union[Table, FromClause]) -> List[ColumnElement]:
    """
    Get the relevant columns from a table.
    :param table: The table to search in.
    :return: A list of relevant columns
    """

    # initialize result
    columns = []

    # iterate over all columns
    for column in table.columns:

        # check if column is primary key of foreign key
        if column.primary_key or column.foreign_keys:
            continue

        # append if relevant
        columns.append(column)

    return columns


def variables_and_dataframe_from_objects(objects: List[Type[DeclarativeBase]]) -> Tuple[
    List[Variable], pd.DataFrame]:
    """
    Get the variables and dataframe from a list of orm objects.
    :param objects: The objects to get the variables and dataframe from.
    :return: The variables and dataframe.
    """

    dataframe = dataframe_from_objects(objects)
    jpt_variables = infer_variables_from_dataframe(dataframe)
    table = objects[0].__table__

    columns = relevant_columns_from(table)

    new_variables = []

    for variable, column in zip(jpt_variables, columns):
        wrapped_column = SQLColumn(column)
        if isinstance(variable, JPTContinuous):
            new_variable = Continuous(wrapped_column, variable.mean, variable.std)

        elif isinstance(variable, JPTInteger):
            new_variable = Integer(wrapped_column, variable.domain, variable.mean, variable.std)

        elif isinstance(variable, RESymbolic):
            new_variable = Symbolic(wrapped_column, variable.domain)

        else:
            raise TypeError(f"Variable of type {type(variable)} not known.")

        new_variables.append(new_variable)

    column_names = [variable.name for variable in new_variables]

    dataframe.columns = column_names

    return new_variables, dataframe


def dataframe_from_objects(objects: List[Type[DeclarativeBase]]) -> pd.DataFrame:
    table = objects[0].__table__

    columns = relevant_columns_from(table)

    data = [[getattr(obj, column.name) for column in columns] for obj in objects]

    result = pd.DataFrame(data, columns=[column.name for column in columns])

    return result
