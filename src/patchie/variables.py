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
from sqlalchemy import FromClause, select, Select, Row

from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.orm.decl_api import DCTransformDeclarative
from sqlalchemy.sql.schema import Table, ColumnElement, ForeignKeyConstraint, ForeignKey, Column as AlchemyColumn
from sqlalchemy.sql.elements import Label
from typing_extensions import Union, List, Iterable, Tuple, Type, Any, Dict, Self, Optional
from sqlalchemy.ext.serializer import loads, dumps
from sqlalchemy_utils import get_referencing_foreign_keys


class TableMixin:
    """
    Mixin Class for tables.
    """

    table: Table

    def __init__(self, table: Table):
        self.table = table


class Column(SubclassJSONSerializer):
    """
    Abstraction of sqlalchemy column elements that is needed for variables.
    """

    table_name: str
    """
    The (aliased) name of the table that this column is in.
    """

    column_name: str
    """
    The (aliased) name of the column.
    """

    def __init__(self, table_name, column_name):
        self.table_name = table_name
        self.column_name = column_name

    @property
    def name(self):
        return f"{self.table_name}.{self.column_name}"

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(),
            "column_name": self.column_name,
            "table_name": self.table_name}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        column_name = data["column_name"]
        table_name = data["table_name"]
        return cls(table_name, column_name)

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        return self.table_name == other.table_name and self.column_name == other.column_name

    @classmethod
    def from_column_element(cls, column: ColumnElement) -> Self:
        if isinstance(column, Label):
            return LabeledColumn.from_label(column)
        else:
            return cls(column.table.name, column.name)


class LabeledColumn(Column):
    """
    A column that is labeled.
    """

    label: str
    """
    The label of the column.
    """

    def __init__(self, table_name, column_name, label):
        super().__init__(table_name, column_name)
        self.label = label

    @property
    def name(self):
        return self.label

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "label": self.label
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["table_name"],
                   data["column_name"],
                   data["label"])

    @classmethod
    def from_label(cls, label: Label) -> Self:
        return cls(label.element.table.name, label.name, label.key)

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

    @classmethod
    def from_jpt_variable_and_column(cls, variable: JPTVariable, column: Column) -> Self:
        """
        Create a Variable from a JPTVariable and a Column.
        :param variable: The JPTVariable to create the Variable from.
        :param column: The Column to use as the column of the Variable.
        :return: The Variable that is usable for the Patchie system.
        """
        if isinstance(variable, JPTContinuous):
            return Continuous(column, variable.mean, variable.std)

        elif isinstance(variable, JPTInteger):
            return Integer(column, variable.domain, variable.mean, variable.std)

        elif isinstance(variable, RESymbolic):
            return Symbolic(column, variable.domain)
        else:
            raise TypeError(f"Variable of type {type(variable)} not known.")


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


def relevant_columns_from(table: Union[Table, FromClause],
                          relation_depth: int = 1, origin_of_call: Optional[Table] = None)\
        -> Tuple[List[Label], List[Label], Select]:
    """
    Get the relevant columns from a table including relations.

    :param table: The table to search in.
    :param relation_depth: The depth of the relations to include.
    :param origin_of_call: The table that called this function. This is used to prevent useless cycles.

    :return: A list of relevant columns of this table, a list of relevant columns from all foreign tables and the query
    that joins all those columns together.
    """

    # initialize result
    columns: List[Label] = []
    foreign_columns: List[Label] = []

    # for every column in the table
    for column in table.columns:

        # type hinting
        column: AlchemyColumn

        # skip primary and foreign keys
        if column.primary_key or column.foreign_keys:
            continue

        # append and rename if relevant
        columns.append(column.label(f"{table.name}.{column.name}"))

    # construct query
    query: Select = select(*columns)

    # if the (remaining) relation depth is 0
    if relation_depth == 0:
        return columns, foreign_columns, query

    # for every foreign key that has this table as source
    for foreign_key in table.foreign_keys:

        query, current_foreign_columns = query_from_relevant_columns_result(table, foreign_key, query,
                                                                            relation_depth - 1, False)
        foreign_columns += current_foreign_columns

    # for every foreign key that has this table as target
    for foreign_key in get_referencing_foreign_keys(table):
        query, current_foreign_columns = query_from_relevant_columns_result(table, foreign_key, query,
                                                                            relation_depth - 1, True)
        foreign_columns += current_foreign_columns

    return columns, foreign_columns, query


def query_from_relevant_columns_result(current_table: Table,
                                       foreign_key: ForeignKey,
                                       query: Select,
                                       relation_depth: int,
                                       is_referencing_foreign_key: bool,):
    """
    Update the query with the relevant columns from the other table.

    :param current_table: The current table
    :param foreign_key: The foreign key relation with the other table
    :param is_referencing_foreign_key: Rather the current table is the target of the foreign key or not
    :param query: The query to extend
    :param relation_depth: The relational depth to pass into the recursion
    :return: The updated query and the foreign columns.
    """

    # if this is a call from target table of a foreign key
    if is_referencing_foreign_key:

        # the other table is the parent (source)
        other_table = foreign_key.parent.table
    else:

        # default
        other_table = foreign_key.column.table

    # if the other table is the same table as this one
    if other_table == current_table:

        # skip
        return query, []

    # get the relevant columns from the other table
    c, cj, foreign_query = relevant_columns_from(other_table, relation_depth, current_table)
    foreign_columns = [foreign_column.label(f"{current_table.name}.{foreign_column.name}") for foreign_column in c + cj]

    # join the foreign table
    query = query.join(foreign_key.column.table, foreign_key.parent == foreign_key.column)

    # add the joins from the foreign query
    query = query.add_columns(*foreign_columns)

    # add the joins from the foreign query
    for join_table, join_condition in foreign_query._setup_joins:
        query = query.join(join_table, join_condition)

    return query, foreign_columns


def variables_and_dataframe_from_columns_and_query(columns: List[Label], foreign_columns: List[Label],
                                                   query: Select, session: Session) \
        -> Tuple[List[Variable], List[Variable], pd.DataFrame]:
    """
    Get the variables and dataframe from a query result and columns.

    :param columns: The columns of the main table.
    :param foreign_columns: The columns of the foreign tables.
    :param query: The query to execute.
    :param session: The session to execute the query in.
    :return: The variables, foreign variables and dataframe.
    """

    # gather data
    dataframe = pd.read_sql_query(query, session.bind)

    # construct jpt variables
    jpt_variables = infer_variables_from_dataframe(dataframe)

    # convert columns to patchie variables
    variables = []
    for variable, column in zip(jpt_variables[:len(columns)], columns):
        variables.append(Variable.from_jpt_variable_and_column(variable, Column.from_column_element(column)))

    # convert foreign columns to patchie variables
    foreign_variables = []
    for variable, column in zip(jpt_variables[len(columns):], foreign_columns):
        foreign_variables.append(Variable.from_jpt_variable_and_column(variable, Column.from_column_element(column)))

    return variables, foreign_variables, dataframe
