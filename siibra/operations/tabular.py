import pandas as pd
from io import BytesIO
from typing import List

from .base import DataOp


class ParseAsTabular(DataOp):
    input: bytes
    output: pd.DataFrame
    desc: str = "Converting bytes to pandas dataframe"
    type: str = "tabular/parse"

    def run(self, input, parse_options=None, **kwargs):
        assert isinstance(
            input, bytes
        ), f"Tabular/Parse expected bytes, but got {type(input)}"
        return pd.read_csv(BytesIO(input), **parse_options)

    @classmethod
    def generate_specs(cls, parse_options=None, **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "parse_options": parse_options or {}}


class ConcatTabulars(DataOp):
    input: List[pd.DataFrame]
    output: pd.DataFrame
    desc: str = "Concat dataframes into a single dataframe"
    type: str = "tabular/concat"

    def run(self, input, **kwargs):
        assert isinstance(
            input, list
        ), f"Tabular/concat expected list, but got {type(input)}"
        assert all(
            isinstance(el, pd.DataFrame) for el in input
        ), f"Expected inputs to be list of dfs, but some was not"
        return pd.concat(input)


class GroupByTabular(DataOp):
    input: pd.DataFrame
    output: pd.DataFrame
    desc: str = "Group dataframes into mean/std"
    type: str = "tabular/groupby"

    def run(self, input, by: str, **kwargs):
        assert isinstance(
            input, pd.DataFrame
        ), f"Expected groupby input to be DataFrame, but was {type(input)}"
        return input.groupby(by).agg(["mean", "std"])

    @classmethod
    def generate_specs(cls, by: str, **kwargs):
        """
        Argument:

        by: str
            to be passed to DataFrame.groupby.
            see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html#pandas-dataframe-groupby

        """
        base = super().generate_specs(**kwargs)
        return {**base, "by": by}
