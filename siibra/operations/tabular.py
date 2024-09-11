import pandas as pd
import numpy as np
from io import BytesIO
from typing import List, Union

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
    input: Union[List[pd.DataFrame], List[pd.Series]]
    output: pd.DataFrame
    desc: str = "Concat dataframes/series into a single dataframe"
    type: str = "tabular/concat"

    def run(self, input, axis=None, **kwargs):
        assert isinstance(
            input, list
        ), f"Tabular/concat expected list, but got {type(input)}"
        assert all(
            isinstance(el, (pd.DataFrame, pd.Series)) for el in input
        ), f"Expected inputs to be list of dfs, but some was not"
        if axis is None:
            return pd.concat(input)
        return pd.concat(input, axis=axis)

    @classmethod
    def generate_specs(cls, axis=None, force=False, **kwargs):
        base = super().generate_specs(force, **kwargs)
        return {**base, "axis": axis}


class TabularMeanStd(DataOp):
    input: pd.DataFrame
    output: pd.DataFrame
    desc: str = "Get mean/std from input"
    type: str = "tabular/mean-std"

    def run(self, input, index=None, **kwargs):
        assert isinstance(input, pd.DataFrame)
        return pd.DataFrame(
            np.array(
                [
                    input.mean(axis=1).tolist(),
                    input.std(axis=1).tolist(),
                ]
            ).T,
            columns=["mean", "std"],
            index=index,
        )

    @classmethod
    def generate_specs(cls, index=None, force=False, **kwargs):
        base = super().generate_specs(force, **kwargs)
        return {**base, "index": index}
