# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Union
import pandas as pd
from io import BytesIO

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .base import DataProvider
from ...operations.tabular import ParseAsTabular


@dataclass
class TabularDataProvider(DataProvider):
    schema: str = "siibra/attr/data/tabular/v0.1"
    format: Literal["csv"] = None
    plot_options: dict = field(default_factory=dict)
    parse_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.retrieval_ops) > 0:
            return
        super().__post_init__()
        self.retrieval_ops.append(
            ParseAsTabular.generate_specs(parse_options=self.parse_options)
        )

    def plot(self, *args, **kwargs):
        plot_options = self.plot_options.copy()
        data = self.get_data()
        if "sub_dataframe" in plot_options:
            sub_dataframe_arg = plot_options.pop("sub_dataframe")
            assert isinstance(
                sub_dataframe_arg, list
            ), f"sub_dataframe must be a list, but was {type(sub_dataframe_arg)}"
            for arg in sub_dataframe_arg:
                assert isinstance(
                    arg, str
                ), f"items in sub_dataframe must be str, but found {type(arg)}"
                data = data[arg]
            assert isinstance(
                data, pd.DataFrame
            ), f"after applying {sub_dataframe_arg}, expected result to be dataframe, but was {type(data)}"
        if "matrix" in plot_options:
            from ...commons.logger import logger

            try:
                from nilearn import plotting
            except ImportError as e:
                logger.error(f"Plotting matrix error: {str(e)}")
                return
            matrix_kwargs: Dict = plot_options.get("matrix").copy()
            matrix_kwargs.update(kwargs)
            return plotting.plot_matrix(data, *args, **matrix_kwargs)
        if "scatter" in plot_options:
            scatter_kwargs: Dict[str, Union[str, int, float]] = plot_options.get(
                "scatter"
            ).copy()
            scatter_kwargs.update(kwargs)
            return data.plot.scatter(*args, **scatter_kwargs)
        plot_options.update(kwargs)
        return data.plot(*args, **plot_options)

    def _iter_zippable(self):
        yield from super()._iter_zippable()
        bio = BytesIO()
        self.get_data().to_csv(bio)
        bio.seek(0)
        yield "Tabular data", ".tabular.csv", bio
