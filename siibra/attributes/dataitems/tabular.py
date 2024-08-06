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
from typing import BinaryIO, Dict, Iterable, Tuple, Union
import pandas as pd
from io import BytesIO

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .base import Data

X_DATA = "x-siibra/data/dataframe"


@dataclass
class Tabular(Data):
    schema: str = "siibra/attr/data/tabular/v0.1"
    format: Literal["csv"] = None
    plot_options: dict = field(default_factory=dict)
    parse_options: dict = field(default_factory=dict)

    def get_data(self) -> pd.DataFrame:
        if X_DATA in self.extra:
            return self.extra[X_DATA]
        _bytes = super().get_data()
        if _bytes:
            return pd.read_csv(BytesIO(_bytes), **self.parse_options)
        return pd.read_csv(self.url, **self.parse_options)

    def plot(self, *args, **kwargs):
        if "matrix" in self.plot_options:
            from ...commons_new.logger import logger

            try:
                from nilearn import plotting
            except ImportError as e:
                logger.error(f"Plotting matrix error: {str(e)}")
                return
            matrix_kwargs: Dict = self.plot_options.get("matrix").copy()
            matrix_kwargs.update(kwargs)
            return plotting.plot_matrix(self.get_data(), *args, **matrix_kwargs)
        if "scatter" in self.plot_options:
            scatter_kwargs: Dict[str, Union[str, int, float]] = self.plot_options.get(
                "scatter"
            ).copy()
            scatter_kwargs.update(kwargs)
            return self.get_data().plot.scatter(*args, **scatter_kwargs)
        plot_kwargs = self.plot_options.copy()
        plot_kwargs.update(kwargs)
        return self.get_data().plot(*args, **plot_kwargs)

    def _iter_zippable(
        self,
    ) -> Iterable[Tuple[str, Union[str, None], Union[BinaryIO, None]]]:
        yield from super()._iter_zippable()
        bio = BytesIO()
        self.get_data().to_csv(bio)
        bio.seek(0)
        yield "Tabular data", ".tabular.csv", bio
