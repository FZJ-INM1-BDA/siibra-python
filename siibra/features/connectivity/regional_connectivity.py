# Copyright 2018-2021
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

from zipfile import ZipFile
from ..feature import Feature, Compoundable
from ..tabular.tabular import Tabular

from .. import anchor as _anchor

from ...commons import logger, QUIET
from ...core import region as _region
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector
from ...retrieval.requests import HttpRequest


import pandas as pd
import numpy as np
from typing import Callable, Union, List, Tuple, Iterator

try:
    from typing import Literal
except ImportError:  # support python 3.7
    from typing_extensions import Literal


class RegionalConnectivity(Feature, Compoundable):
    """
    Parcellation-averaged connectivity, providing one or more matrices of a
    given modality for a given parcellation.
    """

    _filter_attrs = ["modality", "cohort", "subject"]
    _compound_attrs = ["modality", "cohort"]

    def __init__(
        self,
        cohort: str,
        modality: str,
        regions: list,
        connector: RepositoryConnector,
        decode_func: Callable,
        filename: str,
        anchor: _anchor.AnatomicalAnchor,
        description: str = "",
        datasets: list = [],
        subject: str = "average",
        feature: str = None
    ):
        """
        Construct a parcellation-averaged connectivity matrix.

        Parameters
        ----------
        cohort: str
            Name of the cohort used for computing the connectivity.
        modality: str
            Connectivity modality, typically set by derived classes.
        regions: list[str]
            Names of the regions from the parcellation
        connector: Connector
            Repository connector for loading the actual data array(s).
        decode_func: function
            Function to convert the bytestream of a loaded file into an array
        files: dict
            A dictionary linking names of matrices (typically subject ids)
            to the relative filenames of the data array(s) in the repository connector.
        anchor: AnatomicalAnchor
            anatomical localization of the matrix, expected to encode the parcellation
            in the region attribute.
        description: str, optional
            textual description of this connectivity matrix.
        datasets : list[Dataset]
            list of datasets corresponding to this feature.
        """
        Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
        )
        self.cohort = cohort.upper()
        if isinstance(connector, str) and connector:
            self._connector = HttpRequest(connector, decode_func)
        else:
            self._connector = connector
        self._filename = filename
        self._decode_func = decode_func
        self.regions = regions
        self._matrix = None
        self._subject = subject
        self._feature = feature

    @property
    def subject(self):
        """Returns the subject identifiers for which the matrix represents."""
        return self._subject

    @property
    def feature(self):
        """If applicable, returns the type of feature for which the matrix represents."""
        return self._feature

    @property
    def name(self):
        return f"{super().name} with cohort {self.cohort} - {self.feature or self.subject}"

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a matrix as a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            A square matrix with region names as the column and row names.
        """
        if self._matrix is None:
            self._load_matrix()
        return self._matrix.copy()

    def _plot_matrix(
        self, regions: List[str] = None,
        logscale: bool = False, *args, backend="nilearn", **kwargs
    ):
        """
        Plots the heatmap of the connectivity matrix using nilearn.plotting.

        Parameters
        ----------
        regions: list[str]
            Display the matrix only for selected regions. By default, shows all the regions.
            It can only be a subset of regions of the feature.
        logscale: bool
            Display the data in log10 scale
        backend: str
            "nilearn" or "plotly"
        **kwargs:
            Can take all the arguments `nilearn.plotting.plot_matrix` can take. See the doc at
            https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_matrix.html
        """
        if regions is None:
            regions = self.regions
        indices = [self.regions.index(r) for r in regions]
        matrix = self.data.iloc[indices, indices].to_numpy()  # nilearn.plotting.plot_matrix works better with a numpy array

        if logscale:
            matrix = np.log10(matrix)

        # default kwargs
        kwargs["title"] = kwargs.get(
            "title",
            "".join([
                f"{self.feature if self.feature else ''} - {self.subject} - ",
                f"{self.modality} in ",
                f"{', '.join({_.name for _ in self.anchor.regions})}"
            ])
        )

        if kwargs.get("reorder") or (backend == "nilearn"):
            kwargs["figure"] = kwargs.get("figure", (15, 15))
            from nilearn import plotting
            plotting.plot_matrix(
                matrix,
                labels=regions,
                **kwargs
            )
        elif backend == "plotly":
            from plotly.express import imshow
            return imshow(matrix, *args, x=regions, y=regions, **kwargs)
        else:
            raise NotImplementedError(
                f"Plotting connectivity matrices with {backend} is not supported."
            )

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        if self.feature is None:
            fh.writestr(f"sub/{self._filename}/matrix.csv", self.data.to_csv())
        else:
            fh.writestr(f"feature/{self._filename}/matrix.csv", self.data.to_csv())

    def get_profile(
        self,
        region: Union[str, _region.Region],
        min_connectivity: float = 0,
        max_rows: int = None,
        direction: Literal['column', 'row'] = 'column'
    ):
        """
        Extract a regional profile from the matrix, to obtain a tabular data
        feature with the connectivity as the single column. Rows are be sorted
        by descending connection strength.

        Parameters
        ----------
        region: str, Region
        min_connectivity: float, default: 0
            Regions with connectivity less than this value are discarded.
        max_rows: int, default: None
            Max number of regions with highest connectivity.
        direction: str, default: 'column'
            Choose the direction of profile extraction particularly for
            non-symmetric matrices. ('column' or 'row')
        """
        matrix = self.data
        if direction.lower() not in ['column', 'row']:
            raise ValueError("Direction can only be 'column' or 'row'")
        if direction.lower() == 'row':
            matrix = matrix.transpose()

        def matches(r1, r2):
            if isinstance(r1, tuple):
                return any(r.matches(r2) for r in r1)
            else:
                assert isinstance(r1, _region.Region)
                return r1.matches(r2)

        regions = [r for r in matrix.index if matches(r, region)]
        if len(regions) == 0:
            raise ValueError(f"Invalid region specificiation: {region}")
        elif len(regions) > 1:
            raise ValueError(f"Region specification {region} matched more than one profile: {regions}")
        else:
            name = self.modality
            series = matrix[regions[0]]
            last_index = len(series) - 1 if max_rows is None \
                else min(max_rows, len(series) - 1)
            return Tabular(
                description=self.description,
                modality=f"{self.modality} {self.cohort}",
                anchor=_anchor.AnatomicalAnchor(
                    species=list(self.anchor.species)[0],
                    region=regions[0]
                ),
                data=(
                    series[:last_index]
                    .to_frame(name=name)
                    .query(f'`{name}` > {min_connectivity}')
                    .sort_values(by=name, ascending=False)
                    .rename_axis('Target regions')
                ),
                datasets=self.datasets
            )

    def plot(
        self,
        regions: Union[str, _region.Region, List[Union[str, _region.Region]]] = None,
        min_connectivity: float = 0,
        max_rows: int = None,
        direction: Literal['column', 'row'] = 'column',
        logscale: bool = False,
        *args,
        backend="matplotlib",
        **kwargs
    ):
        """
        Parameters
        ----------
        regions: Union[str, _region.Region], None
            If None, returns the full connectivity matrix.
            If a region is provided, returns the profile for that region.
            If list of regions is provided, returns the matrix for the selected
            regions.
        min_connectivity: float, default 0
            Only for region profile.
        max_rows: int, default None
            Only for region profile.
        direction: 'column' or 'row', default: 'column'
            Only for matrix.
        logscale: bool, default: False
        backend: str, default: "matplotlib" for profiles and "nilearn" for matrices
        """
        if regions is None or isinstance(regions, list):
            plot_matrix_backend = "nilearn" if backend == "matplotlib" else backend
            return self._plot_matrix(
                regions=regions, logscale=logscale, *args,
                backend=plot_matrix_backend,
                **kwargs
            )

        profile = self.get_profile(regions, min_connectivity, max_rows, direction)
        kwargs["kind"] = kwargs.get("kind", "barh")
        if backend == "matplotlib":
            kwargs["logx"] = kwargs.get("logx", logscale)
            return profile.data.plot(*args, backend=backend, **kwargs)
        elif backend == "plotly":
            kwargs.update({
                "color": kwargs.get("color", profile.data.columns[0]),
                "x": kwargs.get("x", profile.data.columns[0]),
                "y": kwargs.get("y", [r.name for r in profile.data.index]),
                "log_x": logscale,
                "labels": {"y": " ", "x": ""},
                "color_continuous_scale": "jet",
                "width": 600, "height": 15 * len(profile.data)
            })
            fig = profile.data.plot(*args, backend=backend, **kwargs)
            fig.update_layout({
                "font": dict(size=9),
                "yaxis": {"autorange": "reversed"},
                "coloraxis": {"colorbar": {
                    "orientation": "h", "title": "", "xpad": 0, "ypad": 10
                }},
                "margin": dict(l=0, r=0, b=0, t=0, pad=0)
            })
            return fig
        else:
            return profile.data.plot(*args, backend=backend, **kwargs)

    def get_profile_colorscale(
        self,
        region: Union[str, _region.Region],
        min_connectivity: float = 0,
        max_rows: int = None,
        direction: Literal['column', 'row'] = 'column',
        colorgradient: str = "jet"
    ) -> Iterator[Tuple[_region.Region, Tuple[int, int, int]]]:
        """
        Extract the colorscale corresponding to the regional profile from the
        matrix sorted by the values. See `get_profile` for further details.

        Note:
        -----
        Requires `plotly`.

        Parameters
        ----------
        region: str, Region
        min_connectivity: float, default: 0
            Regions with connectivity less than this value are discarded.
        max_rows: int, default: None
            Max number of regions with highest connectivity.
        direction: str, default: 'column'
            Choose the direction of profile extraction particularly for
            non-symmetric matrices. ('column' or 'row')
        colorgradient: str, default: 'jet'
            The gradient used to extract colorscale.
        Returns
        -------
        Iterator[Tuple[_region.Region, Tuple[int, int, int]]]
            Color values are in RGB 255.
        """
        from plotly.express.colors import sample_colorscale
        profile = self.get_profile(region, min_connectivity, max_rows, direction)
        normalized = profile.data / profile.data.max()
        colorscale = sample_colorscale(
            colorgradient,
            normalized.values.reshape(len(profile.data))
        )
        return zip(
            profile.data.index.values,
            [eval(c.removeprefix('rgb')) for c in colorscale]
        )

    def __len__(self):
        return len(self._filename)

    def __str__(self):
        return self.name

    def compute_centroids(self, space):
        """
        Computes the list of centroid coordinates corresponding to
        matrix rows, in the given reference space.

        Parameters
        ----------
        space: Space, str
        Returns
        -------
        list[tuple(float, float, float)]
        """
        result = []
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parcmap = next(iter(parcellations)).get_map(space)
        all_centroids = parcmap.compute_centroids()
        for regionname in self.regions:
            region = parcmap.parcellation.get_region(regionname, allow_tuple=True)
            if isinstance(region, tuple):  # deal with sets of matched regions
                found = [c for r in region for c in r if c.name in all_centroids]
            else:
                found = [r for r in region if r.name in all_centroids]
            assert len(found) > 0
            result.append(
                tuple(pointset.PointSet(
                    [all_centroids[r.name] for r in found], space=space
                ).centroid)
            )
        return result

    def _arraylike_to_dataframe(self, array: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert a numpy array with the connectivity matrix to
        a DataFrame with regions as column and row headers.
        """
        if not isinstance(array, np.ndarray):
            array = array.to_numpy()
        assert array.shape[0] == array.shape[1], f"Connectivity matrices must be square but found {array.shape}"
        if not (array == array.T).all():
            logger.warning("The connectivity matrix is not symmetric.")
        df = pd.DataFrame(array)
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        with QUIET:
            indexmap = {
                i: parc.get_region(regionname, allow_tuple=True)
                for i, regionname in enumerate(self.regions)
            }
        nrows = array.shape[0]
        try:
            assert len(indexmap) == nrows
            remapper = {
                label - min(indexmap.keys()): region
                for label, region in indexmap.items()
            }
            df = df.rename(index=remapper).rename(columns=remapper)
        except Exception:
            raise RuntimeError("Could not decode connectivity matrix regions.")
        return df

    def _load_matrix(self):
        """
        Extract connectivity matrix.
        """
        if isinstance(self._connector, HttpRequest):
            array = self._connector.data
        else:
            array = self._connector.get(self._filename, decode_func=self._decode_func)
        nrows = array.shape[0]
        if array.shape[1] != nrows:
            raise RuntimeError(
                f"Non-quadratic connectivity matrix {nrows}x{array.shape[1]} "
                f"from {self._filename} in {str(self._connector)}"
            )
        self._matrix = self._arraylike_to_dataframe(array)
