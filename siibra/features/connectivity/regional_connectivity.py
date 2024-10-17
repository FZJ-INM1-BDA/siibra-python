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
from ..feature import Feature
from ..tabular.tabular import Tabular

from .. import anchor as _anchor

from ...commons import logger, QUIET, siibra_tqdm
from ...core import region as _region
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector

import pandas as pd
import numpy as np
from typing import Callable, Dict, Union, List

try:
    from typing import Literal
except ImportError:  # support python 3.7
    from typing_extensions import Literal


class RegionalConnectivity(Feature):
    """
    Parcellation-averaged connectivity, providing one or more matrices of a
    given modality for a given parcellation.
    """

    def __init__(
        self,
        cohort: str,
        modality: str,
        regions: list,
        connector: RepositoryConnector,
        decode_func: Callable,
        files: Dict[str, str],
        anchor: _anchor.AnatomicalAnchor,
        description: str = "",
        datasets: list = [],
        prerelease: bool = False,
        id: str = None,
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
            description=description or '\n'.join({ds.description for ds in datasets}),
            anchor=anchor,
            datasets=datasets,
            prerelease=prerelease,
            id=id,
        )
        self.cohort = cohort.upper()
        self._connector = connector
        self._files = files
        self._decode_func = decode_func
        self.regions = regions
        self._matrices = {}

    @property
    def subjects(self):
        """
        Returns the subject identifiers for which matrices are available.
        """
        return list(self._files.keys())

    @property
    def name(self):
        supername = super().name
        return f"{supername} with cohort {self.cohort}"

    def get_matrix(self, subject: str = None):
        """
        Returns a matrix as a pandas dataframe.

        Parameters
        ----------
        subject: str, default: None
            Name of the subject (see ConnectivityMatrix.subjects for available names).
            If None, the mean is taken in case of multiple available matrices.
        Returns
        -------
        pd.DataFrame
            A square matrix with region names as the column and row names.
        """
        assert len(self) > 0
        if (subject is None) and (len(self) > 1):
            # multiple matrices available, but no subject given - return mean matrix
            logger.info(
                f"No subject name supplied, returning mean connectivity across {len(self)} subjects. "
                "You might alternatively specify an individual subject."
            )
            if "mean" not in self._matrices:
                all_arrays = [
                    self._connector.get(fname, decode_func=self._decode_func)
                    for fname in siibra_tqdm(
                        self._files.values(),
                        total=len(self),
                        desc=f"Averaging {len(self)} connectivity matrices"
                    )
                ]
                self._matrices['mean'] = self._arraylike_to_dataframe(np.stack(all_arrays).mean(0))
            return self._matrices['mean'].copy()
        if subject is None:
            subject = next(iter(self._files.keys()))
        if subject not in self._files:
            raise ValueError(f"Subject name '{subject}' not known, use one of: {', '.join(self._files)}")
        if subject not in self._matrices:
            self._matrices[subject] = self._load_matrix(subject)
        return self._matrices[subject].copy()

    def plot_matrix(
        self, subject: str = None, regions: List[str] = None,
        logscale: bool = False, *args, backend="nilearn", **kwargs
    ):
        """
        Plots the heatmap of the connectivity matrix using nilearn.plotting.

        Parameters
        ----------
        subject: str
            Name of the subject (see ConnectivityMatrix.subjects for available names).
            If "mean" or None is given, the mean is taken in case of multiple
            available matrices.
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
        matrix = self.get_matrix(subject=subject).iloc[indices, indices].to_numpy()  # nilearn.plotting.plot_matrix works better with a numpy array

        if logscale:
            matrix = np.log10(matrix)

        # default kwargs
        subject_title = subject or ""
        kwargs["title"] = kwargs.get(
            "title",
            f"{subject_title} - {self.modality} in {', '.join({_.name for _ in self.anchor.regions})}"
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

    def __iter__(self):
        return ((sid, self.get_matrix(sid)) for sid in self._files)

    def _export(self, fh: ZipFile):
        super()._export(fh)
        for sub in self.subjects:
            df = self.get_matrix(sub)
            fh.writestr(f"sub/{sub}/matrix.csv", df.to_csv())

    def get_profile(
        self,
        region: Union[str, _region.Region],
        subject: str = None,
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
        subject: str, default: None
        min_connectivity: float, default: 0
            Regions with connectivity less than this value are discarded.
        max_rows: int, default: None
            Max number of regions with highest connectivity.
        direction: str, default: 'column'
            Choose the direction of profile extraction particularly for
            non-symmetric matrices. ('column' or 'row')
        """
        matrix = self.get_matrix(subject)
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
            name = \
                f"Averaged {self.modality}" if subject is None \
                else f"{self.modality}"
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

    def plot_profile(
        self,
        region: Union[str, _region.Region],
        subject: str = None,
        min_connectivity: float = 0,
        max_rows: int = None,
        direction: Literal['column', 'row'] = 'column',
        logscale: bool = False,
        *args,
        backend="matplotlib",
        **kwargs
    ):
        profile = self.get_profile(region, subject, min_connectivity, max_rows, direction)
        kwargs["kind"] = kwargs.get("kind", "barh")
        if backend == "matplotlib":
            kwargs["logx"] = kwargs.get("logx", logscale)
        elif backend == "plotly":
            kwargs.update({
                "color": kwargs.get("color", profile.data.columns[0]),
                "x": kwargs.get("x", profile.data.columns[0]),
                "y": kwargs.get("y", [r.name for r in profile.data.index]),
                "log_x": logscale,
                "labels": {"y": " ", "x": ""},
                "color_continuous_scale": "jet",
                "width": 600, "height": 3800
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
        return profile.plot(*args, backend=backend, **kwargs)

    def __len__(self):
        return len(self._files)

    def __str__(self):
        return "{} connectivity for {} from {} cohort ({} matrices)".format(
            self.paradigm if hasattr(self, "paradigm") else self.modality,
            "_".join(p.name for p in self.anchor.parcellations),
            self.cohort,
            len(self._files),
        )

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

    def _load_matrix(self, subject: str):
        """
        Extract connectivity matrix.
        """
        assert subject in self.subjects
        array = self._connector.get(self._files[subject], decode_func=self._decode_func)
        nrows = array.shape[0]
        if array.shape[1] != nrows:
            raise RuntimeError(
                f"Non-quadratic connectivity matrix {nrows}x{array.shape[1]} "
                f"from {self._files[subject]} in {str(self._connector)}"
            )
        return self._arraylike_to_dataframe(array)
