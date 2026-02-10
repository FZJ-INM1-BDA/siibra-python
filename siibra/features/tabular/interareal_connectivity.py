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

import json
from textwrap import wrap
from zipfile import ZipFile
from io import BytesIO

from ..feature import Feature
from ..tabular import tabular

from .. import anchor as _anchor

from ...commons import logger, QUIET, siibra_tqdm
from ...core import region as _region
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector, ZipfileConnector
from ...retrieval.cache import CACHE

import pandas as pd
import numpy as np
from typing import Callable, Dict, Union, List

try:
    from typing import Literal
except ImportError:  # support python 3.7
    from typing_extensions import Literal

def name_to_code(name):
    map_ = {
        "areas 1 and 2 of cortex": "A1-2",
        "area 10 of cortex": "A10",
        "area 11 of cortex": "A11",
        "area 13 of cortex, lateral part": "A13L",
        "area 13 of cortex, medial part": "A13M",
        "area 13a of cortex": "A13a",
        "area 13b of cortex": "A13b",
        "area 14 of cortex, caudal part": "A14C",
        "area 14 of cortex, rostral part": "A14R",
        "area 19 of cortex, dorsointermediate part": "A19DI",
        "area 19 of cortex, medial part": "A19M",
        "area 23 of cortex, ventral part": "A23V",
        "area 23a of cortex": "A23a",
        "area 23b of cortex": "A23b",
        "area 23c of cortex": "A23c",
        "area 24a of cortex": "A24a",
        "area 24b of cortex": "A24b",
        "area 24c of cortex": "A24c",
        "area 24d of cortex": "A24d",
        "area 25 of cortex": "A25",
        "area 29a-c of cortex": "A29a-c",
        "area 29d of cortex": "A29d",
        "area 30 of cortex": "A30",
        "area 31 of cortex": "A31",
        "area 32 of cortex": "A32",
        "area 32 of cortex, ventral part": "A32V",
        "area 35 of cortex": "A35",
        "area 36 of cortex": "A36",
        "area 3a of cortex (somatosensory)": "A3a",
        "area 3b of cortex (somatosensory)": "A3b",
        "area 45 of cortex": "A45",
        "area 46 of cortex, dorsal part": "A46D",
        "area 46 of cortex, ventral part": "A46V",
        "area 47 (old 12) of cortex, lateral part": "A47L",
        "area 47 (old 12) of cortex, medial part": "A47M",
        "area 47 (old 12) of cortex, orbital part": "A47O",
        "area 4 of cortex, parts a and b (primary motor)": "A4ab",
        "area 4 of cortex, part c (primary motor)": "A4c",
        "area 6 of cortex, dorsocaudal part": "A6DC",
        "area 6 of cortex, dorsorostral part": "A6DR",
        "area 6 of cortex, medial (supplementary motor) part": "A6M",
        "area 6 of cortex, ventral, part a": "A6Va",
        "area 6 of cortex, ventral, part b": "A6Vb",
        "area 8 of cortex, caudal part": "A8C",
        "area 8a of cortex, dorsal part": "A8aD",
        "area 8a of cortex, ventral part": "A8aV",
        "area 8b of cortex": "A8b",
        "area 9 of cortex": "A9",
        "agranular insular cortex": "AI",
        "anterior intraparietal area of cortex": "AIP",
        "amygdalopiriform transition area": "APir",
        "auditory cortex, primary area": "AuA1",
        "auditory cortex, anterolateral area": "AuAL",
        "auditory cortex, caudolateral area": "AuCL",
        "auditory cortex, caudomedial area": "AuCM",
        "auditory cortex, caudal parabelt area": "AuCPB",
        "auditory cortex, middle lateral area": "AuML",
        "auditory cortex, rostral area": "AuR",
        "auditory cortex, rostromedial area": "AuRM",
        "auditory cortex, rostral parabelt": "AuRPB",
        "auditory cortex, rostrotemporal": "AuRT",
        "auditory cortex, rostrotemporal lateral area": "AuRTL",
        "auditory cortex, rostrotemporal medial area": "AuRTM",
        "dysgranular insular cortex": "DI",
        "entorhinal cortex": "Ent",
        "fundus of superior temporal sulcus area of cortex": "FST",
        "granular insular cortex": "GI",
        "gustatory cortex": "Gu",
        "insular proisocortex": "IPro",
        "lateral intraparietal area of cortex": "LIP",
        "medial intraparietal area of cortex": "MIP",
        "medial superior temporal area of cortex": "MST",
        "orbital periallocortex": "OPAl",
        "orbital proisocortex": "OPro",
        "occipito-parietal transitional area of cortex": "OPt",
        "parietal area PE": "PE",
        "parietal area PE, caudal part": "PEC",
        "parietal area PF (cortex)": "PF",
        "parietal area PFG (cortex)": "PFG",
        "parietal area PG": "PG",
        "parietal area PG, medial part (cortex)": "PGM",
        "parietal areas PGa and IPa (fundus of superior temporal ventral area)": "PGa-IPa",
        "parainsular cortex, lateral part": "PaIL",
        "parainsular cortex, medial part": "PaIM",
        "piriform cortex": "Pir",
        "proisocortical motor region (precentral opercular cortex)": "ProM",
        "prostriate area": "ProSt",
        "retroinsular area (cortex)": "ReI",
        "secondary somatosensory cortex, external part": "S2E",
        "secondary somatosensory cortex, internal part": "S2I",
        "secondary somatosensory cortex, parietal rostral area": "S2PR",
        "secondary somatosensory cortex, parietal ventral area": "S2PV",
        "superior temporal rostral area (cortex)": "STR",
        "temporal area TE1 (inferior temporal cortex)": "TE1",
        "temporal area TE2 (inferior temporal cortex)": "TE2",
        "temporal area TE3 (inferior temporal cortex)": "TE3",
        "temporal area TE, occipital part": "TEO",
        "temporal area TF": "TF",
        "temporal area TF, occipital part": "TFO",
        "temporal area TH": "TH",
        "temporal area TL": "TL",
        "temporal area TL, occipital part": "TLO",
        "temporo-parieto-occipital association area (superior temporal polysensory cortex)": "TPO",
        "temporopolar proisocortex": "TPPro",
        "temporal proisocortex": "TPro",
        "temporoparietal transitional area": "TPt",
        "primary visual cortex": "V1",
        "visual area 2": "V2",
        "visual area 3 (ventrolateral posterior area)": "V3",
        "visual area 3A (dorsoanterior area)": "V3A",
        "visual area 4 (ventrolatereral anterior area)": "V4",
        "visual area 4, transitional part (middle temporal crescent)": "V4T",
        "visual area 5 (middle temporal area)": "V5",
        "visual area 6 (dorsomedial area)": "V6",
        "visual area 6A (posterior parietal medial area)": "V6A",
        "ventral intraparietal area of cortex": "VIP"
    }
    return map_[name]


class DFWithMeta(pd.DataFrame):
    _metadata = ['meta']
    @property
    def _constructor(self):
        return DFWithMeta

class InterarealConnectivityMatrix(
    tabular.Tabular,
    configuration_folder="features/tabular/connectivitystrength",
    category="cellular",
):
    """
    Parcellation-averaged connectivity, providing one or more matrices of a
    given modality for a given parcellation.
    """

    @classmethod
    def decode_meta(cls, spec):
        decoder_spec = spec.get("decoder", {})
        if decoder_spec["@type"].endswith('csv'):
            kwargs = {k: v for k, v in decoder_spec.items() if k != "@type"}
            return lambda b: pd.read_csv(BytesIO(b), **kwargs)
        else:
            return None


    class ConnectivityConnector(ZipfileConnector):

        class ZipFileLoader:
            """
            Loads a file from the zip archive, but mimics the behaviour
            of cached http requests used in other connectors.
            """
            def __init__(self, zipfile, filename, decode_func, meta=None):
                self.zipfile = zipfile
                self.filename = filename
                self.func = decode_func
                self.cachefile = CACHE.build_filename(zipfile + filename)
                self.meta = meta

            @property
            def cached(self):
                return os.path.isfile(self.cachefile)

            @property
            def data(self):
                container = ZipFile(self.zipfile)
                df = self.func(container.open(self.filename).read())
                if self.meta is not None:
                    df = DFWithMeta(df)
                    df.meta = self.meta
                return df


        def get_loader(self, filename, folder="", decode_func=None):
            """Get a lazy loader for a file, for loading data
            only once loader.data is accessed."""
            meta = self.ZipFileLoader(self.zipfile, 'meta.json', lambda b: json.loads(b)).data
            loader = self.ZipFileLoader(self.zipfile, filename, decode_func, meta)
            return loader

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
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description or '\n'.join({ds.description for ds in datasets}),
            anchor=anchor,
            datasets=datasets,
            data=None,  # lazy loading below
            prerelease=prerelease,
            id=id,
        )
        self.cohort = cohort.upper()
        self._connector = connector
        self._files = files
        if decode_func is not None:
            self._decode_func = decode_func
        else:
            self._decode_func = self.decode_meta
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
            return tabular.Tabular(
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

    @property
    def data(self):
        m = self.get_matrix(subject=None)
        return m

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

        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        with QUIET:
            indexmap = {
                i: parc.get_region(regionname, allow_tuple=True)
                for i, regionname in enumerate(self.regions)
            }
        try:
            #assert len(indexmap) == nrows
            df = self._connector.get(self._files[subject], decode_func=self._decode_func)
            remapper = {
                label - min(indexmap.keys()): region
                for label, region in indexmap.items()
            }
            df.rename(index=remapper, inplace=True)
            df.rename(columns=remapper, inplace=True)
        except Exception:
            raise RuntimeError("Could not decode connectivity matrix regions.")
        return df



    def plot(self, subject: str = None, regions: str = None,
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
        try:
            if regions is None:
                return None
                #regions = self.regions
            #indices = [self.regions.index(r) for r in regions]
            m = self.get_matrix(subject=subject)
            #matrix = self.get_matrix(subject=subject).iloc[indices, indices].to_numpy()  # nilearn.plotting.plot_matrix works better with a numpy array
            metadata = m.meta['data']
            d = m.rename(str, axis='columns').rename(str, axis='index')

            d = d[d[regions] != 0][[regions]]
            d = d.map(np.log10)
            d.sort_values(by=[regions], inplace=True)
            scale = [
                (0, '#e6e6e6'),
                (0.005, '#f4f78c'),
                (0.13, '#e28b21'),
                (0.28, '#d75223'),
                (0.32, '#d44324'),
                (0.74, '#7c5359'),
                (0.92, '#513331'),
                (1, '#513331')
            ]
        except Exception as e:
            logger.exception('error')
        try:
            d = d.rename(name_to_code, axis='columns').rename(name_to_code, axis='index')
            regions = name_to_code(regions)
        except Exception as e:
            logger.error("error rename column", exc_info=True)
        if backend == "plotly":
            try:
                import plotly.express as px
                #fig = px.bar(d, x=regions, orientation='h', color=regions, color_continuous_scale=scale, range_color=[-6, 0])
                fig = px.scatter(
                    d, x=regions,
                    color=regions, color_continuous_scale=scale, range_color=[-6, 0],
                )
                #fig.update_yaxes(autorange='reversed')
                fig.update_layout(
                    xaxis={'range': [-6, 0], 'title': f'log<sub>10</sub>(FLNe) for injections in area {regions}'},
                    yaxis={
                        'dtick': 1,
                        'tick0': 0,
                        'range': [-1, len(d[regions])],
                        'autorange': False,
                        'title': 'source cortical area'
                    },
                    height=len(d) * 18,
                    coloraxis_showscale=False,
                    font={
                        'family': 'Sans Serif',
                        'size': 10,
                    }
                )
                fig.update_traces(
                    hovertemplate=f'log10(FLNe %{{y}} → {regions}) = %{{x:.2f}}'
                )
                inj = metadata['injections'][regions]
                _i = [f'  • <a href="{i["link"]}" target="_blank">{i["injection"]}</a> (⤓<a href="{i["csv"]}" target="_blank">csv</a>, ⤓<a href="{i.get("cells")}" target="_blank">xyz</a>)  ' for i in inj]
                fig.add_annotation(text=f'Injections into {regions}: <br>{"<br>".join(_i)}', xref='paper', yref='paper', x=0.9, y=0.01, showarrow=False, bordercolor="rgba(0, 0, 0, 1)", bgcolor="rgba(255, 255, 255, 1)", align='left')
            except Exception:
                logger.exception('error')
        return fig
