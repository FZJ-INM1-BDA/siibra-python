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

from .feature import RegionalFeature
from .query import FeatureQuery
from ..commons import logger
from ..retrieval.requests import EbrainsKgQuery, HttpRequest
from ..core.datasets import EbrainsDataset, DatasetJsonModel, ConfigBaseModel
from ..core.serializable_concept import NpArrayDataModel


from typing import Dict, Optional
import PIL.Image as Image
import numpy as np
from io import BytesIO
from collections import namedtuple
import re
import importlib
import hashlib
from pydantic import Field


RECEPTOR_SYMBOLS = {
    "5-HT1A": {
        "receptor": {
            "latex": "$5-HT_{1A}$",
            "markdown": "5-HT<sub>1A</sub>",
            "name": "5-hydroxytryptamine receptor type 1A",
        },
        "neurotransmitter": {
            "label": "5-HT",
            "latex": "$5-HT$",
            "markdown": "5-HT",
            "name": "serotonin",
        },
    },
    "5-HT2": {
        "receptor": {
            "latex": "$5-HT_2$",
            "markdown": "5-HT<sub>2</sub>",
            "name": "5-hydroxytryptamine receptor type 2",
        },
        "neurotransmitter": {
            "label": "5-HT",
            "latex": "$5-HT$",
            "markdown": "5-HT",
            "name": "serotonin",
        },
    },
    "AMPA": {
        "receptor": {
            "latex": "$AMPA$",
            "markdown": "AMPA",
            "name": "alpha-amino-3hydroxy-5-methyl-4isoxazolepropionic acid receptor",
        },
        "neurotransmitter": {
            "label": "Glu",
            "latex": "$Glu$",
            "markdown": "Glu",
            "name": "glutamate",
        },
    },
    "BZ": {
        "receptor": {
            "latex": "$GABA_A/BZ$",
            "markdown": "GABA<sub>A</sub>/BZ",
            "name": "gamma-aminobutyric acid receptor type A / benzodiazepine associated binding site",
        },
        "neurotransmitter": {
            "label": "GABA",
            "latex": "$GABA$",
            "markdown": "GABA",
            "name": "gamma-aminobutyric acid",
        },
    },
    "D1": {
        "receptor": {
            "latex": "$D_1$",
            "markdown": "D<sub>1</sub>",
            "name": "dopamine receptor type 1",
        },
        "neurotransmitter": {
            "label": "DA",
            "latex": "$DA$",
            "markdown": "DA",
            "name": "dopamine",
        },
    },
    "GABAA": {
        "receptor": {
            "latex": "$GABA_A$",
            "markdown": "GABA<sub>A</sub>",
            "name": "gamma-aminobutyric acid receptor type A",
        },
        "neurotransmitter": {
            "label": "GABA",
            "latex": "$GABA$",
            "markdown": "GABA",
            "name": "gamma-aminobutyric acid",
        },
    },
    "GABAB": {
        "receptor": {
            "latex": "$GABA_B$",
            "markdown": "GABA<sub>B</sub>",
            "name": "gamma-aminobutyric acid receptor type B",
        },
        "neurotransmitter": {
            "label": "GABA",
            "latex": "$GABA$",
            "markdown": "GABA",
            "name": "gamma-aminobutyric acid",
        },
    },
    "M1": {
        "receptor": {
            "latex": "$M_1$",
            "markdown": "M<sub>1</sub>",
            "name": "muscarinic acetylcholine receptor type 1",
        },
        "neurotransmitter": {
            "label": "ACh",
            "latex": "$ACh$",
            "markdown": "ACh",
            "name": "acetylcholine",
        },
    },
    "M2": {
        "receptor": {
            "latex": "$M_2$",
            "markdown": "M<sub>2</sub>",
            "name": "muscarinic acetylcholine receptor type 2",
        },
        "neurotransmitter": {
            "label": "ACh",
            "latex": "$ACh$",
            "markdown": "ACh",
            "name": "acetylcholine",
        },
    },
    "M3": {
        "receptor": {
            "latex": "$M_3$",
            "markdown": "M<sub>3</sub>",
            "name": "muscarinic acetylcholine receptor type 3",
        },
        "neurotransmitter": {
            "label": "ACh",
            "latex": "$ACh$",
            "markdown": "ACh",
            "name": "acetylcholine",
        },
    },
    "NMDA": {
        "receptor": {
            "latex": "$NMDA$",
            "markdown": "NMDA",
            "name": "N-methyl-D-aspartate receptor",
        },
        "neurotransmitter": {
            "label": "Glu",
            "latex": "$Glu$",
            "markdown": "Glu",
            "name": "glutamate",
        },
    },
    "alpha1": {
        "receptor": {
            "latex": "$\\alpha_1$",
            "markdown": "&#945<sub>1</sub>",
            "name": "alpha-1 adrenergic receptor",
        },
        "neurotransmitter": {
            "label": "NE",
            "latex": "$NE$",
            "markdown": "NE",
            "name": "norepinephrine",
        },
    },
    "alpha2": {
        "receptor": {
            "latex": "$\\alpha_2$",
            "markdown": "&#945<sub>2</sub>",
            "name": "alpha-2 adrenergic receptor",
        },
        "neurotransmitter": {
            "label": "NE",
            "latex": "$NE$",
            "markdown": "NE",
            "name": "norepinephrine",
        },
    },
    "alpha4beta2": {
        "receptor": {
            "latex": "$\\alpha_4\\beta_2$",
            "markdown": "&#945<sub>4</sub>&#946<sub>2</sub>",
            "name": "alpha-4 beta-2 nicotinic receptor",
        },
        "neurotransmitter": {
            "label": "ACh",
            "latex": "$ACh$",
            "markdown": "ACh",
            "name": "acetylcholine",
        },
    },
    "kainate": {
        "receptor": {
            "latex": "$kainate$",
            "markdown": "kainate",
            "name": "kainate receptors",
        },
        "neurotransmitter": {
            "label": "Glu",
            "latex": "$Glu$",
            "markdown": "Glu",
            "name": "glutamate",
        },
    },
    "mGluR2_3": {
        "receptor": {
            "latex": "$mGluR2/3$",
            "markdown": "mGluR2/3",
            "name": "metabotropic glutamate receptor type 2 and 3",
        },
        "neurotransmitter": {
            "label": "Glu",
            "latex": "$Glu$",
            "markdown": "Glu",
            "name": "glutamate",
        },
    },
}


def unify_stringlist(L: list):
    """Adds asterisks to strings that appear multiple times, so the resulting
    list has only unique strings but still the same length, order, and meaning.
    For example:
        unify_stringlist(['a','a','b','a','c']) -> ['a','a*','b','a**','c']
    """
    assert all([isinstance(_, str) for _ in L])
    return [L[i] + "*" * L[:i].count(L[i]) for i in range(len(L))]


def decode_tsv(bytearray):
    bytestream = BytesIO(bytearray)
    header = bytestream.readline()
    lines = [_.strip() for _ in bytestream.readlines() if len(_.strip()) > 0]
    sep = b"{" if b"{" in lines[0] else b"\t"
    keys = unify_stringlist(
        [
            n.decode("utf8").replace('"', "").replace("'", "").strip()
            for n in header.split(sep)
        ]
    )
    if any(k.endswith("*") for k in keys):
        logger.warning("Redundant headers in receptor file")
    assert len(keys) == len(set(keys))
    return {
        _.split(sep)[0].decode("utf8"): dict(
            zip(
                keys,
                [re.sub(r"\\+", r"\\", v.decode("utf8").strip()) for v in _.split(sep)],
            )
        )
        for _ in lines
    }


class DensityProfile:
    def __init__(self, data):
        units = {list(v.values())[3] for v in data.values()}
        assert len(units) == 1
        self.unit = next(iter(units))
        self.densities = {
            int(k): float(list(v.values())[2]) for k, v in data.items() if k.isnumeric()
        }

    def __iter__(self):
        return self.densities.values()


Density = namedtuple("Density", "name, mean, std, unit")


class FingerPrintDataModel(ConfigBaseModel):
    mean: float
    std: float
    unit: str


class ProfileDataModel(ConfigBaseModel):
    density: NpArrayDataModel
    unit: str


class AutoradiographyDataModel(NpArrayDataModel):
    pass


class ReceptorMarkupModel(ConfigBaseModel):
    latex: str
    markdown: str
    name: str


class NeurotransmitterMarkupModel(ReceptorMarkupModel):
    label: str


class SymbolMarkupClass(ConfigBaseModel):
    receptor: ReceptorMarkupModel
    neurotransmitter: NeurotransmitterMarkupModel


class ReceptorDataModel(ConfigBaseModel):
    autoradiographs: Dict[str, AutoradiographyDataModel]
    profiles: Dict[str, ProfileDataModel]
    fingerprints: Dict[str, FingerPrintDataModel]
    receptor_symbols: Dict[str, SymbolMarkupClass]


class DensityFingerprint:

    unit = None
    labels = []
    meanvals = []
    stdvals = []
    n = 0

    def __init__(self, datadict):
        """
        Create a DensityFingerprint from a data dictionary coming from a
        receptor fingerprint tsv file.
        """
        units = {list(v.values())[3] for v in datadict.values()}
        assert len(units) == 1
        self.unit = next(iter(units))
        self.labels = list(datadict.keys())
        try:
            mean = [datadict[_]["density (mean)"] for _ in self.labels]
            std = [datadict[_]["density (sd)"] for _ in self.labels]
        except KeyError as e:
            print(str(e))
            logger.error("Could not parse fingerprint from this dictionary")
        self.meanvals = [float(m) if m.isnumeric() else 0 for m in mean]
        self.stdvals = [float(s) if s.isnumeric() else 0 for s in std]

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self.labels):
                return None
            index_ = index
        elif isinstance(index, str):
            if index not in self.labels:
                return None
            index_ = self.labels.index(index)
        return Density(
            name=self.labels[index_],
            mean=self.meanvals[index_],
            std=self.stdvals[index_],
            unit=self.unit,
        )

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.labels):
            self.n += 1
            return self[self.n - 1]
        else:
            raise StopIteration

    def __str__(self):
        return "\n".join(
            "{d.name:15.15s} {d.mean:8.1f} {d.unit} (+/-{d.std:5.1f})".format(d=d)
            for d in iter(self)
        )


class ReceptorDistribution(RegionalFeature, EbrainsDataset):
    """
    Reprecent a receptor distribution dataset with fingerprint, profiles and
    autoradiograph samples. This implements a lazy loading scheme.
    TODO lazy loading could be more elegant.
    """

    def __init__(self, region, kg_result, **kwargs):

        RegionalFeature.__init__(self, region, **kwargs)
        EbrainsDataset.__init__(self, kg_result["@id"], kg_result["name"])

        self.info = kg_result["description"]
        self.url = "https://search.kg.ebrains.eu/instances/Dataset/{}".format(self.id)
        self.modality = kg_result["modality"]

        urls = kg_result["files"]
        
        self.files=urls

        def urls_matching(regex):
            return filter(lambda u: re.match(regex, u), urls)

        # add fingerprint if a url is found
        self._fingerprint_loader = None
        for url in urls_matching(".*_fp[._]"):
            if self._fingerprint_loader is not None:
                logger.warning(f"More than one fingerprint found for {self}")
            self._fingerprint_loader = HttpRequest(
                url, lambda u: DensityFingerprint(decode_tsv(u))
            )

        # add any cortical profiles
        self._profile_loaders = {}
        for url in urls_matching(r".*_pr[._].*\.tsv"):
            rtype, basename = url.split("/")[-2:]
            if rtype not in basename:
                continue
            if rtype in self._profile_loaders:
                logger.warning(f"More than one profile for '{rtype}' in {self.url}")
            self._profile_loaders[rtype] = HttpRequest(
                url, lambda u: DensityProfile(decode_tsv(u))
            )

        # add autoradiograph
        self._autoradiograph_loaders = {}

        def img_from_bytes(b):
            # PIL is column major but numpy is row major
            # see https://stackoverflow.com/questions/19016144/conversion-between-pillow-image-object-and-numpy-array-changes-dimension
            return np.transpose(np.array(Image.open(BytesIO(b))), axes=[1,0,2])

        for url in urls_matching(".*_ar[._]"):
            rtype, basename = url.split("/")[-2:]
            if rtype not in basename:
                continue
            if rtype in self._autoradiograph_loaders:
                logger.warning(
                    f"More than one autoradiograph for '{rtype}' in {self.url}"
                )
            self._autoradiograph_loaders[rtype] = HttpRequest(url, img_from_bytes)

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/receptor"

    @property
    def model_id(self):
        return f'{ReceptorDistribution.get_model_type()}/{hashlib.md5(super().model_id.encode("utf-8")).hexdigest()}'

    def to_model(self, detail=False, **kwargs) -> 'ReceptorDatasetModel':
        base_dict = dict(super().to_model(detail=detail, **kwargs).dict())
        base_dict["@type"] = ReceptorDistribution.get_model_type()
        if not detail:
            return ReceptorDatasetModel(
                **base_dict,
                data=None,
            )

        data_model=ReceptorDataModel(
            autoradiographs={
                key: AutoradiographyDataModel(autoradiograph)
                for key, autoradiograph in self.autoradiographs.items()},
            profiles={
                key: ProfileDataModel(
                    density=NpArrayDataModel(np.array([val for val in profile.densities.values()], dtype="float32")),
                    unit=profile.unit
                ) for key, profile in self.profiles.items()
            },
            fingerprints={
                key: FingerPrintDataModel(
                    mean=mean,
                    std=std,
                    unit=self.fingerprint.unit
                ) for key, mean, std in zip(self.fingerprint.labels, self.fingerprint.meanvals, self.fingerprint.stdvals)
            },
            receptor_symbols=RECEPTOR_SYMBOLS
        )
        return ReceptorDatasetModel(
            **base_dict,
            data=data_model
        )

    @property
    def fingerprint(self):
        """The receptor fingerprint, with mean and standard
        deviations of receptor densities for different
        receptor types, measured across multiple samples."""
        if self._fingerprint_loader is None:
            return None
        else:
            return self._fingerprint_loader.data

    @property
    def profiles(self):
        """Dictionary of cortical receptor distribution
        profiles available for this feature, keyed by
        receptor type."""
        return {rtype: l.data for rtype, l in self._profile_loaders.items()}

    @property
    def autoradiographs(self):
        """Dictionary of sample autoradiographs available for this feature,
        keyed by receptor type."""
        return {rtype: l.data for rtype, l in self._autoradiograph_loaders.items()}

    def plot(self, title=None):
        """
        Produce a fingerprint and cortical profile plot of this receptor distribution feature.
        """

        if importlib.util.find_spec("matplotlib") is None:
            logger.warning("matplotlib not available. Plotting disabled.")
            return None

        from matplotlib import pyplot
        from collections import deque

        # plot profiles and fingerprint
        fig = pyplot.figure(figsize=(12, 4))
        pyplot.subplot(121)
        for _, profile in self.profiles.items():
            pyplot.plot(
                list(profile.densities.keys()),
                np.fromiter(profile.densities.values(), dtype="d"),
            )
        pyplot.xlabel("Cortical depth (%)")
        pyplot.ylabel("Receptor density")
        pyplot.grid(True)
        pyplot.suptitle(str(self) if title is None else title)
        pyplot.legend(
            labels=[_ for _ in self.profiles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=4,
            prop={"size": 8},
        )

        ax = pyplot.subplot(122, projection="polar")
        angles = deque(
            np.linspace(0, 2 * np.pi, len(self.fingerprint.labels) + 1)[:-1][::-1]
        )
        angles.rotate(5)
        angles = list(angles)
        means = [d.mean for d in self.fingerprint]
        stds = [d.mean + d.std for d in self.fingerprint]
        pyplot.plot(angles + [angles[0]], means + [means[0]], "k-", lw=3)
        pyplot.plot(angles + [angles[0]], stds + [stds[0]], "k", lw=1)
        ax.set_xticks(angles)
        ax.set_xticklabels([_ for _ in self.fingerprint.labels])
        ax.tick_params(pad=9, labelsize=10)
        ax.tick_params(axis="y", labelsize=8)
        return fig


class ReceptorDatasetModel(DatasetJsonModel):
    data: Optional[ReceptorDataModel]
    type: str = Field(ReceptorDistribution.get_model_type(), const=True, alias="@type")


class ReceptorQuery(FeatureQuery):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self,**kwargs):
        FeatureQuery.__init__(self)
        kg_req = EbrainsKgQuery(
            query_id="siibra_receptor_densities-0_0_2",
            params={'vocab': 'https://schema.hbp.eu/myQuery/' }
        )
        kg_query = kg_req.get()
        
        not_used = 0
        for kg_result in kg_query["results"]:
            region_names = [p_region["name"] for p_region in kg_result["parcellationRegion"]]
            species = kg_result.get('species', [])
            for region_name in region_names:
                f = ReceptorDistribution(region_name, kg_result, species=species)
                if f.fingerprint is None:
                    not_used += 1
                else:
                    self.register(f)
        
        if not_used > 0:
            logger.info(f'{not_used} receptor datasets skipped due to unsupported format.')
