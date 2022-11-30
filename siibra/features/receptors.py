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

from .feature import CorticalProfile, RegionalFingerprint
from ..commons import logger, create_key
from ..retrieval.requests import HttpRequest
from ..core.datasets import EbrainsDataset


from io import BytesIO
import re


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


def decode_receptor_tsv(bytearray):
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


class ReceptorDensityProfile(CorticalProfile, EbrainsDataset):

    DESCRIPTION = (
        "Cortical profile of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The profiles provide, for a "
        "single tissue sample, an exemplary density distribution for a single receptor from the pial surface "
        "to the border between layer VI and the white matter."
    )

    def __init__(
        self,
        dataset_id: str,
        species: dict,
        regionname: str,
        receptor_type: str,
        url: str,
    ):
        """Generate a receptor density profile from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        EbrainsDataset.__init__(self, dataset_id, f"Receptor density for {receptor_type} in {regionname}")
        self.type = receptor_type
        self._data_cached = None
        self._loader = HttpRequest(
            url,
            lambda url: self.parse_tsv_data(decode_receptor_tsv(url)),
        )
        self._unit_cached = None
        CorticalProfile.__init__(self, f"{receptor_type} receptor density", species, regionname, self.DESCRIPTION)

    @property
    def key(self):
        return "{}_{}_{}_{}_{}".format(
            create_key(self.__class__.__name__),
            self.id,
            create_key(self.species_name),
            create_key(self.regionspec),
            create_key(self.type)
        )

    @property
    def receptor(self):
        return "{} ({})".format(
            self.type,
            RECEPTOR_SYMBOLS[self.type]['receptor']['name'],
        )

    @property
    def neurotransmitter(self):
        return "{} ({})".format(
            RECEPTOR_SYMBOLS[self.type]['neurotransmitter']['label'],
            RECEPTOR_SYMBOLS[self.type]['neurotransmitter']['name'],
        )

    @property
    def unit(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data["unit"]

    @property
    def _values(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data["density"]

    @property
    def _depths(self):
        return self._loader.data["depth"]

    @classmethod
    def parse_tsv_data(self, data):
        units = {list(v.values())[3] for v in data.values()}
        assert len(units) == 1
        return {
            "depth": [float(k) / 100.0 for k in data.keys() if k.isnumeric()],
            "density": [
                float(list(v.values())[2]) for k, v in data.items() if k.isnumeric()
            ],
            "unit": next(iter(units)),
        }


class ReceptorDensityFingerprint(RegionalFingerprint, EbrainsDataset):
    DESCRIPTION = (
        "Fingerprint of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The fingerprint provides average "
        "density measurments for different receptors measured in tissue samples from different subjects "
        "together with the corresponding standard deviations. "
    )

    def __init__(
        self,
        dataset_id: str,
        species: dict,
        regionname: str,
        url: str,
    ):
        """ Generate a receptor fingerprint from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        self._data_cached = None
        self._loader = HttpRequest(
            url,
            lambda url: self.parse_tsv_data(decode_receptor_tsv(url)),
        )
        RegionalFingerprint.__init__(
            self,
            measuretype="Neurotransmitter receptor density",
            species=species,
            regionname=regionname,
            description=self.DESCRIPTION,
        )
        EbrainsDataset.__init__(self, dataset_id, self.name)

    @property
    def unit(self):
        return self._loader.data['unit']

    @property
    def receptors(self):
        return self._loader.data['labels']

    @property
    def neurotransmitters(self):
        return [
            "{} ({})".format(
                RECEPTOR_SYMBOLS[t]['neurotransmitter']['label'],
                RECEPTOR_SYMBOLS[t]['neurotransmitter']['name'],
            )
            for t in self.receptors
        ]

    @property
    def _labels(self):
        return self.receptors

    @property
    def _means(self):
        return self._loader.data['means']

    @property
    def _stds(self):
        return self._loader.data['stds']

    @property
    def key(self):
        return "{}_{}_{}_{}".format(
            create_key(self.__class__.__name__),
            self.id,
            create_key(self.species_name),
            create_key(self.regionspec),
        )

    @classmethod
    def parse_tsv_data(cls, data: dict):
        units = {list(v.values())[3] for v in data.values()}
        labels = list(data.keys())
        assert len(units) == 1
        try:
            mean = [data[_]["density (mean)"] for _ in labels]
            std = [data[_]["density (sd)"] for _ in labels]
        except KeyError as e:
            print(str(e))
            logger.error("Could not parse fingerprint from this dictionary")
        return {
            'unit': next(iter(units)),
            'labels': labels,
            'means': [float(m) if m.isnumeric() else 0 for m in mean],
            'stds': [float(s) if s.isnumeric() else 0 for s in std],
        }
