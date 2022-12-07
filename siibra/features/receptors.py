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
from ..vocabularies import RECEPTOR_SYMBOLS

from io import BytesIO
import re


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


class ReceptorDensityProfile(CorticalProfile, configuration_folder="features/profiles/receptor"):

    DESCRIPTION = (
        "Cortical profile of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The profiles provide, for a "
        "single tissue sample, an exemplary density distribution for a single receptor from the pial surface "
        "to the border between layer VI and the white matter."
    )

    def __init__(
        self,
        receptor: str,
        tsvfile: str,
        anchor: "AnatomicalAnchor",
        datasets: list = []
    ):
        """Generate a receptor density profile from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            measuretype=f"{receptor} receptor density",
            anchor=anchor,
            datasets=datasets,
        )
        self.type = receptor
        self._data_cached = None
        self._loader = HttpRequest(
            tsvfile,
            lambda url: self.parse_tsv_data(decode_receptor_tsv(url)),
        )
        self._unit_cached = None

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


class ReceptorDensityFingerprint(RegionalFingerprint, configuration_folder="features/fingerprints/receptor"):

    DESCRIPTION = (
        "Fingerprint of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The fingerprint provides average "
        "density measurments for different receptors measured in tissue samples from different subjects "
        "together with the corresponding standard deviations. "
    )

    def __init__(
        self,
        tsvfile: str,
        anchor: "AnatomicalAnchor",
        datasets: list = []
    ):
        """ Generate a receptor fingerprint from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        RegionalFingerprint.__init__(
            self,
            description=self.DESCRIPTION,
            measuretype="Neurotransmitter receptor density",
            anchor=anchor,
            datasets=datasets,
        )

        self._data_cached = None
        self._loader = HttpRequest(
            tsvfile,
            lambda url: self.parse_tsv_data(decode_receptor_tsv(url)),
        )

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
