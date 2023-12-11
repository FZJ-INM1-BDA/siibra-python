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

from .. import anchor as _anchor
from . import cortical_profile

from ... import vocabularies
from ...commons import create_key
from ...retrieval import requests


class ReceptorDensityProfile(
    cortical_profile.CorticalProfile,
    configuration_folder="features/tabular/corticalprofiles/receptor",
    category='molecular'
):

    DESCRIPTION = (
        "Cortical profile of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The profiles provide, for a "
        "single tissue sample, an exemplary density distribution for a single receptor from the pial surface "
        "to the border between layer VI and the white matter."
    )

    _filter_attrs = cortical_profile.CorticalProfile._filter_attrs + ["receptor"]

    def __init__(
        self,
        receptor: str,
        tsvfile: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        """Generate a receptor density profile from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Receptor density",
            anchor=anchor,
            datasets=datasets,
        )
        self.receptor = receptor
        self._data_cached = None
        self._loader = requests.HttpRequest(tsvfile)
        self._unit_cached = None

    @property
    def key(self):
        return "{}_{}_{}_{}_{}".format(
            create_key(self.__class__.__name__),
            self.id,
            create_key(self.species_name),
            create_key(self.regionspec),
            create_key(self.receptor)
        )

    @property
    def receptor_fullname(self):
        return vocabularies.RECEPTOR_SYMBOLS[self.receptor]['receptor']['name']

    @property
    def name(self):
        return super().name + f" for {self.receptor}"

    @property
    def neurotransmitter(self):
        return "{} ({})".format(
            vocabularies.RECEPTOR_SYMBOLS[self.receptor]['neurotransmitter']['label'],
            vocabularies.RECEPTOR_SYMBOLS[self.receptor]['neurotransmitter']['name'],
        )

    @property
    def unit(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data.iloc[:, -1][0]

    @property
    def _values(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data.iloc[:, -2].values

    @property
    def _depths(self):
        return self._loader.data.iloc[:, 0].values / 100.

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
