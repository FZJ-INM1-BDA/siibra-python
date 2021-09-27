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
from ..core.datasets import Dataset
from ..core.region import Region
from ..core.concept import AtlasConcept
from ..core.atlas import Atlas, SPECIES_MAP
from ..core.parcellation import Parcellation
from ..retrieval import decoders, requests

from requests.utils import quote


DISCLAIMER = """
    siibra accesses the API of NeuroMorpho.org to find neuron morphology datasets.
    Any use of these needs to comply with the terms of use specified at
    http://neuromorpho.org/useterm.jsp.

    Ascoli GA, Donohue DE, Halavi M. (2007) NeuroMorpho.Org: a central resource
    for neuronal morphologies.J Neurosci., 27(35):9247-51).

"""


class NeuronMorphology(RegionalFeature):
    
    def __init__(self, regionspec, species):
        RegionalFeature.__init__(self, regionspec, species)


class NeuroMorphoDataset(NeuronMorphology, Dataset):

    def __init__(self, regionspec, species, info):
        Dataset.__init__(self, identifier=info["_links"]["self"]["href"])
        NeuronMorphology.__init__(self, regionspec, species)
        if info["png_url"] is not None:
            self._image_loader = requests.LazyHttpRequest(
                info["png_url"], func=decoders.PNG
            )
        self._doi_loader = requests.LazyHttpRequest(
            url=f"https://doi.org/{info['reference_doi'][0]}",
            headers={"Accept": "text/x-bibliography"},
        )

        # add all fields as attributes
        for key in info:
            if not hasattr(self, key):
                setattr(self, key, info[key])

    @property
    def publications(self):
        return [self._doi_loader.data]

    @property
    def image(self):
        if self._image_loader is None:
            return None
        else:
            return self._image_loader.data

    @property
    def url(self):
        return f"http://neuromorpho.org/neuron_info.jsp?neuron_name={self.neuron_name}"

    def __str__(self):
        return Dataset.__str__(self)

    def __hash__(self):
        return Dataset.__hash__(self)

    def __eq__(self, o: object) -> bool:
        return Dataset.__eq__(self, o)

    def open(self):
        " Open the dataset page in a web browser "
        import webbrowser
        webbrowser.open(self.url, autoraise=True)


class NeuroMorphoQuery(FeatureQuery):
    _FEATURETYPE = NeuronMorphology

    BASE_URL = "http://neuromorpho.org/api/neuron"
    MAX_PAGES = 100
    _regions_cached = None
    _queries = {}

    @classmethod
    def regionnames(cls):
        """
        Lazy load region names known in neuromorpho.org.
        """
        if cls._regions_cached is None:
            cls._regions_cached = []
            page = 0
            while True:
                r = requests.HttpRequest(
                    f"{cls.BASE_URL}/fields/brain_region?page={page}",
                    func=decoders.JSON,
                ).get()
                cls._regions_cached.extend(r["fields"])
                page = r["page"]["number"] + 1
                num_pages = r["page"]["totalPages"]
                if num_pages > cls.MAX_PAGES:
                    raise RuntimeError(
                        f"{cls.__name__} trying to read {num_pages} pages."
                    )
                if page >= num_pages:
                    break
        return cls._regions_cached

    def __init__(self, **kwargs):
        """
        We do not load anything on construction, since this query
        reimplements the specific execute() method to load only
        requested features.
        """
        FeatureQuery.__init__(self)
        print(DISCLAIMER)

    @classmethod
    def _query_region(cls, regionname, species: str):
        """
        Run a NeuroMorpho query with a valid NeuroMorpho brain_region field value,
        or return the already cached result.
        """

        # Require a valid species specification
        species_ = SPECIES_MAP.get(species.lower())
        if species_ is None:
            raise RuntimeError(f"Cannot resolve species specification '{species}'.")

        # Require the regionnames to be known to NeuroMorpho.og
        if regionname not in cls.regionnames():
            raise RuntimeError(
                f"{cls.__name__} region query called with "
                f"invalid region name {regionname}."
            )

        if (regionname, species_) not in cls._queries:
            page = 0
            featurespecs = []
            url = f"{cls.BASE_URL}/select?q=species:{species_}&fq=brain_region:{quote(regionname)}"

            while True:

                # load a page
                r = requests.HttpRequest(
                    url + f"&page={page}", func=decoders.JSON
                ).get()
                if "_embedded" not in r:
                    break

                for info in r["_embedded"]["neuronResources"]:
                    featurespecs.append(info)

                # see if another page needs to be leaded
                page = r["page"]["number"] + 1
                num_pages = r["page"]["totalPages"]
                if num_pages > cls.MAX_PAGES:
                    raise RuntimeError(
                        f"{cls.__name__} trying to read {num_pages} pages."
                    )
                if page >= num_pages:
                    break

            cls._queries[regionname, species] = featurespecs

        return cls._queries[regionname, species]

    def execute(self, concept: AtlasConcept):
        """
        Executes a NeuroMorpho query associated with an atlas object.
        """
        matches = []

        if isinstance(concept, (Atlas, Parcellation)):
            logger.warning(
                f"{self.__class__.__name__} does not suppot queries for "
                f"{concept.__class__.__name__} concepts, no morphologies returned."
            )

        elif isinstance(concept, Region):

            # Find valid NeuroMorpho brain_region values that would match the given region
            region = concept
            matched_names = [
                n for n in self.__class__.regionnames() if region.matches(n)
            ]

            # Determine uniqueness of matched names.
            # To how many regions of the parcellation would each of them match?
            # We start by matching to the most unique ones, and do not
            # match region names that match to more than 30% of the regions
            # in the parcellation.
            num_regions = len(region.parcellation.regiontree.leaves)
            uniqueness = {
                n: len(region.parcellation.find_regions(n)) / num_regions
                for n in matched_names
            }

            for regionname, fraction in sorted(uniqueness.items(), key=lambda i: i[1]):

                if fraction > 0.3:
                    logger.debug(
                        f'Name {regionname} matches {fraction*100:.2f}% of all '
                        f'regions in {region.parcellation.name}, not using it.')
                    continue
                
                for info in self.__class__._query_region(
                    regionname, region.parcellation.species
                ):
                    matches.append(
                        NeuroMorphoDataset(
                            regionspec=region.name,
                            species=region.parcellation.species,
                            info=info,
                        )
                    )

                if len(matches) > 0:
                    logger.info(f'Found {len(matches)} matches for {regionname}, not considering more matched names.')
                    break

        return matches
