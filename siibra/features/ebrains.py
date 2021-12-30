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

from .. import logger
from ..core.parcellation import Parcellation
from ..core.datasets import EbrainsDataset
from ..retrieval.requests import EbrainsKgQuery

from collections import defaultdict
import re
# we use this for sorting version strings of EBRAINS datasets
from distutils.version import LooseVersion 

class EbrainsRegionalDataset(RegionalFeature, EbrainsDataset):
    def __init__(self, regionspec, kg_id, name, embargo_status, species = []):
        RegionalFeature.__init__(self, regionspec, species)
        EbrainsDataset.__init__(self, kg_id, name, embargo_status)
        self.version = None
        self._next = None
        self._prev = None

    @property
    def version_history(self):
        if self._prev is None:
            return [self.version]
        else:
            return [self.version] + self._prev.version_history

    @property
    def url(self):
        return (
            f"https://search.kg.ebrains.eu/instances/{self.id.split('/')[-1]}"
        )

    def __str__(self):
        return EbrainsDataset.__str__(self)

    def __hash__(self):
        return EbrainsDataset.__hash__(self)

    def __eq__(self, o: object) -> bool:
        return EbrainsDataset.__eq__(self, o)



class EbrainsRegionalFeatureQuery(FeatureQuery):

    _FEATURETYPE = EbrainsRegionalDataset

    # in EBRAINS knowledge graph prior to v3, versions were modelled 
    # in dataset names. Typically found formats are (v1.0) and [rat, v2.1]
    VERSION_PATTERN = re.compile(r'^(.*?) *[\[\(][^v]*?(v[0-9].*?)[\]\)]')
    COMPACT_FEATURE_LIST = True

    # ids of EBRAINS datasets which represent siibra parcellations
    _PARCELLATION_IDS = [
        dset.id
        for parc in Parcellation.REGISTRY
        for dset in parc.datasets 
        if isinstance(dset, EbrainsDataset)
    ]
    
    # datasets whose name contains any of these strings will be ignored
    _BLACKLIST = {
        "Whole-brain parcellation of the Julich-Brain Cytoarchitectonic Atlas",
        "whole-brain collections of cytoarchitectonic probabilistic maps",
        "DiFuMo atlas",
        "Automated Anatomical Labeling (AAL1) atlas",
    }


    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)

        loader = EbrainsKgQuery(
            query_id="siibra-kg-feature-summary-0_0_4",
            schema="parcellationregion",
            params={"vocab": "https://schema.hbp.eu/myQuery/"},
        )

        versioned_datasets = defaultdict(dict)

        for r in loader.data.get("results", []):

            species_alt = []
            # List, keys @id, name
            for dataset in r.get("datasets", []):
                species_alt = [
                    *species_alt,
                    *dataset.get('ds_specimengroup_subject_species', []),
                    *dataset.get('s_subject_species', []),
                ]
            for dataset in r.get("datasets", []):

                ds_name = dataset.get("name")
                ds_id = dataset.get("@id")

                if (
                    self.COMPACT_FEATURE_LIST and 
                    any(ds_id.endswith(i) for i in self._PARCELLATION_IDS)
                ):
                    continue

                if (
                    self.COMPACT_FEATURE_LIST and
                    any(e.lower() in ds_name.lower() for e in self._BLACKLIST)
                ):
                    continue

                ds_embargo_status = dataset.get("embargo_status")
                if "dataset" not in ds_id:
                    logger.debug(
                        f"'{ds_name}' is not an interpretable dataset and will be skipped.\n(id:{ds_id})"
                    )
                    continue
                regionname: str = r.get("name", None)
                alias: str = r.get("alias", None)

                # species defined for the current dataset
                dataset_species = [
                    *dataset.get('ds_specimengroup_subject_species', []),
                    *dataset.get('s_subject_species', []),
                ]

                # if the current dataset has species defined, use the current species, 
                # else use the general species
                species = [*r.get("species", []), *(dataset_species if dataset_species else species_alt)] # list with keys @id, identifier, name

                # filter species by @id attribute
                unique_species = []
                for sp in species:
                    if sp.get('@id') in [s.get('@id') for s in unique_species]:
                        continue
                    unique_species.append(sp)

                dset = EbrainsRegionalDataset(
                    alias or regionname, ds_id, ds_name, ds_embargo_status, unique_species
                )

                version_match = self.VERSION_PATTERN.search(ds_name)
                if version_match is None or (not self.COMPACT_FEATURE_LIST):
                    self.register(dset)
                else:
                    # store version, add only the latest version after the loop
                    name, version = version_match.groups()
                    versioned_datasets[name][version] = dset
        
        # if versioned datasets have been recorded, register only 
        # the newest one with older ones linked as a version history.
        for name, datasets in versioned_datasets.items():
            try:
                # if possible, sort by version tag
                sorted_versions = sorted(datasets.keys(), key=LooseVersion)
            except TypeError:
                # else sort lexicographically
                sorted_versions = sorted(datasets.keys())

            # chain the dataset versions
            prev = None
            for version in sorted_versions:
                curr = datasets[version]
                curr.version = version
                if prev is not None:
                    curr._prev = prev
                    prev._next= curr
                prev = curr

            # register the last recent one
            self.register(curr)
            logger.debug(
                f"Registered only version {version} of {', '.join(sorted_versions)} for {name}. "
                f"Its version history is: {curr.version_history}"
            )



        # NOTE:
        # Potentially, using ebrains_id is a lot quicker, but if user selects region with higher level of hierarchy,
        # this benefit may be offset by numerous http calls even if they were cached.
        #       ebrains_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)
        #       for region, ds_id, ds_name, ds_embargo_status in get_dataset(atlas.selected_parcellation):
        #           feature = EbrainsRegionalDataset(region=region, id=ds_id, name=ds_name,
        #               embargo_status=ds_embargo_status)
        #       self.register(feature)
