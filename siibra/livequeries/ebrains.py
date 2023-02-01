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

from ..features.external import ebrains as _ebrains
from . import query

from ..commons import logger
from ..features import anchor as _anchor
from ..retrieval import requests, datasets
from ..core import parcellation, region

from collections import defaultdict
import re
from distutils.version import LooseVersion
from tqdm import tqdm
from tempfile import NamedTemporaryFile


class EbrainsFeatureQuery(query.LiveQuery, args=[], FeatureType=_ebrains.EbrainsDataFeature):

    # in EBRAINS knowledge graph prior to v3, versions were modelled
    # in dataset names. Typically found formats are (v1.0) and [rat, v2.1]
    VERSION_PATTERN = re.compile(r"^(.*?) *[\[\(][^v]*?(v[0-9].*?)[\]\)]")
    COMPACT_FEATURE_LIST = True

    # datasets whose name contains any of these strings will be ignored
    _BLACKLIST = {
        "Whole-brain parcellation of the Julich-Brain Cytoarchitectonic Atlas",
        "whole-brain collections of cytoarchitectonic probabilistic maps",
        "DiFuMo atlas",
        "Automated Anatomical Labeling (AAL1) atlas",
    }

    loader = requests.MultiSourcedRequest(
        requests=[
            requests.GitlabProxy(
                flavour=requests.GitlabProxyEnum.PARCELLATIONREGION_V1,
            ),
            requests.EbrainsKgQuery(
                query_id="siibra-kg-feature-summary-0_0_4",
                schema="parcellationregion",
                params={"vocab": "https://schema.hbp.eu/myQuery/"},
            )
        ]
    )

    parcellation_ids = None

    def __init__(self, **kwargs):
        query.LiveQuery.__init__(self, **kwargs)

        if self.__class__.parcellation_ids is None:
            self.__class__.parcellation_ids = [
                dset.id
                for parc in parcellation.Parcellation.registry()
                for dset in parc.datasets
                if isinstance(dset, datasets.EbrainsDataset)
            ]

    def query(self, region: region.Region):

        versioned_datasets = defaultdict(dict)
        invalid_species_datasets = {}
        results = self.loader.data.get("results", [])

        for r in tqdm(results, total=len(results)):

            regionname = r.get("name", None)
            alias = r.get("alias", None)
            for ds_spec in r.get("datasets", []):

                ds_name = ds_spec.get("name")
                ds_id = ds_spec.get("@id")
                if "dataset" not in ds_id:
                    continue
                ds_embargo_status = ds_spec.get("embargo_status")

                try:
                    ds_species = _anchor.Species.decode(ds_spec)
                except ValueError:
                    logger.debug(f"Cannot decode {ds_spec}")
                    invalid_species_datasets[ds_id] = ds_name
                    continue

                if self.COMPACT_FEATURE_LIST:
                    if any(ds_id.endswith(i) for i in self.parcellation_ids):
                        continue
                    if any(e.lower() in ds_name.lower() for e in self._BLACKLIST):
                        continue

                dset = _ebrains.EbrainsDataFeature(
                    dataset_id=ds_id,
                    name=ds_name,
                    anchor=_anchor.AnatomicalAnchor(
                        region=alias or regionname,
                        species=ds_species,
                    ),
                    embargo_status=ds_embargo_status
                )
                if not dset.matches(region):
                    continue

                version_match = self.VERSION_PATTERN.search(ds_name)
                if version_match is None or not self.COMPACT_FEATURE_LIST:
                    yield dset
                else:  # store version, add only the latest version after the loop
                    name, version = version_match.groups()
                    versioned_datasets[name][version] = dset

        if len(invalid_species_datasets) > 0:
            with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                for dsid, dsname in invalid_species_datasets.items():
                    f.write(f"{dsid} {dsname}\n")
                logger.warning(
                    f"{len(invalid_species_datasets)} datasets have been ignored, "
                    "because siibra could not decode their species. "
                    f"See {f.name}"
                )

        # if versioned datasets have been recorded, register only
        # the newest one with older ones linked as a version history.
        for name, dsets in versioned_datasets.items():
            try:  # if possible, sort by version tag
                sorted_versions = sorted(dsets.keys(), key=LooseVersion)
            except TypeError:  # else sort lexicographically
                sorted_versions = sorted(dsets.keys())

            # chain the dataset versions
            prev = None
            for version in sorted_versions:
                curr = dsets[version]
                curr.version = version
                if prev is not None:
                    curr._prev = prev
                    prev._next = curr
                prev = curr

            logger.debug(
                f"Registered only version {version} of {', '.join(sorted_versions)} for {name}. "
                f"Its version history is: {curr.version_history}"
            )
            yield curr
