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


def build_atlas(spec):
    from .core.atlas import Atlas
    assert spec.get("@type") == "juelich/iav/atlas/v1.0.0"
    from .registry import REGISTRY
    atlas = Atlas(
        spec["@id"],
        spec["name"],
        species=spec["species"]
    )
    for space_id in spec["spaces"]:
        if not REGISTRY.Space.provides(space_id):
            raise ValueError(
                f"Invalid atlas configuration for {str(atlas)} - space {space_id} not known"
            )
        atlas._register_space(REGISTRY.Space[space_id])
    for parcellation_id in spec["parcellations"]:
        if not REGISTRY.Parcellation.provides(parcellation_id):
            raise ValueError(
                f"Invalid atlas configuration for {str(atlas)} - parcellation {parcellation_id} not known"
            )
        atlas._register_parcellation(REGISTRY.Parcellation[parcellation_id])
    return atlas


def build_space(spec):
    assert spec.get("@type") == "siibra/space/v0.0.1"
    from .core.space import Space
    volumes = list(map(build_volume, spec.get("volumes", [])))
    return Space(
        identifier=spec["@id"],
        name=spec["name"],
        volumes=volumes,
        shortname=spec.get("shortName", ""),
        description=spec.get("description"),
        modality=spec.get("modality"),
        publications=spec.get("publications", []),
        ebrains_ids=spec.get("ebrains", {})
    )


def build_region(cls, spec):
    from .core.region import Region
    return Region(
        name=spec["name"],
        children=map(build_region, spec.get("children", [])),
        shortname=spec.get("shortname", ""),
        description=spec.get("description", ""),
        publications=spec.get("publications", []),
        ebrains_ids=spec.get("ebrains_ids", {})
    )


def build_parcellation(spec):
    assert spec.get("@type", None) == "siibra/parcellation/v0.0.1"
    from .core.parcellation import Parcellation, ParcellationVersion
    # construct child region objects
    regions = []
    for regionspec in spec.get("regions", []):
        try:
            regions.append(build_region(regionspec))
        except Exception as e:
            print(regionspec)
            raise e

    # create the parcellation. This will create a parent region node for the regiontree.
    parcellation = Parcellation(
        identifier=spec["@id"],
        name=spec["name"],
        regions=regions,
        shortname=spec.get("shortName", ""),
        description=spec.get("description", ""),
        modality=spec.get('modality', ""),
        publications=spec.get("publications", []),
        ebrains_ids=spec.get("ebrains", {}),
    )

    # add version object, if any is specified
    versionspec = spec.get('@version', None)
    if versionspec is not None:
        version = ParcellationVersion(
            name=versionspec.get("name", None),
            parcellation=parcellation,
            collection=versionspec.get("collectionName", None),
            prev_filename=versionspec.get("@prev", None),
            next_filename=versionspec.get("@next", None),
            deprecated=versionspec.get("deprecated", False)
        )
        parcellation.version = version

    return parcellation


def build_volume(spec):
    assert spec.get("@type", None) == "siibra/volume/v0.0.1"
    from .volumes.volume import Volume
    return Volume(
        name=spec.get("name", ""),
        space_info=spec.get("space", {}),
        urls=spec.get("urls", {})
    )


def build_ebrains_dataset(spec):
    assert spec.get("@type", None) == "siibra/snapshots/ebrainsquery/v1"
    from .core.datasets import EbrainsDataset
    return EbrainsDataset(
        id=spec["id"],
        name=spec["name"],
        embargo_status=spec["embargoStatus"],
        cached_data=spec,
    )


class Factory:

    BUILDERS = {
        "juelich/iav/atlas/v1.0.0": build_atlas,
        "siibra/space/v0.0.1": build_space,
        "siibra/parcellation/v0.0.1": build_parcellation,
        "siibra/volume/v0.0.1": build_volume,
        "siibra/space/v0.0.1": build_space,
        "siibra/snapshots/ebrainsquery/v1": build_ebrains_dataset,
    }

    @classmethod
    def from_json(cls, spec: dict):
        spectype = spec.get("@type", None)
        if spectype in cls.BUILDERS:
            return cls.BUILDERS[spectype](spec)
        else:
            raise RuntimeError(f"No factory method for specification type {spectype}.")
