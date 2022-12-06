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

from .commons import logger
from .anchor import AnatomicalAnchor

from .core.atlas import Atlas
from .core.parcellation import Parcellation, ParcellationVersion
from .core.space import Space
from .core.region import Region
from .core.location import Point, PointSet

from .retrieval.datasets import EbrainsDataset

from .volumes.volume import Volume, NiftiFetcher, NeuroglancerVolumeFetcher, ZipContainedNiftiFetcher
from .volumes.mesh import NeuroglancerMesh, GiftiSurface
from .volumes.map import Map
from .volumes.sparsemap import SparseMap

from .features.receptors import ReceptorDensityFingerprint, ReceptorDensityProfile
from .features.cells import CellDensityFingerprint, CellDensityProfile

from os import path
import json
import numpy as np

BUILDFUNCS = {
    "juelich/iav/atlas/v1.0.0": "build_atlas",
    "siibra/space/v0.0.1": "build_space",
    "siibra/parcellation/v0.0.1": "build_parcellation",
    "siibra/volume/v0.0.1": "build_volume",
    "siibra/map/v0.0.1": "build_map",
    "siibra/space/v0.0.1": "build_space",
    "siibra/snapshots/ebrainsquery/v1": "build_ebrains_dataset",
    "https://openminds.ebrains.eu/sands/CoordinatePoint": "build_point",
    "tmp/poly": "build_pointset",
    "siibra/feature/fingerprint/receptor/v0.1": "build_receptor_density_fingerprint",
    "siibra/feature/profile/receptor/v0.1": "build_receptor_density_profile",
    "siibra/feature/fingerprint/celldensity/v1.0.0": "build_cell_density_fingerprint",
    "siibra/feature/profile/celldensity/v1.0.0": "build_cell_density_profile",
}


class Factory:

    @classmethod
    def extract_datasets(cls, spec):
        datasets = []
        if "minds/core/dataset/v1.0.0" in spec.get("ebrains", {}):
            datasets.append(
                EbrainsDataset(id=spec["ebrains"]["minds/core/dataset/v1.0.0"])
            )
        return datasets

    @classmethod
    def extract_volumes(cls, spec):
        return list(map(cls.build_volume, spec.get("volumes", [])))

    @classmethod
    def extract_anchor(cls, spec):
        return AnatomicalAnchor(
            region=spec.get('region', None),
            location=None,
            species=spec.get("species", None),
        )

    @classmethod
    def build_atlas(cls, spec):
        atlas = Atlas(
            spec["@id"],
            spec["name"],
            species=spec["species"]
        )
        for space_id in spec["spaces"]:
            atlas._register_space(space_id)
        for parcellation_id in spec["parcellations"]:
            atlas._register_parcellation(parcellation_id)
        return atlas

    @classmethod
    def build_space(cls, spec):
        return Space(
            identifier=spec["@id"],
            name=spec["name"],
            volumes=cls.extract_volumes(spec),
            shortname=spec.get("shortName", ""),
            description=spec.get("description"),
            modality=spec.get("modality"),
            publications=spec.get("publications", []),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_region(cls, spec):
        return Region(
            name=spec["name"],
            children=map(cls.build_region, spec.get("children", [])),
            shortname=spec.get("shortname", ""),
            description=spec.get("description", ""),
            publications=spec.get("publications", []),
            datasets=cls.extract_datasets(spec),
            rgb=spec.get("rgb", None),
        )

    @classmethod
    def build_parcellation(cls, spec):
        regions = []
        for regionspec in spec.get("regions", []):
            try:
                regions.append(cls.build_region(regionspec))
            except Exception as e:
                print(regionspec)
                raise e
        parcellation = Parcellation(
            identifier=spec["@id"],
            name=spec["name"],
            regions=regions,
            shortname=spec.get("shortName", ""),
            description=spec.get("description", ""),
            modality=spec.get('modality', ""),
            publications=spec.get("publications", []),
            datasets=cls.extract_datasets(spec),
        )

        # add version object, if any is specified
        versionspec = spec.get('@version', None)
        if versionspec is not None:
            version = ParcellationVersion(
                name=versionspec.get("name", None),
                parcellation=parcellation,
                collection=versionspec.get("collectionName", None),
                prev_id=versionspec.get("@prev", None),
                next_id=versionspec.get("@next", None),
                deprecated=versionspec.get("deprecated", False)
            )
            parcellation.version = version

        return parcellation

    @classmethod
    def build_volume(cls, spec):
        providers = []
        provider_types = [
            NeuroglancerVolumeFetcher,
            NiftiFetcher,
            ZipContainedNiftiFetcher,
            NeuroglancerMesh,
            GiftiSurface
        ]
        for srctype, url in spec.get("urls", {}).items():
            for ProviderType in provider_types:
                if srctype == ProviderType.srctype:
                    providers.append(ProviderType(url))
                    break
            else:
                logger.warn(f"Volume source type {srctype} not yet supported, ignoring this specification.")

        return Volume(
            name=spec.get("name", ""),
            space_spec=spec.get("space", {}),
            providers=providers
        )

    @classmethod
    def build_map(cls, spec):
        # maps have no configured identifier - we require the spec filename to build one
        assert "filename" in spec
        basename = path.splitext(path.basename(spec['filename']))[0]
        name = basename.replace('-', ' ').replace('_', ' ')
        identifier = f"{spec['@type'].replace('/','-')}_{basename}"
        volumes = cls.extract_volumes(spec)
        Maptype = Map if len(volumes) < 10 else SparseMap

        return Maptype(
            identifier=spec.get("@id", identifier),
            name=spec.get("name", name),
            space_spec=spec.get("space", {}),
            parcellation_spec=spec.get("parcellation", {}),
            indices=spec.get("indices", {}),
            volumes=volumes,
            shortname=spec.get("shortName", ""),
            description=spec.get("description"),
            modality=spec.get("modality"),
            publications=spec.get("publications", []),
            ebrains=cls.build_ebrains_items(spec.get("ebrains", {}))
        )

    @classmethod
    def build_ebrains_dataset(cls, spec):
        return EbrainsDataset(
            id=spec["id"],
            name=spec["name"],
            embargo_status=spec["embargoStatus"],
            cached_data=spec,
        )

    @classmethod
    def build_point(cls, spec):
        space_id = spec["coordinateSpace"]["@id"]
        assert all(c["unit"]["@id"] == "id.link/mm" for c in spec["coordinates"])
        return Point(
            list(np.float16(c["value"]) for c in spec["coordinates"]),
            space_id=space_id,
        )

    @classmethod
    def build_pointset(cls, spec):
        space_id = spec["coordinateSpace"]["@id"]
        coords = []
        for coord in spec["coordinates"]:
            assert all(c["unit"]["@id"] == "id.link/mm" for c in coord)
            coords.append(list(np.float16(c["value"]) for c in coord))
        return PointSet(coords, space_id=space_id)

    @classmethod
    def build_receptor_density_fingerprint(cls, spec):
        return ReceptorDensityFingerprint(
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_receptor_density_profile(cls, spec):
        return ReceptorDensityProfile(
            receptor=spec['receptor'],
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_cell_density_fingerprint(cls, spec):
        return CellDensityFingerprint(
            species=spec['species'],
            regionname=spec['region_name'],
            segmentfiles=spec['segmentfiles'],
            layerfiles=spec['layerfiles'],
            dataset_id=spec['kgId']
        )

    @classmethod
    def build_cell_density_profile(cls, spec):
        return CellDensityProfile(
            species=spec['species'],
            regionname=spec['region_name'],
            url=spec['url'],
            dataset_id=spec['kgId'],
            section=spec['section'],
            patch=spec['patch']
        )

    @classmethod
    def from_json(cls, spec: dict):

        if isinstance(spec, str):
            if path.isfile(spec):
                fname = spec
                with open(spec, "r") as f:
                    spec = json.load(f)
                    assert "filename" not in spec
                    spec['filename'] = fname
            else:
                spec = json.loads(spec)

        spectype = spec.get("@type", None)
        if spectype in BUILDFUNCS:
            func = getattr(cls, BUILDFUNCS[spectype])
            return func(spec)
        else:
            raise RuntimeError(f"No factory method for specification type {spectype}.")
