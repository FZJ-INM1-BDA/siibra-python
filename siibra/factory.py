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
from .core.atlas import Atlas
from .core.parcellation import Parcellation, ParcellationVersion
from .core.space import Space
from .core.region import Region
from .core.datasets import EbrainsDataset
from .core.location import Point, PointSet
from .volumes.volume import Volume, NiftiVolume, NeuroglancerVolume, ZipContainedNiftiVolume
from .volumes.mesh import NeuroglancerMesh, GiftiSurface
from .volumes.map import Map
from .volumes.sparsemap import SparseMap

from os import path
import json
import numpy as np


class Factory:

    @classmethod
    def build_atlas(cls, spec):
        assert spec.get("@type") == "juelich/iav/atlas/v1.0.0"
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
        assert spec.get("@type") == "siibra/space/v0.0.1"
        volumes = list(map(cls.build_volume, spec.get("volumes", [])))
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

    @classmethod
    def build_region(cls, spec):
        return Region(
            name=spec["name"],
            children=map(cls.build_region, spec.get("children", [])),
            shortname=spec.get("shortname", ""),
            description=spec.get("description", ""),
            publications=spec.get("publications", []),
            ebrains_ids=spec.get("ebrains", {}),
            rgb=spec.get("rgb", None),
        )

    @classmethod
    def build_parcellation(cls, spec):
        assert spec.get("@type", None) == "siibra/parcellation/v0.0.1"
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
            ebrains_ids=spec.get("ebrains", {}),
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
        assert spec.get("@type", None) == "siibra/volume/v0.0.1"

        providers = []
        provider_types = [
            NeuroglancerVolume,
            NiftiVolume,
            ZipContainedNiftiVolume,
            NeuroglancerMesh,
            GiftiSurface
        ]
        for srctype, url in spec.get("urls", {}).items():
            for provider_type in provider_types:
                if srctype == provider_type.srctype:
                    providers.append(provider_type(url))
                    break
            else:
                logger.warn(f"Volume source type {srctype} not yet supported, ignoring this specification.")
                print(srctype, url)
    
        return Volume(
            name=spec.get("name", ""),
            space_spec=spec.get("space", {}),
            providers=providers
        )

    @classmethod
    def build_map(cls, spec):
        assert spec.get("@type") == "siibra/map/v0.0.1"
        # maps have no configured identifier - we require the spec filename to build one
        assert "filename" in spec
        basename = path.splitext(path.basename(spec['filename']))[0]
        name = basename.replace('-', ' ').replace('_', ' ')
        identifier = f"{spec['@type'].replace('/','-')}_{basename}"
        volumes = list(map(cls.build_volume, spec.get("volumes", [])))

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
            ebrains_ids=spec.get("ebrains", {})
        )

    @classmethod
    def build_ebrains_dataset(cls, spec):
        assert spec.get("@type", None) == "siibra/snapshots/ebrainsquery/v1"
        return EbrainsDataset(
            id=spec["id"],
            name=spec["name"],
            embargo_status=spec["embargoStatus"],
            cached_data=spec,
        )

    @classmethod
    def build_point(cls, spec):
        assert spec["@type"] == "https://openminds.ebrains.eu/sands/CoordinatePoint"
        space_id = spec["coordinateSpace"]["@id"]
        assert all(c["unit"]["@id"] == "id.link/mm" for c in spec["coordinates"])
        return Point(
            list(np.float16(c["value"]) for c in spec["coordinates"]),
            space_id=space_id,
        )

    @classmethod
    def build_pointset(cls, spec):
        assert spec["@type"] == "tmp/poly"
        space_id = spec["coordinateSpace"]["@id"]
        coords = []
        for coord in spec["coordinates"]:
            assert all(c["unit"]["@id"] == "id.link/mm" for c in coord)
            coords.append(list(np.float16(c["value"]) for c in coord))
        return PointSet(coords, space_id=space_id)

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

        if spectype == "juelich/iav/atlas/v1.0.0":
            return cls.build_atlas(spec)
        elif spectype == "siibra/space/v0.0.1":
            return cls.build_space(spec)
        elif spectype == "siibra/parcellation/v0.0.1":
            return cls.build_parcellation(spec)
        elif spectype == "siibra/volume/v0.0.1":
            return cls.build_volume(spec)
        elif spectype == "siibra/map/v0.0.1":
            return cls.build_map(spec)
        elif spectype == "siibra/space/v0.0.1":
            return cls.build_space(spec)
        elif spectype == "siibra/snapshots/ebrainsquery/v1":
            return cls.build_ebrains_dataset(spec)
        elif spectype == "https://openminds.ebrains.eu/sands/CoordinatePoint":
            return cls.build_point(spec)
        elif spectype == "tmp/poly":
            return cls.build_pointset(spec)
        else:
            raise RuntimeError(f"No factory method for specification type {spectype}.")
