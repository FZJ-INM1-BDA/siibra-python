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
from .features.anchor import AnatomicalAnchor

from .core.atlas import Atlas
from .core.parcellation import Parcellation, ParcellationVersion
from .core.space import Space
from .core.region import Region
from .core.location import Point, PointSet

from .retrieval.datasets import EbrainsDataset
from .retrieval.repositories import ZipfileConnector, GitlabConnector

from .volumes.volume import Volume, NiftiFetcher, NeuroglancerVolumeFetcher, ZipContainedNiftiFetcher
from .volumes.mesh import NeuroglancerMesh, GiftiSurface
from .volumes.map import Map
from .volumes.sparsemap import SparseMap

from .features.receptors import ReceptorDensityFingerprint, ReceptorDensityProfile
from .features.cells import CellDensityFingerprint, CellDensityProfile
from .features.connectivity import StreamlineCounts, StreamlineLengths, FunctionalConnectivity

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
    "siibra/feature/profile/receptor/v0.1": "build_receptor_density_profile",
    "siibra/feature/profile/celldensity/v0.1": "build_cell_density_profile",
    "siibra/feature/fingerprint/receptor/v0.1": "build_receptor_density_fingerprint",
    "siibra/feature/fingerprint/celldensity/v0.1": "build_cell_density_fingerprint",
    "siibra/resource/feature/connectivitymatrix/v0.1": "build_connectivity_matrix",
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
    def extract_volumes(cls, spec, space_id: str = None):
        volume_specs = spec.get("volumes", [])
        if space_id:
            for vspec in volume_specs:
                if 'space' in vspec:
                    logger.warn(f"Replacing space spec {vspec['space']} in volume spec with {space_id}")
                vspec['space'] = {"@id": space_id}
        return list(map(cls.build_volume, volume_specs))

    @classmethod
    def extract_species(cls, spec):
        species_map = {
            "Homo sapiens": "Homo sapiens",
            "0ea4e6ba-2681-4f7d-9fa9-49b915caaac9": "Homo sapiens",
        }
        idspec = spec.get("ebrains", {}).get("minds/core/species/v1.0.0")
        namespec = spec.get("species")
        if idspec in species_map:
            return species_map[idspec]
        elif namespec in species_map:
            return species_map[namespec]
        else:
            logger.error(f"Species specifications '{(idspec, namespec)}' unexpected - check if supported.")
            return None

    @classmethod
    def extract_anchor(cls, spec):
        if spec.get('region'):
            region = spec['region']
        elif spec.get('parcellation', {}).get('@id'):
            # a parcellation is a special region,
            # and can be used if no region is found
            region = spec['parcellation']['@id']
        elif spec.get('parcellation', {}).get('name'):
            region = spec['parcellation']['name']
        else:
            region = None
        if region is None:
            print("no region in spec!")
            print(spec)
            raise RuntimeError
        return AnatomicalAnchor(
            region=region,
            location=None,
            species=cls.extract_species(spec)
        )

    @classmethod
    def extract_connector(cls, spec):
        repospec = spec.get('repository', {})
        spectype = repospec["@type"]
        if spectype == "siibra/repository/zippedfile/v1.0.0":
            return ZipfileConnector(repospec['url'])
        elif spectype == "siibra/repository/gitlab/v1.0.0":
            return GitlabConnector(
                server=repospec['server'],
                project=repospec['project'],
                reftag=repospec['branch']
            )
        else:
            logger.warn(
                "Do not know how to create a repository "
                f"connector from specification type {spectype}."
            )
            return None

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
            volumes=cls.extract_volumes(spec, space_id=spec.get("@id")),
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

        ignored = []
        for srctype, url in spec.get("urls", {}).items():
            for ProviderType in provider_types:
                if srctype == ProviderType.srctype:
                    providers.append(ProviderType(url))
                    break
            else:
                ignored.append(srctype)

        result = Volume(
            name=spec.get("name", ""),
            space_spec=spec.get("space", {}),
            providers=providers
        )

        if len(ignored) > 0:
            if len(providers) == 0:
                logger.error(
                    f"No volume provider for {result} after ignoring specs "
                    f"{', '.join(str(s) for s in set(ignored))}."
                )
            else:
                logger.warn(
                    f"Some volume providers ignored for {result}: "
                    f"{', '.join(str(s) for s in set(ignored))}."
                )

        return result

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
            datasets=cls.extract_datasets(spec)
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
    def build_cell_density_fingerprint(cls, spec):
        return CellDensityFingerprint(
            species=spec['species'],
            regionname=spec['region_name'],
            segmentfiles=spec['segmentfiles'],
            layerfiles=spec['layerfiles'],
            dataset_id=spec['kgId']
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
    def build_cell_density_profile(cls, spec):
        return CellDensityProfile(
            section=spec['section'],
            patch=spec['patch'],
            url=spec['url'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_connectivity_matrix(cls, spec):
        measuretype = spec["modality"]
        kwargs = {
            "cohort": spec["cohort"],
            "subject": spec["subject"],
            "measuretype": measuretype,
            "connector": cls.extract_connector(spec),
            "files": spec.get('files', {}),
            "anchor": cls.extract_anchor(spec),
            "datasets": cls.extract_datasets(spec),
        }
        if measuretype == "StreamlineCounts":
            return StreamlineCounts(**kwargs)
        elif measuretype == "StreamlineLengths":
            return StreamlineLengths(**kwargs)
        elif measuretype == "Functional":
            kwargs["paradigm"] = spec.get("paradigm")
            return FunctionalConnectivity(**kwargs)
        elif measuretype == "RestingState":
            kwargs["paradigm"] = "RestingState"
            return FunctionalConnectivity(**kwargs)
        else:
            raise ValueError(f"Do not know how to build connectivity matrix of type {measuretype}.")

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
