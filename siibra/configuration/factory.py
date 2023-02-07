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

from ..commons import logger, Species
from ..features import anchor
from ..features.molecular import receptor_density_fingerprint, receptor_density_profile
from ..features.cellular import cell_density_profile, layerwise_cell_density
from ..features.basetypes import volume_of_interest
from ..core import atlas, parcellation, space, region
from ..locations import point, pointset
from ..retrieval import datasets, repositories
from ..volumes import gifti, volume, nifti, neuroglancer, sparsemap, parcellationmap
from ..features import connectivity

from os import path
import json
import numpy as np
from typing import List, Type
import pandas as pd
from io import BytesIO

MIN_VOLUMES_FOR_SPARSE_MAP = 100

BUILDFUNCS = {
    "juelich/iav/atlas/v1.0.0": "build_atlas",
    "siibra/space/v0.0.1": "build_space",
    "siibra/parcellation/v0.0.1": "build_parcellation",
    "siibra/volume/v0.0.1": "build_volume",
    "siibra/map/v0.0.1": "build_map",
    "siibra/snapshots/ebrainsquery/v1": "build_ebrains_dataset",
    "https://openminds.ebrains.eu/sands/CoordinatePoint": "build_point",
    "siibra/location/point/v0.1": "build_point",
    "tmp/poly": "build_pointset",
    "siibra/location/pointset/v0.1": "build_pointset",
    "siibra/feature/profile/receptor/v0.1": "build_receptor_density_profile",
    "siibra/feature/profile/celldensity/v0.1": "build_cell_density_profile",
    "siibra/feature/fingerprint/receptor/v0.1": "build_receptor_density_fingerprint",
    "siibra/feature/fingerprint/celldensity/v0.1": "build_cell_density_fingerprint",
    "siibra/feature/connectivitymatrix/v0.2": "build_connectivity_matrix",
    "siibra/feature/voi/v0.1": "build_volume_of_interest",
}


class Factory:

    _warnings_issued = []

    @classmethod
    def extract_datasets(cls, spec):
        result = []
        if "minds/core/dataset/v1.0.0" in spec.get("ebrains", {}):
            result.append(
                datasets.EbrainsDataset(id=spec["ebrains"]["minds/core/dataset/v1.0.0"])
            )
        return result

    @classmethod
    def extract_volumes(cls, spec, space_id: str = None, name: str = None):
        volume_specs = spec.get("volumes", [])
        for vspec in volume_specs:
            if space_id:
                if 'space' in vspec:
                    logger.warn(f"Replacing space spec {vspec['space']} in volume spec with {space_id}")
                vspec['space'] = {"@id": space_id}
            if name and vspec.get('name') is None:  # only use provided name if the volume has no specific name
                vspec['name'] = name
        return list(map(cls.build_volume, volume_specs))

    @classmethod
    def extract_decoder(cls, spec):
        decoder_spec = spec.get("decoder", {})
        if decoder_spec["@type"].endswith('csv'):
            kwargs = {k: v for k, v in decoder_spec.items() if k != "@type"}
            return lambda b: pd.read_csv(BytesIO(b), **kwargs)
        else:
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

        if 'location' in spec:
            location = cls.from_json(spec['location'])
        else:
            location = None

        if (region is None) and (location is None):
            print(spec)
            raise RuntimeError("Spec provides neither region or location - no anchor can be extracted.")

        if 'species' in spec:
            species = Species.decode(spec['species'])
        elif ('ebrains' in spec):
            species = Species.decode(spec['ebrains'])
        else:
            raise ValueError(f"No species information found in spec {spec}")

        return anchor.AnatomicalAnchor(
            region=region,
            location=location,
            species=species
        )

    @classmethod
    def extract_connector(cls, spec):
        repospec = spec.get('repository', {})
        spectype = repospec["@type"]
        if spectype == "siibra/repository/zippedfile/v1.0.0":
            return repositories.ZipfileConnector(repospec['url'])
        elif spectype == "siibra/repository/gitlab/v1.0.0":
            return repositories.GitlabConnector(
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
        a = atlas.Atlas(
            spec["@id"],
            spec["name"],
            species=Species.decode(spec.get('species')),
        )
        for space_id in spec["spaces"]:
            a._register_space(space_id)
        for parcellation_id in spec["parcellations"]:
            a._register_parcellation(parcellation_id)
        return a

    @classmethod
    def build_space(cls, spec):
        return space.Space(
            identifier=spec["@id"],
            name=spec["name"],
            species=Species.decode(spec.get('species')),
            volumes=cls.extract_volumes(spec, space_id=spec.get("@id"), name=spec.get("name")),
            shortname=spec.get("shortName", ""),
            description=spec.get("description"),
            modality=spec.get("modality"),
            publications=spec.get("publications", []),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_region(cls, spec):
        return region.Region(
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
        p = parcellation.Parcellation(
            identifier=spec["@id"],
            name=spec["name"],
            species=Species.decode(spec.get('species')),
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
            version = parcellation.ParcellationVersion(
                name=versionspec.get("name", None),
                parcellation=p,
                collection=versionspec.get("collectionName", None),
                prev_id=versionspec.get("@prev", None),
                next_id=versionspec.get("@next", None),
                deprecated=versionspec.get("deprecated", False)
            )
            p.version = version

        return p

    @classmethod
    def build_volume(cls, spec):
        providers: List[volume.VolumeProvider] = []
        provider_types: List[Type[volume.VolumeProvider]] = [
            neuroglancer.NeuroglancerProvider,
            neuroglancer.NeuroglancerMesh,
            neuroglancer.NeuroglancerSurfaceMesh,
            nifti.NiftiProvider,
            nifti.ZipContainedNiftiProvider,
            gifti.GiftiMesh,
            gifti.GiftiSurfaceLabeling
        ]

        for srctype, provider_spec in spec.get("providers", {}).items():
            for ProviderType in provider_types:
                if srctype == ProviderType.srctype:
                    providers.append(ProviderType(provider_spec))
                    break
            else:
                if srctype not in cls._warnings_issued:
                    logger.warn(f"No provider defined for volume Source type {srctype}")
                    cls._warnings_issued.append(srctype)

        assert all([isinstance(provider, volume.VolumeProvider) for provider in providers])
        result = volume.Volume(
            space_spec=spec.get("space", {}),
            providers=providers,
            name=spec.get("name", {}),
            variant=spec.get("variant"),
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

        Maptype = parcellationmap.Map
        if len(volumes) > MIN_VOLUMES_FOR_SPARSE_MAP:
            logger.debug(
                f"Using sparse map for {spec['filename']} to code its {len(volumes)} volumes efficiently."
            )
            Maptype = sparsemap.SparseMap
        else:
            max_z = max(
                d.get('z', 0)
                for _, l in spec.get("indices", {}).items()
                for d in l
            ) + 1
            if max_z > MIN_VOLUMES_FOR_SPARSE_MAP:
                logger.debug(
                    f"Using sparse map for {spec['filename']} to code its {max_z} z levels efficiently."
                )
                Maptype = sparsemap.SparseMap

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
        return datasets.EbrainsDataset(
            id=spec["id"],
            name=spec["name"],
            embargo_status=spec["embargoStatus"],
            cached_data=spec,
        )

    @classmethod
    def build_point(cls, spec):
        if spec.get('@type') == "https://openminds.ebrains.eu/sands/CoordinatePoint":
            space_id = spec["coordinateSpace"]["@id"]
            coord = list(np.float16(c["value"]) for c in spec["coordinates"])
            assert all(c["unit"]["@id"] == "id.link/mm" for c in spec["coordinates"])
        elif spec.get('@type') == "siibra/location/point/v0.1":
            space_id = spec.get("space").get("@id")
            coord = spec.get("coordinate")
        else:
            raise ValueError(f"Unknown point specification: {spec}")
        return point.Point(
            coordinatespec=coord,
            space=space_id,
        )

    @classmethod
    def build_pointset(cls, spec):
        if spec.get('@type') == '/tmp/poly':
            space_id = spec["coordinateSpace"]["@id"]
            coords = []
            for coord in spec["coordinates"]:
                assert all(c["unit"]["@id"] == "id.link/mm" for c in coord)
                coords.append(list(np.float16(c["value"]) for c in coord))
        elif spec.get('@type') == 'siibra/location/pointset/v0.1':
            space_id = spec.get("space").get("@id")
            coords = [tuple(c) for c in spec.get("coordinates")]
        return pointset.PointSet(coords, space=space_id)

    @classmethod
    def build_receptor_density_fingerprint(cls, spec):
        return receptor_density_fingerprint.ReceptorDensityFingerprint(
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_cell_density_fingerprint(cls, spec):
        return layerwise_cell_density.LayerwiseCellDensity(
            segmentfiles=spec['segmentfiles'],
            layerfiles=spec['layerfiles'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_receptor_density_profile(cls, spec):
        return receptor_density_profile.ReceptorDensityProfile(
            receptor=spec['receptor'],
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_cell_density_profile(cls, spec):
        return cell_density_profile.CellDensityProfile(
            section=spec['section'],
            patch=spec['patch'],
            url=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_volume_of_interest(cls, spec):
        vol = cls.build_volume(spec)
        return volume_of_interest.VolumeOfInterest(
            name=vol.name,
            modality=spec.get('modality', ""),
            region=spec.get('region', None),
            space_spec=vol._space_spec,
            providers=vol._providers.values(),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    def build_connectivity_matrix(cls, spec):
        modality = spec["modality"]
        kwargs = {
            "cohort": spec["cohort"],
            "modality": modality,
            "regions": spec["regions"],
            "connector": cls.extract_connector(spec),
            "decode_func": cls.extract_decoder(spec),
            "files": spec.get("files", {}),
            "anchor": cls.extract_anchor(spec),
            "description": spec.get("description", ""),
            "datasets": cls.extract_datasets(spec),
        }
        if modality == "StreamlineCounts":
            return connectivity.StreamlineCounts(**kwargs)
        elif modality == "StreamlineLengths":
            return connectivity.StreamlineLengths(**kwargs)
        elif modality == "Functional":
            kwargs["paradigm"] = spec.get("paradigm")
            return connectivity.FunctionalConnectivity(**kwargs)
        elif modality == "RestingState":
            kwargs["paradigm"] = "RestingState"
            return connectivity.FunctionalConnectivity(**kwargs)
        else:
            raise ValueError(f"Do not know how to build connectivity matrix of type {modality}.")

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
