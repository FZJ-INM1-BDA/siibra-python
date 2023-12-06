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
from ..features import anchor, connectivity
from ..features.tabular import (
    receptor_density_profile,
    receptor_density_fingerprint,
    cell_density_profile,
    layerwise_cell_density,
    regional_timeseries_activity
)
from ..features.image import sections, volume_of_interest
from ..core import atlas, parcellation, space, region
from ..locations import point, pointset
from ..retrieval import datasets, repositories
from ..volumes import volume, sparsemap, parcellationmap
from ..volumes.providers import provider, gifti, neuroglancer, nifti

from os import path
import json
import numpy as np
from typing import List, Type, Dict, Callable
import pandas as pd
from io import BytesIO
from functools import wraps


_registered_build_fns: Dict[str, Callable] = {}


def build_type(type_str: str):
    def outer(fn):
        _registered_build_fns[type_str] = fn

        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner
    return outer


class Factory:

    _warnings_issued = []

    @classmethod
    def extract_datasets(cls, spec):
        result = []
        if "minds/core/dataset/v1.0.0" in spec.get("ebrains", {}):
            result.append(
                datasets.EbrainsDataset(id=spec["ebrains"]["minds/core/dataset/v1.0.0"])
            )
        if "openminds/DatasetVersion" in spec.get("ebrains", {}):
            result.append(
                datasets.EbrainsV3DatasetVersion(id=spec["ebrains"]["openminds/DatasetVersion"])
            )
        if "openminds/Dataset" in spec.get("ebrains", {}):
            result.append(
                datasets.EbrainsV3Dataset(id=spec["ebrains"]["openminds/Dataset"])
            )
        if "publications" in spec:
            result.extend(
                datasets.GenericDataset(
                    name=pub["name"],
                    contributors=pub["authors"],
                    url=pub["url"],
                    description=pub["description"],
                    license=pub.get("license")
                )
                for pub in spec["publications"] if pub.get('name')
            )
        return result

    @classmethod
    def extract_volumes(
        cls,
        spec,
        space_id: str = None,
        names: List[str] = None,
        name_prefix: str = ""
    ):
        volume_specs = spec.get("volumes", [])
        if names:
            if len(names) != len(volume_specs) and len(names) == 1:
                variants = [vol['variant'] for vol in volume_specs]
                names = [f"{name_prefix}{names[0]} {var} variant" for var in variants]
        else:
            names = [f"{name_prefix} - volume {i}" for i in range(len(volume_specs))]
        for i, vspec in enumerate(volume_specs):
            if space_id:
                if 'space' in vspec:
                    logger.warning(f"Replacing space spec {vspec['space']} in volume spec with {space_id}")
                vspec['space'] = {"@id": space_id}
            if names and vspec.get('name') is None:  # only use provided name if the volume has no specific name
                vspec['name'] = names[i]
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
        if spectype == "siibra/repository/localfolder/v1.0.0":
            return repositories.LocalFileRepository(repospec['folder'])
        if spectype == "siibra/repository/gitlab/v1.0.0":
            return repositories.GitlabConnector(
                server=repospec['server'],
                project=repospec['project'],
                reftag=repospec['branch']
            )

        logger.warning(
            "Do not know how to create a repository "
            f"connector from specification type {spectype}."
        )
        return None

    @classmethod
    @build_type("juelich/iav/atlas/v1.0.0")
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
    @build_type("siibra/space/v0.0.1")
    def build_space(cls, spec):
        return space.Space(
            identifier=spec["@id"],
            name=spec["name"],
            species=Species.decode(spec.get('species')),
            volumes=cls.extract_volumes(spec, space_id=spec.get("@id"), names=[spec.get("name")]),
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
            spec=spec,
        )

    @classmethod
    @build_type("siibra/parcellation/v0.0.1")
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
    @build_type("siibra/volume/v0.0.1")
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
                    logger.warning(f"No provider defined for volume Source type {srctype}")
                    cls._warnings_issued.append(srctype)

        assert all([isinstance(p, provider.VolumeProvider) for p in providers])
        result = volume.Volume(
            space_spec=spec.get("space", {}),
            providers=providers,
            name=spec.get("name", ""),
            variant=spec.get("variant"),
            datasets=cls.extract_datasets(spec),
        )

        return result

    @classmethod
    @build_type("siibra/map/v0.0.1")
    def build_map(cls, spec):
        # maps have no configured identifier - we require the spec filename to build one
        assert "filename" in spec
        basename = path.splitext(path.basename(spec['filename']))[0]
        name = basename.replace('-', ' ').replace('_', ' ').replace('continuous', 'statistical')
        identifier = f"{spec['@type'].replace('/','-')}_{basename}"
        volumes = cls.extract_volumes(spec, space_id=spec["space"].get("@id"), name_prefix=basename)

        if spec.get("sparsemap", {}).get("is_sparsemap"):
            Maptype = sparsemap.SparseMap
        else:
            Maptype = parcellationmap.Map
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
    @build_type("siibra/snapshots/ebrainsquery/v1")
    def build_ebrains_dataset(cls, spec):
        return datasets.EbrainsDataset(
            id=spec["id"],
            name=spec["name"],
            embargo_status=spec["embargoStatus"],
            cached_data=spec,
        )

    @classmethod
    @build_type("https://openminds.ebrains.eu/sands/CoordinatePoint")
    @build_type("siibra/location/point/v0.1")
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
    @build_type("tmp/poly")
    @build_type("siibra/location/pointset/v0.1")
    def build_pointset(cls, spec):
        if spec.get('@type') == 'tmp/poly':
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
    @build_type("siibra/feature/fingerprint/receptor/v0.1")
    def build_receptor_density_fingerprint(cls, spec):
        return receptor_density_fingerprint.ReceptorDensityFingerprint(
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    @build_type("siibra/feature/fingerprint/celldensity/v0.1")
    def build_cell_density_fingerprint(cls, spec):
        return layerwise_cell_density.LayerwiseCellDensity(
            segmentfiles=spec['segmentfiles'],
            layerfiles=spec['layerfiles'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    @build_type("siibra/feature/profile/receptor/v0.1")
    def build_receptor_density_profile(cls, spec):
        return receptor_density_profile.ReceptorDensityProfile(
            receptor=spec['receptor'],
            tsvfile=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    @build_type("siibra/feature/profile/celldensity/v0.1")
    def build_cell_density_profile(cls, spec):
        return cell_density_profile.CellDensityProfile(
            section=spec['section'],
            patch=spec['patch'],
            url=spec['file'],
            anchor=cls.extract_anchor(spec),
            datasets=cls.extract_datasets(spec),
        )

    @classmethod
    @build_type("siibra/feature/section/v0.1")
    def build_section(cls, spec):
        vol = cls.build_volume(spec)
        kwargs = {
            "name": spec.get('name', ""),
            "region": spec.get('region', None),
            "space_spec": vol._space_spec,
            "providers": vol._providers.values(),
            "datasets": cls.extract_datasets(spec),
        }
        modality = spec.get('modality', "")
        if modality == "cell body staining":
            return sections.CellbodyStainedSection(**kwargs)
        else:
            raise ValueError(f"No method for building image section feature type {modality}.")

    @classmethod
    @build_type("siibra/feature/voi/v0.1")
    def build_volume_of_interest(cls, spec):
        vol = cls.build_volume(spec)
        kwargs = {
            "name": spec.get('name', ""),
            "region": spec.get('region', None),
            "space_spec": vol._space_spec,
            "providers": vol._providers.values(),
            "datasets": cls.extract_datasets(spec),
        }
        modality = spec.get('modality', "")
        if modality == "cell body staining":
            return volume_of_interest.CellBodyStainedVolumeOfInterest(**kwargs)
        elif modality == "blockface":
            return volume_of_interest.BlockfaceVolumeOfInterest(**kwargs)
        elif modality == "PLI HSV fibre orientation map":
            return volume_of_interest.PLIVolumeOfInterest(
                modality="HSV fibre orientation map", **kwargs
            )
        elif modality == "transmittance":
            return volume_of_interest.PLIVolumeOfInterest(
                modality="transmittance", **kwargs
            )
        elif modality == "XPCT":
            return volume_of_interest.XPCTVolumeOfInterest(
                modality="XPCT", **kwargs
            )
        elif modality == "DTI":
            return volume_of_interest.DTIVolumeOfInterest(
                modality=modality, **kwargs
            )
        # elif modality == "segmentation":
        #     return volume_of_interest.SegmentedVolumeOfInterest(**kwargs)
        elif "MRI" in modality:
            return volume_of_interest.MRIVolumeOfInterest(
                modality=modality, **kwargs
            )
        elif modality == "LSFM":
            return volume_of_interest.LSFMVolumeOfInterest(
                modality="Light Sheet Fluorescence Microscopy", **kwargs
            )
        else:
            raise ValueError(f"No method for building image section feature type {modality}.")

    @classmethod
    @build_type("siibra/feature/connectivitymatrix/v0.3")
    def build_connectivity_matrix(cls, spec):
        files = spec.get("files", {})
        modality = spec["modality"]
        try:
            conn_cls = getattr(connectivity, modality)
        except Exception:
            raise ValueError(f"No method for building connectivity matrix of type {modality}.")

        decoder_func = cls.extract_decoder(spec)
        repo_connector = cls.extract_connector(spec) if spec.get('repository', None) else None
        if repo_connector is None:
            base_url = spec.get("base_url", "")
        kwargs = {
            "cohort": spec.get("cohort", ""),
            "modality": modality,
            "regions": spec["regions"],
            "connector": repo_connector,
            "decode_func": decoder_func,
            "anchor": cls.extract_anchor(spec),
            "description": spec.get("description", ""),
            "datasets": cls.extract_datasets(spec)
        }
        paradigm = spec.get("paradigm")
        if paradigm:
            kwargs["paradigm"] = paradigm
        files_indexed_by = spec.get("files_indexed_by", "subject")
        assert files_indexed_by in ["subject", "feature"]
        conn_by_file = []
        for fkey, filename in files.items():
            kwargs.update({
                "filename": filename,
                "subject": fkey if files_indexed_by == "subject" else "average",
                "feature": fkey if files_indexed_by == "feature" else None,
                "connector": repo_connector or base_url + filename
            })
            conn_by_file.append(conn_cls(**kwargs))
        return conn_by_file

    @classmethod
    @build_type("siibra/feature/timeseries/activity/v0.1")
    def build_activity_timeseries(cls, spec):
        files = spec.get("files", {})
        modality = spec["modality"]
        try:
            timeseries_cls = getattr(regional_timeseries_activity, modality)
        except Exception:
            raise ValueError(f"No method for building signal table of type {modality}.")

        kwargs = {
            "cohort": spec.get("cohort", ""),
            "modality": modality,
            "regions": spec["regions"],
            "connector": cls.extract_connector(spec),
            "decode_func": cls.extract_decoder(spec),
            "anchor": cls.extract_anchor(spec),
            "description": spec.get("description", ""),
            "datasets": cls.extract_datasets(spec),
            "timestep": spec.get("timestep")
        }
        paradigm = spec.get("paradigm")
        if paradigm:
            kwargs["paradigm"] = paradigm
        timeseries_by_file = []
        for fkey, filename in files.items():
            kwargs.update({
                "filename": filename,
                "subject": fkey
            })
            timeseries_by_file.append(timeseries_cls(**kwargs))
        return timeseries_by_file

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
        if spectype in _registered_build_fns:
            return _registered_build_fns[spectype](cls, spec)
        else:
            raise RuntimeError(f"No factory method for specification type {spectype}.")
