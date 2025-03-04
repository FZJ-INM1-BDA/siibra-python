# Copyright 2018-2025
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
"""Matches BigBrain intensity profiles extracted by Wagstyl et al. to volumes."""

from typing import List
from os import path

import numpy as np
from scipy.spatial import KDTree

from . import query
from ..features import anchor as _anchor
from ..features.tabular import bigbrain_intensity_profile, layerwise_bigbrain_intensities
from ..features.image import CellbodyStainedSection, BigBrain1MicronPatch
from ..commons import logger, siibra_tqdm
from ..locations import point, pointcloud, location, experimental
from ..core import structure
from ..core.concept import get_registry
from ..retrieval import requests, cache
from ..retrieval.datasets import GenericDataset
from ..volumes import Volume, from_array


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial/raw/main"
    PROFILES_FILE_LEFT = "data/profiles_left.npy"
    THICKNESSES_FILE_LEFT = "data/thicknesses_left.npy"
    MESH_FILE_LEFT = "data/gray_left_327680.surf.gii"
    _profiles = None
    _vertices = None
    _boundary_depths = None
    DATASET = GenericDataset(
        name="HIBALL workshop on cortical layers",
        contributors=[
            "Konrad Wagstyl",
            "Stéphanie Larocque",
            "Guillem Cucurull",
            "Claude Lepage",
            "Joseph Paul Cohen",
            "Sebastian Bludau",
            "Nicola Palomero-Gallagher",
            "Lindsay B. Lewis",
            "Thomas Funck",
            "Hannah Spitzer",
            "Timo Dickscheid",
            "Paul C. Fletcher",
            "Adriana Romero",
            "Karl Zilles",
            "Katrin Amunts",
            "Yoshua Bengio",
            "Alan C. Evans",
        ],
        url="https://github.com/kwagstyl/cortical_layers_tutorial/",
        description="Cortical profiles of BigBrain staining intensities computed by Konrad Wagstyl, "
        "as described in the publication 'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678."
        "The data is taken from the tutorial at "
        "https://github.com/kwagstyl/cortical_layers_tutorial. Each vertex is "
        "assigned to the regional map when queried.",
    )

    def __init__(self):
        if self._profiles is None:
            self.__class__._load()

    @property
    def profile_labels(self):
        return np.arange(0.0, 1.0, 1.0 / self._profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last column which is the total thickness
        thickness = requests.HttpRequest(
            f"{cls.REPO}/{cls.THICKNESSES_FILE_LEFT}"
        ).data.T
        total_thickness = thickness[:, :-1].sum(
            1
        )  # last column is the computed total thickness
        valid = np.where(total_thickness > 0)[0]
        cls._boundary_depths = np.c_[
            np.zeros_like(valid),
            (thickness[valid, :] / total_thickness[valid, None]).cumsum(1),
        ]
        cls._boundary_depths[:, -1] = 1  # account for float calculation errors

        # find profiles with valid thickness
        profile_l_url = f"{cls.REPO}/{cls.PROFILES_FILE_LEFT}"
        if not path.exists(cache.CACHE.build_filename(profile_l_url)):
            logger.info(
                "First request to BigBrain profiles. Preprocessing the data "
                "now. This may take a little."
            )
        profiles_l_all = requests.HttpRequest(profile_l_url).data
        cls._profiles = profiles_l_all[valid, :]

        # read mesh vertices
        mesh_left = requests.HttpRequest(f"{cls.REPO}/{cls.MESH_FILE_LEFT}").data
        mesh_vertices = mesh_left.darrays[0].data
        cls._vertices = mesh_vertices[valid, :]

        logger.debug(f"{cls._profiles.shape[0]} BigBrain intensity profiles.")
        assert cls._vertices.shape[0] == cls._profiles.shape[0]

    def __len__(self):
        return self._vertices.shape[0]


cache.Warmup.register_warmup_fn()(lambda: WagstylProfileLoader._load())


class BigBrainProfileQuery(
    query.LiveQuery,
    args=[],
    FeatureType=bigbrain_intensity_profile.BigBrainIntensityProfile,
):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(
        self, concept: structure.BrainStructure, **kwargs
    ) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
        loader = WagstylProfileLoader()
        mesh_vertices = pointcloud.PointCloud(loader._vertices, space="bigbrain")
        matched = concept.intersection(
            mesh_vertices
        )  # returns a reduced PointCloud with og indices as labels
        if matched is None:
            return []
        if isinstance(matched, point.Point):
            matched = pointcloud.from_points([matched])
        assert isinstance(matched, pointcloud.PointCloud)
        if isinstance(concept, location.Location):
            mesh_as_list = mesh_vertices.as_list()
            matched.labels = [mesh_as_list.index(v.coordinate) for v in matched]
        indices = matched.labels
        assert indices is not None
        features = []
        for i in indices:
            anchor = _anchor.AnatomicalAnchor(
                location=point.Point(loader._vertices[i], space="bigbrain"),
                region=str(concept),
                species="Homo sapiens",
            )
            prof = bigbrain_intensity_profile.BigBrainIntensityProfile(
                anchor=anchor,
                depths=loader.profile_labels,
                values=loader._profiles[i],
                boundaries=loader._boundary_depths[i],
            )
            prof.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
                query_structure=concept,
                assigned_structure=concept,
                qualification=_anchor.Qualification.CONTAINED,
                explanation=f"Surface vertex of BigBrain cortical profile was filtered using {concept}",
            )
            prof.datasets = [WagstylProfileLoader.DATASET]
            features.append(prof)

        return features


class LayerwiseBigBrainIntensityQuery(
    query.LiveQuery,
    args=[],
    FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities,
):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(
        self, concept: structure.BrainStructure, **kwargs
    ) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:

        loader = WagstylProfileLoader()
        mesh_vertices = pointcloud.PointCloud(loader._vertices, space="bigbrain")
        matched = concept.intersection(
            mesh_vertices
        )  # returns a reduced PointCloud with og indices as labels if the concept is a region
        if matched is None:
            return []
        if isinstance(matched, point.Point):
            matched = pointcloud.from_points([matched])
        assert isinstance(matched, pointcloud.PointCloud)
        if isinstance(concept, location.Location):
            mesh_as_list = mesh_vertices.as_list()
            matched.labels = [mesh_as_list.index(v.coordinate) for v in matched]
        indices = matched.labels
        matched_profiles = loader._profiles[indices, :]
        boundary_depths = loader._boundary_depths[indices, :]
        # compute array of layer labels for all coefficients in profiles_left
        N = matched_profiles.shape[1]
        prange = np.arange(N)
        layer_labels = 7 - np.array(
            [
                [
                    np.array(
                        [
                            [(prange < T) * 1]
                            for i, T in enumerate((b * N).astype("int"))
                        ]
                    )
                    .squeeze()
                    .sum(0)
                ]
                for b in boundary_depths
            ]
        ).reshape((-1, 200))

        anchor = _anchor.AnatomicalAnchor(
            location=pointcloud.PointCloud(
                loader._vertices[indices, :], space="bigbrain"
            ),
            region=str(concept),
            species="Homo sapiens",
        )
        result = layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities(
            anchor=anchor,
            means=[
                matched_profiles[layer_labels == layer].mean() for layer in range(1, 7)
            ],
            stds=[
                matched_profiles[layer_labels == layer].std() for layer in range(1, 7)
            ],
        )
        result.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
            query_structure=concept,
            assigned_structure=concept,
            qualification=_anchor.Qualification.CONTAINED,
            explanation=f"Surface vertices of BigBrain cortical profiles were filtered using {concept}",
        )
        result.datasets = [WagstylProfileLoader.DATASET]

        return [result]


class BigBrain1MicronPatchQuery(
    query.LiveQuery, args=[], FeatureType=BigBrain1MicronPatch
):
    """
    Sample approximately orthogonal cortical image patches
    from BigBrain 1 micron sections, guided by an image volume
    in a supported reference space providing. The image
    volume is used as a weighted mask to extract patches
    along the cortical midsurface with nonzero weights in the
    input image.
    An optional lower_threshold can be used to narrow down
    the search
    The weight is stored with the resulting features.
    """

    def __init__(self, lower_threshold=0.):
        self.layermap = get_registry("Map").get("cortical layers bigbrain")
        self.lower_threshold = lower_threshold
        query.LiveQuery.__init__(self)

    def query(
        self, concept: structure.BrainStructure, **kwargs
    ) -> List[BigBrain1MicronPatch]:

        # make sure input is an image volume
        # TODO function should be extended to deal with other concepts as well
        if not isinstance(concept, Volume):
            logger.warning(
                "Querying BigBrain1MicronPatch features requires to "
                "query with an image volume."
            )
            return []

        # threshold image volume, if requested
        if self.lower_threshold > 0.0:
            logger.info(
                f"Applying lower threshold of {self.lower_threshold} "
                "for BigBrain 1 micron patch query."
            )
            img = concept.fetch()
            arr = np.asanyarray(img.dataobj)
            arr[arr < self.lower_threshold] = 0
            query_vol = from_array(arr, img.affine, space=concept.space, name="filtered volume")
        else:
            query_vol = concept
        bb_bbox = query_vol.get_boundingbox().warp('bigbrain')

        # find 1 micron BigBrain sections intersecting the thresholded volume
        sections: List[CellbodyStainedSection] = [
            s
            for s in CellbodyStainedSection._get_instances()
            if isinstance(s, CellbodyStainedSection)
            and s.get_boundingbox(clip=False).intersects(query_vol)
        ]
        if not sections:
            return []

        # extract relevant patches
        features = []
        for hemisphere in ["left", "right"]:

            # get layer 4 mesh in the hemisphere
            l4 = self.layermap.parcellation.get_region("4 " + hemisphere)
            l4mesh = self.layermap.fetch(l4, format="mesh")
            layerverts = {
                n: self.layermap.fetch(region=n, format="mesh")["verts"]
                for n in self.layermap.regions if hemisphere in n
            }
            l4verts = pointcloud.PointCloud(layerverts[l4.name], "bigbrain")
            if not l4verts.boundingbox.intersects(bb_bbox):
                continue

            # for each relevant BigBrain 1 micron section, intersect layer IV mesh
            # to obtain midcortex-locations, and build their orthogonal patches.
            # store the concept's value with the patch.
            vertex_tree = KDTree(layerverts[l4.name])
            for s in siibra_tqdm(
                sections, unit="sections", desc=f"Sampling patches in {hemisphere} hemisphere"
            ):

                # compute layer IV contour in the image plane
                imgplane = experimental.Plane.from_image(s)
                try:
                    contour_segments = imgplane.intersect_mesh(l4mesh)
                except AssertionError:
                    logger.error(f"Could not intersect with layer 4 mesh: {s.name}")
                    continue
                if len(contour_segments) == 0:
                    continue

                # score the contour points with the query image volume
                all_points = pointcloud.from_points(sum(map(list, contour_segments), []))
                all_probs = query_vol.evaluate_points(all_points)
                points_prob_lookup = {
                    pt.coordinate: prob
                    for pt, prob in zip(all_points, all_probs)
                    if prob >= self.lower_threshold
                }
                if len(points_prob_lookup) == 0:
                    continue

                # For each contour point,
                # - find the closest BigBrain layer surface vertex,
                # - build the profile of corresponding vertices across all layers
                # - project the profile to the image section
                # - determine the oriented patch along the profile
                _, indices = vertex_tree.query(np.array(list(points_prob_lookup.keys())))
                for prob, nnb in zip(points_prob_lookup.values(), indices):

                    prof = pointcloud.Contour(
                        [
                            layerverts[_][nnb]
                            for _ in self.layermap.regions
                            if hemisphere in _
                        ],
                        space=self.layermap.space,
                    )
                    patch = imgplane.get_enclosing_patch(prof)
                    if patch is None:
                        continue

                    anchor = _anchor.AnatomicalAnchor(
                        location=patch, species="Homo sapiens"
                    )
                    anchor._assignments[concept] = _anchor.AnatomicalAssignment(
                        query_structure=query_vol,
                        assigned_structure=s.anchor.volume,
                        qualification=_anchor.Qualification.CONTAINED
                    )
                    features.append(
                        BigBrain1MicronPatch(
                            patch=patch,
                            profile=prof,
                            section=s,
                            vertex=nnb,
                            relevance=prob,
                            anchor=anchor
                        )
                    )

        # return the patches sorted by relevance (ie. probability)
        return sorted(features, key=lambda p: p.relevance, reverse=True)
