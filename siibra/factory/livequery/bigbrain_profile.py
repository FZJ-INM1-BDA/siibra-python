# Copyright 2018-2024
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

import numpy as np
import pandas as pd
from typing import Iterator
import requests
from io import BytesIO
from nibabel import GiftiImage
from dataclasses import asdict
import json
from hashlib import md5

from .base import LiveQuery
from ...cache import fn_call_cache, Warmup, WarmupLevel, CACHE
from ...commons.logger import logger
from ...concepts.feature import Feature, SUMMARY_NAME
from ...attributes.descriptions import Modality, register_modalities
from ...attributes.datarecipes.tabular import TabularDataRecipe
from ...attributes.locations.layerboundary import (
    LayerBoundary,
    X_PRECALCULATED_BOUNDARY_KEY,
    LAYERS,
)
from ...attributes.locations import intersect, PointCloud, Point, PolyLine
from ...exceptions import UnregisteredAttrCompException, InvalidAttrCompException

modalities_of_interest = [
    Modality(value="Modified silver staining"),
    Modality(value="Layerwise modified silver staining"),
]


X_BIGBRAIN_PROFILE_VERTEX_IDX = "x-siibra/bigbrainprofile/vertex-idx"
X_BIGBRAIN_LAYERWISE_INTENSITY = "x-siibra/bigbrainprofile/layerwiseintensity"


@register_modalities()
def add_modified_silver_staining():
    yield from modalities_of_interest


class BigBrainProfile(LiveQuery[Feature], generates=Feature):
    def generate(self) -> Iterator[Feature]:
        requested_mods = [
            mod for mods in self.find_attributes(Modality) for mod in mods
        ]
        if all(mod not in requested_mods for mod in modalities_of_interest):
            return
        valid_depths, boundary_depths, profiles, vertices = get_all()
        bigbrain_vertices = PointCloud(
            space_id="minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588",
            coordinates=vertices.tolist(),
        )
        root_coords = np.array(bigbrain_vertices.coordinates)
        dtype = {"names": ["x", "y", "z"], "formats": [root_coords.dtype] * 3}
        root_coords = root_coords.view(dtype)

        input_attrs = [attr for attr_col in self.input for attr in attr_col.attributes]

        for input_attr in input_attrs:
            attributes = [*modalities_of_interest]
            try:
                matched_verts = intersect(input_attr, bigbrain_vertices)
            except UnregisteredAttrCompException:
                continue
            except InvalidAttrCompException:
                continue
            if matched_verts is None:
                continue
            if isinstance(matched_verts, Point):
                matched_verts = PointCloud(
                    space_id=matched_verts.space_id,
                    coordinates=[matched_verts.coordinate],
                )
            assert isinstance(matched_verts, PointCloud)
            attributes.append(matched_verts)

            matched_coords = np.array(matched_verts.coordinates)
            matched_coords = matched_coords.view(dtype)
            coordidx_in_matched = np.in1d(root_coords, matched_coords)
            coordidx = np.argwhere(coordidx_in_matched)[:, 0]

            matched_profiles = profiles[coordidx]
            matched_boundary_depths = boundary_depths[coordidx]
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
                    for b in matched_boundary_depths
                ]
            ).reshape((-1, 200))
            means = [
                matched_profiles[layer_labels == layer].mean() for layer in range(1, 7)
            ]
            std = [
                matched_profiles[layer_labels == layer].std() for layer in range(1, 7)
            ]

            dataframe = pd.DataFrame(
                np.array([means, std]).T,
                columns=["mean", "std"],
                index=LAYERS[1:-1],
            )

            hashed_io = md5(
                (
                    json.dumps(asdict(input_attr)) + json.dumps(asdict(matched_verts))
                ).encode("utf-8")
            ).hexdigest()

            filename = CACHE.build_filename(hashed_io, suffix=".csv")
            dataframe.to_csv(filename)
            input_attr = TabularDataRecipe(
                url=filename,
                name=SUMMARY_NAME,
                plot_options={"y": "mean", "yerr": "std", "kind": "bar"},
            )
            attributes.append(input_attr)

            for index in coordidx.tolist():
                _profile = profiles[index]
                depth = np.arange(0.0, 1.0, 1.0 / (profiles[index].shape[0]))

                df = pd.DataFrame(_profile, index=depth)
                filename = CACHE.build_filename(hashed_io, suffix=f"-pr-{index}.csv")
                df.to_csv(filename)

                tabular_attr = TabularDataRecipe(
                    url=filename,
                    name=f"Intensity profile for {bigbrain_vertices[index]}",
                    extra={
                        X_BIGBRAIN_PROFILE_VERTEX_IDX: index,
                    },
                )

                # TODO fix to port/leverage the ops mechanism, rather than
                # this adhoc mess
                layer_boundary = LayerBoundary(
                    extra={
                        X_PRECALCULATED_BOUNDARY_KEY: [
                            PolyLine(
                                coordinates=[
                                    (value, 0, 0)
                                    for value in boundary_depths[index].tolist()
                                ],
                                space_id=None,
                            )
                        ],
                        X_BIGBRAIN_PROFILE_VERTEX_IDX: index,
                    }
                )
                attributes.append(tabular_attr)
                attributes.append(layer_boundary)

            yield Feature(attributes=attributes)


REPO = "https://github.com/kwagstyl/cortical_layers_tutorial/raw/main"
PROFILES_FILE_LEFT = "data/profiles_left.npy"
THICKNESSES_FILE_LEFT = "data/thicknesses_left.npy"
MESH_FILE_LEFT = "data/gray_left_327680.surf.gii"


@Warmup.register_warmup_fn(WarmupLevel.DATA)
def get_all():
    thickness_url = f"{REPO}/{THICKNESSES_FILE_LEFT}"
    valid, boundary_depths = get_thickness(thickness_url)
    profile_url = f"{REPO}/{PROFILES_FILE_LEFT}"
    fresh_call = not get_profile.check_call_in_cache(profile_url, valid)
    if fresh_call:
        logger.info(
            "First request to BigBrain profiles. Preprocessing the data "
            "now. This may take a little."
        )
    profile = get_profile(profile_url, valid)
    if fresh_call:
        logger.info("Data cached.")
    vertices_url = f"{REPO}/{MESH_FILE_LEFT}"
    vertices = get_vertices(vertices_url, valid)

    return valid, boundary_depths, profile, vertices


@fn_call_cache
def get_thickness(url: str):
    assert url.endswith(".npy"), "Thickness URL must end with .npy"

    resp = requests.get(url)
    resp.raise_for_status()
    thickness: np.ndarray = np.load(BytesIO(resp.content)).T

    total_thickness = thickness[:, :-1].sum(
        1
    )  # last column is the computed total thickness
    valid = np.where(total_thickness > 0)[0]

    boundary_depths = np.c_[
        np.zeros_like(valid),
        (thickness[valid, :] / total_thickness[valid, None]).cumsum(1),
    ]
    boundary_depths[:, -1] = 1  # account for float calculation errors

    return valid, boundary_depths


@fn_call_cache
def get_profile(url: str, valid: np.ndarray):
    assert url.endswith(".npy"), "Profile URL must end with .npy"
    resp = requests.get(url)
    resp.raise_for_status()
    profiles_l_all: np.ndarray = np.load(BytesIO(resp.content))
    profiles = profiles_l_all[valid, :]
    return profiles


@fn_call_cache
def get_vertices(url: str, valid: np.ndarray):
    assert url.endswith(".gii"), "Vertices URL must end with .gii"
    resp = requests.get(url)
    resp.raise_for_status()
    mesh = GiftiImage.from_bytes(resp.content)
    mesh_vertices: np.ndarray = mesh.darrays[0].data
    vertices = mesh_vertices[valid, :]
    return vertices
