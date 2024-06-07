import requests
import numpy as np
from io import BytesIO
from nibabel import GiftiImage
import pandas as pd

from ...assignment import register_collection_generator
from ...concepts import Feature, AttributeCollection
from ...descriptions import Modality, register_modalities
from ...dataitems.tabular import Tabular, X_DATA
from ...cache import fn_call_cache, Warmup, WarmupLevel
from ...commons import logger
from ...locations import intersect, PointCloud, Pt, Polyline
from ...locations.layerboundary import (
    LayerBoundary,
    X_PRECALCULATED_BOUNDARY_KEY,
    LAYERS,
)
from ...exceptions import UnregisteredAttrCompException

modality_of_interest = Modality(value="Modified silver staining")
X_BIGBRAIN_PROFILE_VERTEX_IDX = "x-siibra/bigbrainprofile/vertex-idx"
X_BIGBRAIN_LAYERWISE_INTENSITY = "x-siibra/bigbrainprofile/layerwiseintensity"


@register_modalities()
def add_modified_silver_staining():
    yield modality_of_interest


@register_collection_generator(Feature)
def query_bigbrain_profile(input: AttributeCollection):
    if modality_of_interest not in input._find(Modality):
        return []
    valid, boundary_depths, profile, vertices = _get_all()
    ptcld = PointCloud(
        space_id="minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588",
        coordinates=vertices.tolist(),
    )
    root_coords = np.array(ptcld.coordinates)
    dtype = {"names": ["x", "y", "z"], "formats": [root_coords.dtype] * 3}
    root_coords = root_coords.view(dtype)

    attributes = []

    for attr in input.attributes:

        try:
            matched = intersect(attr, ptcld)
        except UnregisteredAttrCompException:
            continue
        if matched is None:
            continue
        if isinstance(matched, Pt):
            matched = PointCloud(
                space_id=matched.space_id, coordinates=[matched.coordinate]
            )
        if isinstance(matched, PointCloud):
            matched_coord = np.array(matched.coordinates)
            matched_coord = matched_coord.view(dtype)
            coordidx_in_matched = np.in1d(root_coords, matched_coord)
            coordidx = np.argwhere(coordidx_in_matched)[:, 0]

            matched_profiles = profile[coordidx]
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
                np.array([means, std]).T, columns=["mean", "std"], index=LAYERS[1:-1]
            )
            attr = Tabular(
                extra={X_DATA: dataframe, X_BIGBRAIN_LAYERWISE_INTENSITY: True}
            )
            attributes.append(attr)

            for index in coordidx.tolist():
                _profile = profile[index]
                depth = np.arange(0.0, 1.0, 1.0 / (profile[index].shape[0]))

                tabular_attr = Tabular(
                    extra={
                        X_DATA: pd.DataFrame(_profile, index=depth),
                        X_BIGBRAIN_PROFILE_VERTEX_IDX: index,
                    }
                )
                layer_boundary = LayerBoundary(
                    extra={
                        X_PRECALCULATED_BOUNDARY_KEY: [
                            Polyline(
                                points=(Pt(coordinate=[value, 0, 0], space_id=None)),
                                space_id=None,
                            )
                            for value in boundary_depths[index].tolist()
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
def _get_all():
    thickness_url = f"{REPO}/{THICKNESSES_FILE_LEFT}"
    valid, boundary_depths = get_thickness(thickness_url)
    profile_url = f"{REPO}/{PROFILES_FILE_LEFT}"
    if not get_profile.check_call_in_cache(profile_url, valid):
        logger.info(
            "First request to BigBrain profiles. Preprocessing the data "
            "now. This may take a little."
        )
    profile = get_profile(profile_url, valid)

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
