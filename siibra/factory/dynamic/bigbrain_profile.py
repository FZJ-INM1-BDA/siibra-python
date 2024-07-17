import numpy as np
import pandas as pd

from ...assignment import filter_by_query_param
from ...concepts import Feature, QueryParamCollection
from ...attributes.descriptions import Modality, register_modalities
from ...attributes.dataitems.tabular import Tabular, X_DATA
from ...attributes.locations.layerboundary import (
    LayerBoundary,
    X_PRECALCULATED_BOUNDARY_KEY,
    LAYERS,
)
from ...attributes.locations import intersect, PointCloud, Pt, Polyline
from ...exceptions import UnregisteredAttrCompException, InvalidAttrCompException
from ...retrieval_new.api_fetcher.bigbrain_profile import get_all

modality_of_interest = Modality(value="Modified silver staining")


X_BIGBRAIN_PROFILE_VERTEX_IDX = "x-siibra/bigbrainprofile/vertex-idx"
X_BIGBRAIN_LAYERWISE_INTENSITY = "x-siibra/bigbrainprofile/layerwiseintensity"


@register_modalities()
def add_modified_silver_staining():
    yield modality_of_interest


@filter_by_query_param.register(Feature)
def query_bigbrain_profile(input: QueryParamCollection):
    mods = [mod for cri in input.criteria for mod in cri._find(Modality)]
    if modality_of_interest not in mods:
        return []
    valid, boundary_depths, profile, vertices = get_all()
    ptcld = PointCloud(
        space_id="minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588",
        coordinates=vertices.tolist(),
    )
    root_coords = np.array(ptcld.coordinates)
    dtype = {"names": ["x", "y", "z"], "formats": [root_coords.dtype] * 3}
    root_coords = root_coords.view(dtype)

    attributes = []

    input_attrs = [attr for cri in input.criteria for attr in cri.attributes]

    for attr in input_attrs:

        try:
            matched = intersect(attr, ptcld)
        except UnregisteredAttrCompException:
            continue
        except InvalidAttrCompException:
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
