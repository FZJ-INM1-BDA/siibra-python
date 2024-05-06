from . import attributes

from ..core.structure import AnatomicalStructure
from ..configuration import Configuration
from ..configuration.factory import build_type
from ..commons import KeyAccessor

from dataclasses import dataclass, field
from typing import List, Callable
import uuid
import numpy as np
import pandas as pd
from functools import wraps
from itertools import count


NAME_ATTRS = [
    "siibra/attr/meta/modality",
    "siibra/attr/meta/regionspec",
    "siibra/attr/meta/species",
]

data_filters: dict[str, Callable] = {}


def register_data_filter(*, keywords: list[str]):
    def outer(fn):

        for keyword in keywords:
            assert keyword not in data_filters, f"{keyword} already registered!"
            data_filters[keyword] = fn

        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)

        return inner

    return outer


def get_data_filter(keyword: str) -> Callable:
    if keyword in data_filters:
        return data_filters[keyword]
    raise NotImplementedError


@dataclass
class DataFeature:
    """A multimodal data feature characterized by a set of attributes."""

    schema = "siibra/feature/v0.2"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: List["attributes.Attribute"] = field(default_factory=list)
    name: str = field(default=None)

    def __post_init__(self):
        # Construct nested FeatureAttribute objects from their specs
        # This can also be achieved with MetaClass, but IMO, using post_init is more explicit
        parsed_attrs: List["attributes.Attribute"] = []
        for att in self.attributes:

            if isinstance(att, DataFeature):
                raise RuntimeError("feature attributes cannot be of type DataFeature")

            if not isinstance(att, dict):
                raise RuntimeError(
                    f"Expecting a dictionary as feature attribute specification, not '{type(att)}'"
                )

            parsed_attrs.extend(attributes.Attribute.from_dict(att))

        self.attributes = parsed_attrs

        if self.name is None:
            parts = [a.name for a in self.attributes if a.schema in NAME_ATTRS]
            self.name = ", ".join(parts) if len(parts) > 0 else "Unnamed"

    def matches(self, *args, **kwargs):
        """Returns true if this feature or one of its attributes match any of the given arguments.
        TODO One might prefer if this matches agains **all** instead of **any** arguments, but *any* is simpler at this stage.
        """
        return any(a.matches(*args, **kwargs) for a in self.attributes)

    def get_data(self, keyword: str = None):
        """Return a list data obtained from DataAttributes.
        TODO this is just for beta development, later on we might rather
        have properly typed methods to load any available data frames and images.
        """

        if keyword is None:
            yield from (
                attr.data
                for attr in self.attributes
                if isinstance(attr, attributes.DataAttribute)
            )
        try:
            fn = get_data_filter(keyword=keyword)
            yield fn(self)
        except NotImplementedError:
            return

    def get_modalities(self):
        return [
            attr.name
            for attr in self.attributes
            if isinstance(attr, attributes.ModalityAttribute)
        ]

    def plot(self, *args, **kwargs):
        """Plots all data attributes."""
        for attr in self.attributes:
            if isinstance(attr, attributes.DataAttribute):
                yield attr.plot(*args, **kwargs)

    @property
    def data_modalities(self):
        accessor = KeyAccessor()
        for name, fn in data_filters.items():
            try:
                fn(self, dry_run=True)
                accessor.register(name)
            except Exception:
                continue
        return accessor


def get(structure: AnatomicalStructure, modality: str, **kwargs):
    """Query all features of a certain modality with an anatomical structure."""
    cfg = Configuration()
    return list(
        filter(
            lambda f: f.matches(modality=modality)
            and f.matches(
                structure, **kwargs
            ),  # Ideally enforce only keyword arguement
            (DataFeature(**s) for _, s in cfg.specs.get("siibra/feature/v0.2")),
        )
    )


@build_type("siibra/feature/v0.2")
def build_feature(cls, spec):
    spec.pop('filename', None)
    spec.pop("@type", None)
    return DataFeature(**spec)


@register_data_filter(keywords=["layer mask"])
def get_layer_mask(feature: DataFeature, *, dry_run=False):
    from skimage.draw import polygon

    cells = [
        attr
        for attr in feature.attributes
        if (
            isinstance(attr, attributes.TabularDataAttribute)
            and (
                attr.extra.get("x-siibra/celldensity")
                or attr.extra.get("x-siibra/celldensityprofile")
            )
            == "segments"
        )
    ]

    layer_boundaries = [
        attr
        for attr in feature.attributes
        if isinstance(attr, attributes.LayerBoundaryDataAttribute)
    ]

    assert (
        len(cells) == 1
    ), f"Expected one and only one segment attribute, but got {len(cells)}"
    assert (
        len(layer_boundaries) == 1
    ), f"Expected one and only one layer boundary attribute, but got {len(layer_boundaries)}"

    if dry_run:
        return None

    (cell,) = cells
    (layer_boundary,) = layer_boundaries

    shape = tuple(cell.data[["y", "x"]].max().astype("int") + 1)

    layer_mask = np.zeros(np.array(np.array(shape).astype("int") + 1))
    y1, x1 = shape

    layer_annotations: list[list[list[float]]] = [
        [[0, 0], [x1, 0]],
        *[layer.coordinates for layer in layer_boundary.layers],
        [[0, y1], [x1, y1]],
    ]

    paired_layers = zip(
        count(),
        layer_annotations[:-1],
        layer_annotations[1:],
    )
    for layer, start, end in paired_layers:

        start = np.array(start)
        end = np.array(end)

        start[0, 0] = 0
        start[-1, 0] = x1
        end[0, 0] = 0
        end[-1, 0] = x1

        end = end[end[:, 0].argsort()[::-1], :]

        pl = np.vstack((start, end, start[0, :]))
        X, Y = polygon(pl[:, 0], pl[:, 1])
        layer_mask[Y, X] = layer
    return layer_mask


@register_data_filter(keywords=["density image"])
def get_density_image(feature: DataFeature, *, dry_run=False):
    from skimage.transform import resize

    cells = [
        attr
        for attr in feature.attributes
        if (
            isinstance(attr, attributes.TabularDataAttribute)
            and (
                attr.extra.get("x-siibra/celldensity")
                or attr.extra.get("x-siibra/celldensityprofile")
            )
            == "segments"
        )
    ]
    layers = [
        attr
        for attr in feature.attributes
        if (
            isinstance(attr, attributes.TabularDataAttribute)
            and (
                attr.extra.get("x-siibra/celldensity")
                or attr.extra.get("x-siibra/celldensityprofile")
            )
            == "layerinfo"
        )
    ]
    assert (
        len(cells) == 1
    ), f"Expected one and only one segment attribute, but got {len(cells)}"
    assert (
        len(layers) == 1
    ), f"Expected one and only one layerinfo attribute, but got {len(layers)}"

    layer_mask = get_layer_mask(feature, dry_run=dry_run)

    if dry_run:
        return

    (cell,) = cells
    (layer,) = layers

    pixel_size_micron = 100
    counts, xedges, yedges = np.histogram2d(
        cell.data.y,
        cell.data.x,
        bins=(np.array(layer_mask.shape) / pixel_size_micron + 0.5).astype("int"),
    )

    counts = counts / pixel_size_micron**2 / 20 * 100**3

    counts /= (
        np.cbrt(attributes.data_attributes.BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR) ** 2
    )
    return resize(counts, layer_mask.shape, order=2)


@register_data_filter(keywords=["layer statistics"])
def get_layer_statistics(feature: DataFeature, *, dry_run=False):
    LAYER_NAMES = ("0", "I", "II", "III", "IV", "V", "VI", "WM")
    cell_attrs = [
        attr
        for attr in feature.attributes
        if (
            isinstance(attr, attributes.data_attributes.TabularDataAttribute)
            and attr.extra.get("x-siibra/celldensity") == "segments"
        )
    ]
    layer_attrs = [
        attr
        for attr in feature.attributes
        if (
            isinstance(attr, attributes.data_attributes.TabularDataAttribute)
            and attr.extra.get("x-siibra/celldensity") == "layerinfo"
        )
    ]

    def get_section_patch(attr: attributes.data_attributes.TabularDataAttribute):
        return attr.extra.get("x-siibra/celldensity/section"), attr.extra.get(
            "x-siibra/celldensity/patch"
        )

    paired_attrs: dict[
        tuple[str, str], tuple[attributes.data_attributes.TabularDataAttribute, ...]
    ] = {}

    for cell_attr in cell_attrs:
        key = get_section_patch(cell_attr)
        assert key not in paired_attrs, f"key {key} already in paired attrs"
        paired_attrs[key] = (cell_attr,)

    for layer_attr in layer_attrs:
        key = get_section_patch(layer_attr)
        assert (
            key in paired_attrs
        ), f"expectin {key=} to be in paired attrs, but is not yet"
        (cell_attr,) = paired_attrs[key]
        paired_attrs[key] = (cell_attr, layer_attr)

    if dry_run:
        return

    density_dict = {}
    for index, (cell, layer) in enumerate(paired_attrs.values()):
        counts = cell.data.layer.value_counts()
        areas = layer.data["Area(micron**2)"]
        indices = np.intersect1d(areas.index, counts.index)
        density_dict[index] = counts[indices] / areas * 100**2 * 5

    df = pd.DataFrame(density_dict)
    df = pd.DataFrame(
        np.array([df.mean(axis=1).tolist(), df.std(axis=1).tolist()]).T,
        columns=["mean", "std"],
        index=[LAYER_NAMES[int(idx)] for idx in df.index],
    )
    df.index.name = "layer"
    return df
