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


from typing import List, Optional
from .feature import RegionalFeature, CorticalProfile, RegionalFingerprint
from .query import FeatureQuery

from ..registry import REGISTRY, Preconfigure
from ..commons import logger, QUIET, create_key
from ..core import Dataset
from ..core.space import Point
from ..core.datasets import DatasetJsonModel, EbrainsDataset
from ..core.atlas import Atlas
from ..retrieval import HttpRequest, GitlabConnector, OwncloudConnector, EbrainsKgQuery, SiibraHttpRequestError
from ..openminds.base import ConfigBaseModel

import numpy as np
import os
import importlib
import pandas as pd
import nibabel as nib
from pydantic import Field
from io import BytesIO
from skimage.transform import resize
from skimage.draw import polygon


class CorticalCellModel(ConfigBaseModel):
    x: float
    y: float
    area: float
    layer: int
    instance_label: int = Field(..., alias="instance label")


class CorticalCellDistributionModel(DatasetJsonModel):
    id: str = Field(..., alias="@id")
    type: str = Field("siibra/features/cells", const=True, alias="@type")
    cells: Optional[List[CorticalCellModel]]
    section: Optional[str]
    patch: Optional[str]


class CorticalCellDistribution(RegionalFeature, Dataset):
    """
    Represents a cortical cell distribution dataset.
    Implements lazy and cached loading of actual data.
    """

    def __init__(
        self, regionspec: str, cells: pd.DataFrame, connector, folder, species
    ):

        _, section_id, patch_id = folder.split("/")
        RegionalFeature.__init__(self, regionspec, species=species)
        Dataset.__init__(self, identifier="")
        self.cells = cells
        self.section = section_id
        self.patch = patch_id

        # construct lazy data loaders
        self._info_loader = connector.get_loader(
            "info.txt", folder, CorticalCellDistribution._decode_infotxt
        )
        self._image_loader = connector.get_loader("image.nii.gz", folder)
        self._mask_loader = connector.get_loader("layermask.png", folder)
        self._layerinfo_loader = connector.get_loader(
            "layerinfo.txt", folder, CorticalCellDistribution._decode_layerinfo
        )
        self._connector = connector

    @staticmethod
    def _decode_infotxt(b):
        return dict(_.split(" ") for _ in b.decode("utf8").strip().split("\n"))

    @staticmethod
    def _decode_layerinfo(b):
        data = np.array(
            [tuple(_.split(" ")) for _ in b.decode().strip().split("\n")[1:]],
            dtype=[
                ("index", "i"),
                ("label", "U10"),
                ("area (micron^2)", "f"),
                ("avg. thickness (micron)", "f"),
            ],
        )
        return pd.DataFrame(data)

    def average_density(self):
        """Compute average density of the patch in cells / mm^2."""
        num_cells = self.cells[
            (self.cells["layer"] > 0) & (self.cells["layer"] < 7)
        ].shape[0]
        area = self.layers["area (micron^2)"].sum()
        return num_cells / area * 1e3 ** 2

    def layer_density(self, layerindex: int):
        """Compute density of segmented cells for given layer in cells / mm^2."""
        assert layerindex in range(1, 7)
        num_cells = self.cells[self.cells["layer"] == layerindex].shape[0]
        area = self.layers[self.layers["index"] == layerindex].iloc[0][
            "area (micron^2)"
        ]
        return num_cells / area * 1e3 ** 2

    def load_segmentations(self):
        from PIL import Image
        from io import BytesIO

        def imgdecoder(b):
            return np.array(Image.open(BytesIO(b)))

        index = 0
        result = []
        logger.info(f"Loading cell instance segmentation masks for {self}...")
        while True:
            try:
                target = f"segments_cpn_{index:02d}.tif"

                loader = self._connector.get_loader(target, self.folder, imgdecoder)
                result.append(loader.data)
                index += 1
            except RuntimeError:
                break
        return result

    @property
    def name(self):
        return f"Cortical cell distribution for {self.info.get('brain_area')}"

    @property
    def info(self):
        return self._info_loader.data

    @property
    def layers(self):
        """
        6x4 array of cortical layer attributes:
            Number Name Area(micron**2) AvgThickness(micron)
        """
        return self._layerinfo_loader.data

    @property
    def image(self):
        """
        Nifti1Image representation of the original image patch,
        with an affine matching it to the histological BigBrain space.
        """
        return self._image_loader.data

    @property
    def layermask(self):
        """
        Nifti1Image representation of the layer labeling.
        """
        return nib.Nifti1Image(self._mask_loader.data, self.image.affine)

    @property
    def location(self):
        """
        Location of this image patch in BigBrain histological space in mm.
        """
        center = np.r_[np.array(self.image.shape) // 2, 1]
        return Point(
            np.dot(self.image.affine, center)[:3],
            REGISTRY.Space["BIG_BRAIN"],
        )

    def __str__(self):
        return f"BigBrain cortical cell distribution in {self.regionspec} (section {self.info['section_id']}, patch {self.info['patch_id']})"

    def plot(self, title=None):
        """
        Create & return a matplotlib figure illustrating the patch,
        detected cells, and location in BigBrain space.
        """
        for pkg in ["matplotlib", "nilearn"]:
            if importlib.util.find_spec(pkg) is None:
                logger.warning(f"{pkg} not available. Plotting disabled.")
                return None

        from matplotlib import pyplot
        from nilearn import plotting

        patch = self.image.get_fdata()
        tpl = self.location.space.get_template().fetch()
        fig = pyplot.figure(figsize=(12, 6))
        pyplot.suptitle(str(self))
        ax1 = pyplot.subplot2grid((1, 4), (0, 0))
        ax2 = pyplot.subplot2grid((1, 4), (0, 1), sharex=ax1, sharey=ax1)
        ax3 = pyplot.subplot2grid((1, 4), (0, 2), colspan=2)
        ax1.imshow(patch, cmap="gray")
        ax2.axis("off")
        ax2.imshow(patch, cmap="gray")
        ax2.scatter(
            self.cells["x"],
            self.cells["y"],
            s=np.sqrt(self.cells["area"]),
            c=self.cells["layer"],
        )
        ax2.axis("off")
        view = plotting.plot_img(
            tpl,
            cut_coords=tuple(self.location),
            cmap="gray",
            axes=ax3,
            display_mode="tiled",
        )
        view.add_markers([tuple(self.location)])
        return fig

    @property
    def model_id(self):
        return f"{CorticalCellDistribution.get_model_type()}/{super().model_id}"

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/cells"

    def to_model(self, detail=False, **kwargs) -> CorticalCellDistributionModel:
        super_model = super().to_model(detail=detail, **kwargs).dict()
        super_model["@type"] = CorticalCellDistribution.get_model_type()
        super_model["@id"] = self.model_id
        extra = {}
        if detail:
            d = self.cells.to_dict("index")
            extra["cells"] = [value for value in d.values()]
            extra["section"] = self.section
            extra["patch"] = self.patch

        return CorticalCellDistributionModel(
            **extra,
            **super_model,
        )


class RegionalCellDensityExtractor(FeatureQuery):

    _FEATURETYPE = CorticalCellDistribution
    _JUGIT = GitlabConnector("https://jugit.fz-juelich.de", 4790, "v1.0a1")
    _SCIEBO = OwncloudConnector("https://fz-juelich.sciebo.de", "yDZfhxlXj6YW7KO")

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        logger.warning(
            f"PREVIEW DATA! {self._FEATURETYPE.__name__} data is only a pre-release snapshot. Contact support@ebrains.eu if you intend to use this data."
        )

        species = Atlas.get_species_data("human").dict()
        for cellfile, loader in self._JUGIT.get_loaders(
            suffix="segments.txt", recursive=True, decode_func=lambda b: b.decode()
        ):
            region_folder = os.path.dirname(cellfile)
            regionspec = " ".join(region_folder.split(os.path.sep)[0].split("_")[1:])
            cells = np.array(
                [
                    tuple(float(w) for w in _.strip().split(" "))
                    for _ in loader.data.strip().split("\n")[1:]
                ],
                dtype=[
                    ("x", "f"),
                    ("y", "f"),
                    ("area", "f"),
                    ("layer", "<i8"),
                    ("instance label", "<i"),
                ],
            )
            self.register(
                CorticalCellDistribution(
                    regionspec,
                    pd.DataFrame(cells),
                    self._SCIEBO,
                    region_folder,
                    species=species,
                )
            )


class PolyLine:
    """Simple polyline representation which allows equidistant sampling.."""

    def __init__(self, pts):
        self.pts = pts
        self.lengths = [
            np.sqrt(np.sum((pts[i, :] - pts[i - 1, :]) ** 2))
            for i in range(1, pts.shape[0])
        ]

    def length(self):
        return sum(self.lengths)

    def sample(self, d):

        # if d is interable, we assume a list of sample positions
        try:
            iter(d)
        except TypeError:
            positions = [d]
        else:
            positions = d

        samples = []
        for s_ in positions:
            s = min(max(s_, 0), 1)
            target_distance = s * self.length()
            current_distance = 0
            for i, length in enumerate(self.lengths):
                current_distance += length
                if current_distance >= target_distance:
                    p1 = self.pts[i, :]
                    p2 = self.pts[i + 1, :]
                    r = (target_distance - current_distance + length) / length
                    samples.append(p1 + (p2 - p1) * r)
                    break

        if len(samples) == 1:
            return samples[0]
        else:
            return np.array(samples)


def CELL_READER(b):
    return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0).astype(
        {"layer": int, "label": int}
    )


def LAYER_READER(b):
    return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0, index_col=0)


class CellDensityProfile(CorticalProfile, EbrainsDataset):

    DESCRIPTION = (
        "Cortical profile of estimated densities of detected cell bodies (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR = 1.931

    @staticmethod
    def poly_srt(poly):
        return poly[poly[:, 0].argsort(), :]

    @staticmethod
    def poly_rev(poly):
        return poly[poly[:, 0].argsort()[::-1], :]

    def __init__(
        self,
        dataset_id: str,
        species: dict,
        regionname: str,
        url: str,
    ):
        """Generate a receptor density profile from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        EbrainsDataset.__init__(
            self, dataset_id, f"Cell density profile for {regionname}"
        )

        self._step = 0.01
        self._url = url
        self._cell_loader = HttpRequest(url, CELL_READER)
        self._layer_loader = HttpRequest(
            url.replace("segments", "layerinfo"), LAYER_READER
        )
        self._density_image = None
        self._layer_mask = None
        self._depth_image = None

        CorticalProfile.__init__(
            self,
            measuretype="cell density",
            species=species,
            regionname=regionname,
            description=self.DESCRIPTION,
            unit="detected cells / 0.1mm3",
        )

    @property
    def shape(self):
        return tuple(self.cells[["y", "x"]].max().astype("int") + 1)

    def boundary_annotation(self, boundary):
        """Returns the annotation of a specific layer boundary."""
        y1, x1 = self.shape

        # start of image patch
        if boundary == (-1, 0):
            return np.array([[0, 0], [x1, 0]])

        # end of image patch
        if boundary == (7, 8):
            return np.array([[0, y1], [x1, y1]])

        # retrieve polygon
        basename = "{}_{}.json".format(
            *(self.LAYERS[layer] for layer in boundary)
        ).replace("0_I", "0")
        url = self._url.replace("segments.txt", basename)
        poly = self.poly_srt(np.array(HttpRequest(url).get()["segments"]))

        # ensure full width
        poly[0, 0] = 0
        poly[-1, 0] = x1

        return poly

    def layer_annotation(self, layer):
        return np.vstack(
            (
                self.boundary_annotation((layer - 1, layer)),
                self.poly_rev(self.boundary_annotation((layer, layer + 1))),
                self.boundary_annotation((layer - 1, layer))[0, :],
            )
        )

    @property
    def layer_mask(self):
        """Generates a layer mask from boundary annotations."""
        if self._layer_mask is None:
            self._layer_mask = np.zeros(np.array(self.shape).astype("int") + 1)
            for layer in range(1, 8):
                pl = self.layer_annotation(layer)
                X, Y = polygon(pl[:, 0], pl[:, 1])
                self._layer_mask[Y, X] = layer
        return self._layer_mask

    @property
    def depth_image(self):
        """Cortical depth image from layer boundary polygons by equidistant sampling."""

        if self._depth_image is None:

            # compute equidistant cortical depth image from inner and outer contour
            scale = 0.1
            D = np.zeros((np.array(self.density_image.shape) * scale).astype("int") + 1)

            # determine sufficient stepwidth for profile sampling
            # to match downscaled image resolution
            vstep, hstep = 1.0 / np.array(D.shape) / 2.0
            vsteps = np.arange(0, 1 + vstep, vstep)
            hsteps = np.arange(0, 1 + vstep, hstep)

            # build straight profiles between outer and inner cortical boundary
            s0 = PolyLine(self.boundary_annotation((0, 1)) * scale).sample(hsteps)
            s1 = PolyLine(self.boundary_annotation((6, 7)) * scale).sample(hsteps)
            profiles = [PolyLine(_.reshape(2, 2)) for _ in np.hstack((s0, s1))]

            # write sample depths to their location in the depth image
            for prof in profiles:
                XY = prof.sample(vsteps).astype("int")
                D[XY[:, 1], XY[:, 0]] = vsteps

            # fix wm region, account for rounding error
            XY = self.layer_annotation(7) * scale
            D[polygon(XY[:, 1] - 1, XY[:, 0])] = 1
            D[-1, :] = 1

            # rescale depth image to original patch size
            self._depth_image = resize(D, self.density_image.shape)

        return self._depth_image

    @property
    def boundary_positions(self):
        if self._boundary_positions is None:
            self._boundary_positions = {}
            for b in self.BOUNDARIES:
                XY = self.boundary_annotation(b).astype("int")
                self._boundary_positions[b] = self.depth_image[
                    XY[:, 1], XY[:, 0]
                ].mean()
        return self._boundary_positions

    @property
    def density_image(self):
        if self._density_image is None:
            logger.debug("Computing density image for", self._url)
            # we integrate cell counts into 2D bins
            # of square shape with a fixed sidelength
            pixel_size_micron = 100
            counts, xedges, yedges = np.histogram2d(
                self.cells.y,
                self.cells.x,
                bins=(np.array(self.layer_mask.shape) / pixel_size_micron + 0.5).astype(
                    "int"
                ),
            )

            # rescale the counts from count / pixel_size**2  to count / 0.1mm^3,
            # assuming 20 micron section thickness.
            counts = counts / pixel_size_micron ** 2 / 20 * 100 ** 3

            # apply the Bigbrain shrinkage factor
            # TODO The planar correction factor was used for the datasets, but should
            # clarify if the full volumetric correction factor is not more correct.
            counts /= np.cbrt(self.BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR) ** 2

            # to go to 0.1 millimeter cube, we multiply by 0.1 / 0.0002 = 500
            self._density_image = resize(counts, self.layer_mask.shape, order=2)

        return self._density_image

    @property
    def cells(self):
        return self._cell_loader.get()

    @property
    def layers(self):
        return self._layer_loader.get()

    @property
    def _depths(self):
        return [d + self._step / 2 for d in np.arange(0, 1, self._step)]

    @property
    def _values(self):
        densities = []
        delta = self._step / 2.0
        for d in self._depths:
            mask = (self.depth_image >= d - delta) & (self.depth_image < d + delta)
            if np.sum(mask) > 0:
                densities.append(self.density_image[mask].mean())
            else:
                densities.append(np.NaN)
        return densities


class CellDensityProfileQuery(FeatureQuery):

    _FEATURETYPE = CellDensityProfile

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)

        query_result = EbrainsKgQuery(
            query_id="siibra-cell-densities-0",
            params={"vocab": "https://schema.hbp.eu/myQuery/"},
        ).get()

        for result in query_result.get("results", []):
            dataset_id = result["@id"].split("/")[-1]
            segment_files = [
                fileinfo["url"]
                for fileinfo in result["files"]
                if fileinfo["name"] == "segments.txt"
            ]
            for regionspec in result["regionspec"]:
                for fileurl in segment_files:
                    with QUIET:
                        f = CellDensityProfile(
                            dataset_id,
                            Atlas.get_species_data("human").dict(),
                            regionspec,
                            fileurl,
                        )
                    self.register(f)


@Preconfigure("features/fingerprints/celldensity")
class CellDensityFingerprint(RegionalFingerprint):

    DESCRIPTION = (
        "Layerwise estimated densities of detected cell bodies  (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    def __init__(
        self,
        species: dict,
        regionname: str,
        segmentfiles: list,
        layerfiles: list,
        dataset_id: str = None,
    ):
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None
        self.dataset_id = dataset_id
        RegionalFingerprint.__init__(
            self,
            measuretype="Layerwise cell density",
            species=species,
            regionname=regionname,
            description=self.DESCRIPTION,
            unit="detected cells / 0.1mm3",
        )

    @property
    def densities(self):
        if self._densities is None:
            density_dict = {}
            for i, (cellfile, layerfile) in enumerate(self._filepairs):
                try:
                    cells = HttpRequest(cellfile, func=CELL_READER).data
                    layers = HttpRequest(layerfile, func=LAYER_READER).data
                except SiibraHttpRequestError as e:
                    print(str(e))
                    logger.error(f"Skipping to bootstrap a {self.__class__.__name__} feature, cannot access file resource.")
                    continue
                counts = cells.layer.value_counts()
                areas = layers["Area(micron**2)"]
                density_dict[i] = counts[areas.index] / areas * 100 ** 2 * 5
            self._densities = pd.DataFrame(density_dict)
            self._densities.index.names = ["Layer"]
        return self._densities

    @property
    def _labels(self):
        return [CorticalProfile.LAYERS[_] for _ in self.densities.index]

    @property
    def _means(self):
        return self.densities.mean(axis=1).to_numpy()

    @property
    def _stds(self):
        return self.densities.std(axis=1).to_numpy()
        
    @property
    def key(self):
        assert len(self.species) == 1
        return create_key("{}_{}_{}".format(
            self.dataset_id,
            self.species[0]['name'],
            self.regionspec
        ))

    @classmethod
    def _from_json(cls, spec):
        assert spec.get('@type') == "siibra/resource/feature/fingerprint/celldensity/v1.0.0"
        return cls(
            species=spec['species'],
            regionname=spec['region_name'],
            segmentfiles=spec['segmentfiles'],
            layerfiles=spec['layerfiles'],
            dataset_id=spec['kgId']
        )


