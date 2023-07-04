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

from . import spatial
from . import cortical_profile

from .. import anchor as _anchor
from ...locations import PointSet
from ...commons import PolyLine, logger, siibra_tqdm
from ...retrieval import requests

from skimage.draw import polygon
from skimage.transform import resize
from io import BytesIO
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union


class CellDensityProfile(
    spatial.PointCloud,
    cortical_profile.CorticalProfile,
    configuration_folder="features/tabular/corticalprofiles/celldensity",
    category='cellular'
):

    DESCRIPTION = (
        "Cortical profile of estimated densities of detected cell bodies (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR = 1.931

    @classmethod
    def _CELL_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0).astype(
            {"layer": int, "label": int}
        )

    @classmethod
    def _LAYER_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0, index_col=0)

    @staticmethod
    def _poly_srt(poly: np.ndarray) -> np.ndarray:
        return poly[poly[:, 0].argsort(), :]

    @staticmethod
    def _poly_rev(poly: np.ndarray) -> np.ndarray:
        return poly[poly[:, 0].argsort()[::-1], :]

    def __init__(
        self,
        coords: Union[np.ndarray, List['tuple']],
        urls: List[str],
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        """
        Generate a cell density profile from a URL to a cloud folder
        formatted according to the structure used by Bludau/Dickscheid et al.
        """
        pointset = PointSet(coords, space="bigbrain")
        modality = "Segmented cell body density"
        spatial.PointCloud.__init__(
            self,
            description=self.DESCRIPTION,
            modality=modality,
            anchor=anchor,
            pointset=pointset,
            headers=[]
        )
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality=modality,
            unit="detected cells / 0.1mm3",
            anchor=anchor,
            datasets=datasets
        )
        self._step = 0.01
        self._urls = urls
        self._build_loaders()
        self._extract_patch_info_from_url()
        self._density_image = {}
        self._layer_mask = {}
        self._depth_image = {}
        self._boundary_positions = {}
        self._depths_cached = None

    def _build_loaders(self):
        self._cell_loaders = [
            requests.HttpRequest(url, self._CELL_READER) for url in self._urls
        ]
        self._layer_loaders = [
            requests.HttpRequest(
                url.replace("segments", "layerinfo"), self._LAYER_READER
            )
            for url in self._urls
        ]

    def _extract_patch_info_from_url(self):
        si = self._urls[0].index("segments")
        self.sections, self.patches = zip(
            *[url[si - 10:si - 1].split('/') for url in self._urls]
        )

    def shape(self, point: int = 0) -> Tuple[int, int]:
        return tuple(self.cells[point][["y", "x"]].max().astype("int") + 1)

    def boundary_annotation(self, boundary: Tuple[int, int], point: int = 0) -> np.ndarray:
        """Returns the annotation of a specific layer boundary."""
        y1, x1 = self.shape(point)

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
        url = self._urls[point].replace("segments.txt", basename)
        poly = self._poly_srt(np.array(requests.HttpRequest(url).get()["segments"]))

        # ensure full width
        poly[0, 0] = 0
        poly[-1, 0] = x1

        return poly

    def layer_annotation(self, layer: int, point: int = 0) -> np.ndarray:
        ba0 = self.boundary_annotation((layer - 1, layer), point)
        ba1 = self.boundary_annotation((layer, layer + 1), point)
        return np.vstack((ba0, self._poly_rev(ba1), ba0[0, :]))

    def layer_mask(self, point: int = 0) -> np.ndarray:
        """Generates a layer mask from boundary annotations."""
        if self._layer_mask.get(point) is None:
            self._layer_mask[point] = np.zeros(np.array(self.shape(point)).astype("int") + 1)
            for layer in range(1, 8):
                pl = self.layer_annotation(layer, point)
                X, Y = polygon(pl[:, 0], pl[:, 1])
                try:
                    self._layer_mask[point][Y, X] = layer
                except Exception:
                    self._layer_mask[point].resize((max(Y) + 1, self._layer_mask[point].shape[1]))
                    self._layer_mask[point][Y, X] = layer
        return self._layer_mask[point]

    def depth_image(self, point: int = 0) -> np.ndarray:
        """Cortical depth image from layer boundary polygons by equidistant sampling."""

        if self._depth_image.get(point) is None:
            # compute equidistant cortical depth image from inner and outer contour
            scale = 0.1
            D = np.zeros((np.array(self.density_image(point).shape) * scale).astype("int") + 1)

            # determine sufficient stepwidth for profile sampling
            # to match downscaled image resolution
            vstep, hstep = 1.0 / np.array(D.shape) / 2.0
            vsteps = np.arange(0, 1 + vstep, vstep)
            hsteps = np.arange(0, 1 + vstep, hstep)

            # build straight profiles between outer and inner cortical boundary
            s0 = PolyLine(self.boundary_annotation((0, 1), point) * scale).sample(hsteps)
            s1 = PolyLine(self.boundary_annotation((6, 7), point) * scale).sample(hsteps)
            profiles = [PolyLine(_.reshape(2, 2)) for _ in np.hstack((s0, s1))]

            # write sample depths to their location in the depth image
            for prof in profiles:
                XY = prof.sample(vsteps).astype("int")
                D[XY[:, 1], XY[:, 0]] = vsteps

            # fix wm region, account for rounding error
            XY = self.layer_annotation(7, point) * scale
            D[polygon(XY[:, 1] - 1, XY[:, 0])] = 1
            D[-1, :] = 1

            # rescale depth image to original patch size
            self._depth_image[point] = resize(D, self.density_image(point).shape)

        return self._depth_image[point]

    def set_boundary_positions(self, point: int = 0):
        if self._boundary_positions.get(point) is not None:
            return
        self._boundary_positions[point] = {}
        for b in self.BOUNDARIES:
            XY = self.boundary_annotation(b, point).astype("int")
            self._boundary_positions[point][b] = self.depth_image(point)[
                XY[:, 1], XY[:, 0]
            ].mean()

    @property
    def boundary_positions(self) -> List[Dict[Tuple[int, int], float]]:
        for point in range(len(self)):
            if self._boundary_positions.get(point) is None:
                self.set_boundary_positions(point)
        return list(self._boundary_positions.values())

    def density_image(self, point: int = 0) -> np.ndarray:
        if self._density_image.get(point) is None:
            logger.debug("Computing density image for", self._urls)
            # we integrate cell counts into 2D bins
            # of square shape with a fixed sidelength
            pixel_size_micron = 100
            bins = (np.array(self.layer_mask(point).shape) / pixel_size_micron + 0.5).astype("int")
            counts, xedges, yedges = np.histogram2d(
                self.cells[point].y,
                self.cells[point].x,
                bins=bins
            )

            # rescale the counts from count / pixel_size**2  to count / 0.1mm^3,
            # assuming 20 micron section thickness.
            counts = counts / pixel_size_micron ** 2 / 20 * 100 ** 3

            # apply the Bigbrain shrinkage factor
            # TODO The planar correction factor was used for the datasets, but should
            # clarify if the full volumetric correction factor is not more correct.
            counts /= np.cbrt(self.BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR) ** 2

            # to go to 0.1 millimeter cube, we multiply by 0.1 / 0.0002 = 500
            self._density_image[point] = resize(counts, self.layer_mask(point).shape, order=2)

        return self._density_image[point]

    @property
    def cells(self) -> List[pd.DataFrame]:
        return [loader.get() for loader in self._cell_loaders]

    @property
    def layers(self) -> List[pd.DataFrame]:
        return [loader.get() for loader in self._layer_loaders]

    @property
    def _depths(self) -> np.float64:
        if self._depths_cached is None:
            self._depths_cached = np.arange(self._step / 2., 1., self._step)
            self._headers = self._depths_cached
        return self._depths_cached

    @property
    def _values(self) -> List[np.float64]:
        if self._values_cached is None:
            self._values_cached = []
            delta = self._step / 2.0
            for i in siibra_tqdm(
                range(len(self)),
                unit="Coordinate",
                disable=(len(self) == 1)
            ):
                densities = []
                for d in self._depths:
                    mask = (self.depth_image(i) >= d - delta) & (self.depth_image(i) < d + delta)
                    if np.sum(mask) > 0:
                        densities.append(self.density_image(i)[mask].mean())
                    else:
                        densities.append(np.NaN)
                self._values_cached.append(densities)
        return self._values_cached
