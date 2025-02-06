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

from . import cortical_profile

from .. import anchor as _anchor
from ...commons import logger
from ...retrieval import requests

from skimage.draw import polygon
from skimage.transform import resize
import numpy as np
import pandas as pd

from io import BytesIO
from typing import Union, Tuple, Iterable


def cell_reader(bytes_buffer: bytes):
    return pd.read_csv(BytesIO(bytes_buffer[2:]), delimiter=" ", header=0).astype(
        {"layer": int, "label": int}
    )


def layer_reader(bytes_buffer: bytes):
    return pd.read_csv(BytesIO(bytes_buffer[2:]), delimiter=" ", header=0, index_col=0)


def poly_srt(poly):
    return poly[poly[:, 0].argsort(), :]


def poly_rev(poly):
    return poly[poly[:, 0].argsort()[::-1], :]


class PolyLine:
    """Simple polyline representation which allows equidistant sampling."""

    def __init__(self, pts):
        self.pts = pts
        self.lengths = [
            np.sqrt(np.sum((pts[i, :] - pts[i - 1, :]) ** 2))
            for i in range(1, pts.shape[0])
        ]

    def length(self):
        return sum(self.lengths)

    def sample(self, d: Union[Iterable[float], np.ndarray, float]):
        # if d is iterable, we assume a list of sample positions
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


class CellDensityProfile(
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

    _filter_attrs = cortical_profile.CorticalProfile._filter_attrs + ["location"]

    def __init__(
        self,
        section: int,
        patch: int,
        url: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        """
        Generate a cell density profile from a URL to a cloud folder
        formatted according to the structure used by Bludau/Dickscheid et al.
        """
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Segmented cell body density",
            unit="cells / 0.1mm3",
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )
        self._step = 0.01
        self._url = url
        self._cell_loader = requests.HttpRequest(url, cell_reader)
        self._layer_loader = requests.HttpRequest(
            url.replace("segments", "layerinfo"), layer_reader
        )
        self._density_image = None
        self._layer_mask = None
        self._depth_image = None
        self.section = section
        self.patch = patch

    @property
    def location(self):
        return self.anchor.location

    @property
    def shape(self):
        """(y,x)"""
        return tuple(np.ceil(self.cells[["y", "x"]].max()).astype("int"))

    def boundary_annotation(self, boundary: Tuple[int, int]) -> np.ndarray:
        """Returns the annotation of a specific layer boundary."""
        shape_y, shape_x = self.shape

        # start of image patch
        if boundary == (-1, 0):
            return np.array([[0, 0], [shape_x, 0]])

        # end of image patch
        if boundary == (7, 8):
            return np.array([[0, shape_y], [shape_x, shape_y]])

        # retrieve polygon
        basename = "{}_{}.json".format(
            *(self.LAYERS[layer] for layer in boundary)
        ).replace("0_I", "0")
        poly_url = self._url.replace("segments.txt", basename)
        poly = poly_srt(np.array(requests.HttpRequest(poly_url).get()["segments"]))

        # ensure full width and trim to the image shape
        poly[0, 0] = 0
        poly[poly[:, 0] > shape_x, 0] = shape_x
        poly[poly[:, 1] > shape_y, 1] = shape_y

        return poly

    def layer_annotation(self, layer: int) -> np.ndarray:
        return np.vstack(
            (
                self.boundary_annotation((layer - 1, layer)),
                poly_rev(self.boundary_annotation((layer, layer + 1))),
                self.boundary_annotation((layer - 1, layer))[0, :],
            )
        )

    @property
    def layer_mask(self) -> np.ndarray:
        """Generates a layer mask from boundary annotations."""
        if self._layer_mask is None:
            self._layer_mask = np.zeros(np.array(self.shape, dtype=int) + 1, dtype="int")
            for layer in range(1, 8):
                pl = self.layer_annotation(layer)
                X, Y = polygon(pl[:, 0], pl[:, 1])
                self._layer_mask[Y, X] = layer
        return self._layer_mask

    @property
    def depth_image(self) -> np.ndarray:
        """Cortical depth image from layer boundary polygons by equidistant sampling."""

        if self._depth_image is None:
            logger.info("Calculating cell densities from cell and layer data...")
            # compute equidistant cortical depth image from inner and outer contour
            scale = 0.1
            depth_arr = np.zeros(np.ceil(np.array(self.shape) * scale).astype("int") + 1)

            # determine sufficient stepwidth for profile sampling
            # to match downscaled image resolution
            vstep, hstep = 1.0 / np.array(depth_arr.shape) / 2.0
            vsteps = np.arange(0, 1 + vstep, vstep)
            hsteps = np.arange(0, 1 + hstep, hstep)

            # build straight profiles between outer and inner cortical boundary
            s0 = PolyLine(self.boundary_annotation((0, 1)) * scale).sample(hsteps)
            s1 = PolyLine(self.boundary_annotation((6, 7)) * scale).sample(hsteps)
            profiles = [PolyLine(_.reshape(2, 2)) for _ in np.hstack((s0, s1))]

            # write sample depths to their location in the depth image
            for prof in profiles:
                prof_samples_as_index = prof.sample(vsteps).astype("int")
                depth_arr[prof_samples_as_index[:, 1], prof_samples_as_index[:, 0]] = vsteps

            # fix wm region, account for rounding error
            XY = self.layer_annotation(7) * scale
            depth_arr[polygon(XY[:, 1] - 1, XY[:, 0])] = 1
            depth_arr[-1, :] = 1

            # rescale depth image to original patch size
            self._depth_image = resize(depth_arr, self.density_image.shape)

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
    def density_image(self) -> np.ndarray:
        if self._density_image is None:
            logger.debug("Computing density image for", self._url)
            # we integrate cell counts into 2D bins
            # of square shape with a fixed sidelength
            pixel_size_micron = 100
            counts, xedges, yedges = np.histogram2d(
                self.cells.y,
                self.cells.x,
                bins=np.round(np.array(self.shape) / pixel_size_micron).astype("int"),
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
    def cells(self) -> pd.DataFrame:
        return self._cell_loader.get()

    @property
    def layers(self) -> pd.DataFrame:
        return self._layer_loader.get()

    @property
    def _depths(self):
        return np.arange(0, 1, self._step) + self._step / 2

    @property
    def _values(self):
        # TODO: release a dataset update instead of on the fly computation
        densities = []
        delta = self._step / 2.0
        for d in self._depths:
            mask = (self.depth_image >= d - delta) & (self.depth_image < d + delta)
            if np.sum(mask) > 0:
                densities.append(self.density_image[mask].mean())
            else:
                densities.append(np.nan)
        return np.asanyarray(densities)
