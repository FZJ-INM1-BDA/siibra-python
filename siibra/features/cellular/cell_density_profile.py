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

from .. import anchor as _anchor
from ..basetypes import cortical_profile

from ...commons import PolyLine, logger, create_key
from ...retrieval import requests

from skimage.draw import polygon
from skimage.transform import resize
from io import BytesIO
import numpy as np
import pandas as pd


class CellDensityProfile(cortical_profile.CorticalProfile, configuration_folder="features/profiles/celldensity"):

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
    def CELL_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0).astype(
            {"layer": int, "label": int}
        )

    @classmethod
    def LAYER_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0, index_col=0)

    @staticmethod
    def poly_srt(poly):
        return poly[poly[:, 0].argsort(), :]

    @staticmethod
    def poly_rev(poly):
        return poly[poly[:, 0].argsort()[::-1], :]

    def __init__(
        self,
        section: int,
        patch: int,
        url: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        """
        Generate a cell density profile from a URL to a cloud folder
        formatted according to the structure used by Bludau/Dickscheid et al.
        """
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Segmented cell body density",
            unit="detected cells / 0.1mm3",
            anchor=anchor,
            datasets=datasets,
        )
        self._step = 0.01
        self._url = url
        self._cell_loader = requests.HttpRequest(url, self.CELL_READER)
        self._layer_loader = requests.HttpRequest(
            url.replace("segments", "layerinfo"), self.LAYER_READER
        )
        self._density_image = None
        self._layer_mask = None
        self._depth_image = None
        self.section = section
        self.patch = patch

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
        poly = self.poly_srt(np.array(requests.HttpRequest(url).get()["segments"]))

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

    @property
    def key(self):
        assert len(self.species) == 1
        return create_key("{}_{}_{}_{}_{}".format(
            self.id,
            self.species[0]['name'],
            self.regionspec,
            self.section,
            self.patch
        ))
