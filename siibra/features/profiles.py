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

from .feature import Feature

from ..commons import PolyLine, logger, create_key, decode_receptor_tsv
from ..locations import Point
from ..retrieval.requests import HttpRequest
from ..vocabularies import RECEPTOR_SYMBOLS

import numpy as np
import pandas as pd
from io import BytesIO
from typing import Union
from textwrap import wrap
from skimage.draw import polygon
from skimage.transform import resize


class CorticalProfile(Feature):
    """
    Represents a 1-dimensional profile of measurements along cortical depth,
    measured at relative depths between 0 representing the pial surface,
    and 1 corresponding to the gray/white matter boundary.

    Mandatory attributes are the list of depth coordinates and the list of
    corresponding measurement values, which have to be of equal length,
    as well as a unit and description of the measurements.

    Optionally, the depth coordinates of layer boundaries can be specified.

    Most attributes are modelled as properties, so dervide classes are able
    to implement lazy loading instead of direct initialiation.

    """

    LAYERS = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "WM"}
    BOUNDARIES = list(zip(list(LAYERS.keys())[:-1], list(LAYERS.keys())[1:]))

    def __init__(
        self,
        description: str,
        measuretype: str,
        anchor: "AnatomicalAnchor",
        depths: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        unit: str = None,
        boundary_positions: dict = None,
        datasets: list = []
    ):
        """Initialize profile.

        Args:
            description (str):
                Human-readable of the modality of the measurements.
            measuretype (str):
                Short textual description of the modaility of measurements
            anchor: AnatomicalAnchor
            depths (list, optional):
                List of cortical depthh positions corresponding to each
                measurement, all in the range [0..1].
                Defaults to None.
            values (list, optional):
                List of the actual measurements at each depth position.
                Length must correspond to 'depths'.
                Defaults to None.
            unit (str, optional):
                Textual identifier for the unit of measurements.
                Defaults to None.
            boundary_positions (dict, optional):
                Dictionary of depths at which layer boundaries were identified.
                Keys are tuples of layer numbers, e.g. (1,2), values are cortical
                depth positions in the range [0..1].
                Defaults to None.
            datasets : list
                list of datasets corresponding to this feature
        """
        Feature.__init__(self, measuretype=measuretype, description=description, anchor=anchor, datasets=datasets)

        # cached properties will be revealed as property functions,
        # so derived classes may choose to override for lazy loading.
        self._unit = unit
        self._depths_cached = depths
        self._values_cached = values
        self._boundary_positions = boundary_positions

    def _check_sanity(self):
        # check plausibility of the profile
        assert isinstance(self._depths, (list, np.ndarray))
        assert isinstance(self._values, (list, np.ndarray))
        assert len(self._values) == len(self._depths)
        assert all(0 <= d <= 1 for d in self._depths)
        if self.boundaries_mapped:
            assert all(0 <= d <= 1 for d in self.boundary_positions.values())
            assert all(
                layerpair in self.BOUNDARIES
                for layerpair in self.boundary_positions.keys()
            )

    @property
    def unit(self):
        """Optionally overridden in derived classes."""
        if self._unit is None:
            raise NotImplementedError(f"'unit' not set for {self.__class__.__name__}.")
        return self._unit

    @property
    def boundary_positions(self):
        if self._boundary_positions is None:
            return {}
        else:
            return self._boundary_positions

    def assign_layer(self, depth: float):
        """Compute the cortical layer for a given depth from the
        layer boundary positions. If no positions are available
        for this profile, return None."""
        assert 0 <= depth <= 1
        if len(self.boundary_positions) == 0:
            return None
        else:
            return max(
                [l2 for (l1, l2), d in self.boundary_positions.items() if d < depth]
            )

    @property
    def boundaries_mapped(self) -> bool:
        if self.boundary_positions is None:
            return False
        else:
            return all((b in self.boundary_positions) for b in self.BOUNDARIES)

    @property
    def _layers(self):
        """List of layers assigned to each measurments,
        if layer boundaries are available for this features.
        """
        if self.boundaries_mapped:
            return [self.assign_layer(d) for d in self._depths]
        else:
            return None

    @property
    def data(self):
        """Return a pandas Series representing the profile."""
        self._check_sanity()
        return pd.Series(
            self._values, index=self._depths, name=f"{self.measuretype} ({self.unit})"
        )

    def plot(self, **kwargs):
        """Plot the profile.
        Keyword arguments are passed on to the plot command.
        'layercolor' can be used to specify a color for cortical layer shading.
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        kwargs["title"] = kwargs.get("title", "\n".join(wrap(self.name, wrapwidth)))
        kwargs["xlabel"] = kwargs.get("xlabel", "Cortical depth")
        kwargs["ylabel"] = kwargs.get("ylabel", self.unit)
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["ylim"] = kwargs.get("ylim", (0, max(self._values)))
        layercolor = kwargs.pop("layercolor") if "layercolor" in kwargs else "black"
        axs = self.data.plot(**kwargs)

        if self.boundaries_mapped:
            bvals = list(self.boundary_positions.values())
            for i, (d1, d2) in enumerate(list(zip(bvals[:-1], bvals[1:]))):
                axs.text(
                    d1 + (d2 - d1) / 2.0,
                    10,
                    self.LAYERS[i + 1],
                    weight="normal",
                    ha="center",
                )
                if i % 2 == 0:
                    axs.axvspan(d1, d2, color=layercolor, alpha=0.1)

        axs.set_title(axs.get_title(), fontsize="medium")

        return axs

    @property
    def _depths(self):
        """Returns a list of the relative cortical depths of the measured values in the range [0..1].
        To be implemented in derived class."""
        if self._depths_cached is None:
            raise NotImplementedError(
                f"'_depths' not available for {self.__class__.__name__}."
            )
        return self._depths_cached

    @property
    def _values(self):
        """Returns a list of the measured values per depth.
        To be implemented in derived class."""
        if self._values_cached is None:
            raise NotImplementedError(
                f"'_values' not available for {self.__class__.__name__}."
            )
        return self._values_cached


class CellDensityProfile(CorticalProfile, configuration_folder="features/profiles/celldensity"):

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
        anchor: "AnatomicalAnchor",
        datasets: list = []
    ):
        """
        Generate a cell density profile from a URL to a cloud folder
        formatted according to the structure used by Bludau/Dickscheid et al.
        """
        CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            measuretype="cell density",
            unit="detected cells / 0.1mm3",
            anchor=anchor,
            datasets=datasets,
        )
        self._step = 0.01
        self._url = url
        self._cell_loader = HttpRequest(url, self.CELL_READER)
        self._layer_loader = HttpRequest(
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


class BigBrainIntensityProfile(CorticalProfile):

    DESCRIPTION = (
        "Cortical profiles of BigBrain staining intensities computed by Konrad Wagstyl, "
        "as described in the publication 'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678'."
        "Taken from the tutorial at https://github.com/kwagstyl/cortical_layers_tutorial "
        "and assigned to cytoarchitectonic regions of Julich-Brain."
    )

    def __init__(
        self,
        regionname: str,
        depths: list,
        values: list,
        boundaries: list,
        location: Point
    ):
        from .anchor import AnatomicalAnchor
        anchor = AnatomicalAnchor(
            location=location,
            region=regionname,
            species='Homo sapiens'
        )
        CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            measuretype="BigBrain cortical intensity profile",
            anchor=anchor,
            depths=depths,
            values=values,
            unit="staining intensity",
            boundary_positions={
                b: boundaries[b[0]]
                for b in CorticalProfile.BOUNDARIES
            }
        )
        self.location = location


class ReceptorDensityProfile(CorticalProfile, configuration_folder="features/profiles/receptor"):

    DESCRIPTION = (
        "Cortical profile of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The profiles provide, for a "
        "single tissue sample, an exemplary density distribution for a single receptor from the pial surface "
        "to the border between layer VI and the white matter."
    )

    def __init__(
        self,
        receptor: str,
        tsvfile: str,
        anchor: "AnatomicalAnchor",
        datasets: list = []
    ):
        """Generate a receptor density profile from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            measuretype=f"{receptor} receptor density",
            anchor=anchor,
            datasets=datasets,
        )
        self.type = receptor
        self._data_cached = None
        self._loader = HttpRequest(
            tsvfile,
            lambda url: self.parse_tsv_data(decode_receptor_tsv(url)),
        )
        self._unit_cached = None

    @property
    def key(self):
        return "{}_{}_{}_{}_{}".format(
            create_key(self.__class__.__name__),
            self.id,
            create_key(self.species_name),
            create_key(self.regionspec),
            create_key(self.type)
        )

    @property
    def receptor(self):
        return "{} ({})".format(
            self.type,
            RECEPTOR_SYMBOLS[self.type]['receptor']['name'],
        )

    @property
    def neurotransmitter(self):
        return "{} ({})".format(
            RECEPTOR_SYMBOLS[self.type]['neurotransmitter']['label'],
            RECEPTOR_SYMBOLS[self.type]['neurotransmitter']['name'],
        )

    @property
    def unit(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data["unit"]

    @property
    def _values(self):
        # triggers lazy loading of the HttpRequest
        return self._loader.data["density"]

    @property
    def _depths(self):
        return self._loader.data["depth"]

    @classmethod
    def parse_tsv_data(self, data):
        units = {list(v.values())[3] for v in data.values()}
        assert len(units) == 1
        return {
            "depth": [float(k) / 100.0 for k in data.keys() if k.isnumeric()],
            "density": [
                float(list(v.values())[2]) for k, v in data.items() if k.isnumeric()
            ],
            "unit": next(iter(units)),
        }
