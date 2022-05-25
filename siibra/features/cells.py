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
from .feature import RegionalFeature
from .query import FeatureQuery

from ..commons import logger
from ..core import Dataset
from ..core.space import Space, Point
from ..core.datasets import DatasetJsonModel
from ..retrieval.repositories import GitlabConnector, OwncloudConnector
from ..openminds.base import ConfigBaseModel

import numpy as np
import os
import importlib
import pandas as pd
import nibabel as nib
from pydantic import Field

class CorticalCellModel(ConfigBaseModel):
    x: float
    y: float
    area: float
    layer: int
    instance_label: int = Field(..., alias="instance label")


class CorticalCellDistributionModel(DatasetJsonModel):
    id: str = Field(..., alias="@id")
    type: str = Field('siibra/features/cells', const=True, alias="@type")
    cells: Optional[List[CorticalCellModel]]
    section: Optional[str]
    patch: Optional[str]


class CorticalCellDistribution(RegionalFeature, Dataset):
    """
    Represents a cortical cell distribution dataset.
    Implements lazy and cached loading of actual data.
    """

    def __init__(self, regionspec: str, cells: pd.DataFrame, connector, folder, species):

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
        data =  np.array(
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
        """ Compute average density of the patch in cells / mm^2. """
        num_cells = self.cells[
            (self.cells['layer'] > 0) & (self.cells['layer'] < 7)
        ].shape[0]
        area = self.layers['area (micron^2)'].sum()
        return num_cells / area * 1e3**2

    def layer_density(self, layerindex: int):
        """ Compute density of segmented cells for given layer in cells / mm^2. """
        assert layerindex in range(1, 7)
        num_cells = self.cells[self.cells['layer']==layerindex].shape[0]
        area = self.layers[self.layers['index']==layerindex].iloc[0]['area (micron^2)'] 
        return num_cells / area * 1e3**2

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
        return nib.Nifti1Image(
            self._mask_loader.data,
            self.image.affine
        )


    @property
    def location(self):
        """
        Location of this image patch in BigBrain histological space in mm.
        """
        center = np.r_[np.array(self.image.shape) // 2, 1]
        return Point(
            np.dot(self.image.affine, center)[:3], 
            Space.REGISTRY.BIG_BRAIN,
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
            self.cells['x'],
            self.cells['y'],
            s=np.sqrt(self.cells['area']),
            c=self.cells['layer'])
        ax2.axis("off")
        view = plotting.plot_img(
            tpl, cut_coords=tuple(self.location), cmap="gray", axes=ax3, display_mode="tiled"
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
        super_model['@type'] = CorticalCellDistribution.get_model_type()
        super_model['@id'] = self.model_id
        extra = {}
        if detail:
            d = self.cells.to_dict('index')
            extra['cells'] = [ value for value in d.values() ]
            extra['section'] = self.section
            extra['patch'] = self.patch

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

        species = {
            '@id': 'https://nexus.humanbrainproject.org/v0/data/minds/core/species/v1.0.0/0ea4e6ba-2681-4f7d-9fa9-49b915caaac9', 
            'name': 'Homo sapiens'
        }
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
                ]
            )
            self.register(
                CorticalCellDistribution(regionspec, pd.DataFrame(cells), self._SCIEBO, region_folder, species=species)
            )
