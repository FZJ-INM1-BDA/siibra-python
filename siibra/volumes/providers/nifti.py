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

import os
from typing import Union, Dict, Tuple

import numpy as np
import nibabel as nib

from . import provider as _provider
from ...commons import logger, resample_img_to_img
from ...retrieval import requests
from ...locations import pointcloud, boundingbox as _boundingbox


class NiftiProvider(_provider.VolumeProvider, srctype="nii"):

    def __init__(self, src: Union[str, Dict[str, str], nib.Nifti1Image, Tuple[np.ndarray, np.ndarray]]):
        """
        Construct a new NIfTI volume source, from url, local file, or Nift1Image object.
        """
        _provider.VolumeProvider.__init__(self)

        self._init_url: Union[str, Dict[str, str]] = None

        def loader(url):
            if os.path.isfile(url):
                return lambda fn=url: nib.load(fn)
            else:
                req = requests.HttpRequest(url)
                return lambda req=req: req.data

        if isinstance(src, nib.Nifti1Image):
            self._img_loaders = {None: lambda img=src: img}
        elif isinstance(src, str):  # one single image to load
            self._img_loaders = {None: loader(src)}
        elif isinstance(src, dict):  # assuming multiple for fragment images
            self._img_loaders = {lbl: loader(url) for lbl, url in src.items()}
        elif isinstance(src, tuple):
            assert len(src) == 2
            assert all(isinstance(_, np.ndarray) for _ in src)
            self._img_loaders = {None: lambda data=src[0], affine=src[1]: nib.Nifti1Image(data, affine)}
        else:
            raise ValueError(f"Invalid source specification for {self.__class__}: {src}")
        if not isinstance(src, (nib.Nifti1Image, tuple)):
            self._init_url = src

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url

    @property
    def fragments(self):
        return [k for k in self._img_loaders if k is not None]

    def get_boundingbox(self, **fetch_kwargs) -> "_boundingbox.BoundingBox":
        """
        Return the bounding box in physical coordinates of the union of
        fragments in this nifti volume.

        Parameters
        ----------
        fetch_kwargs:
            Not used
        """
        bbox = None
        for loader in self._img_loaders.values():
            img = loader()
            if len(img.shape) > 3:
                logger.warning(
                    f"N-D NIfTI volume has shape {img.shape}, but "
                    f"bounding box considers only {img.shape[:3]}"
                )
            shape = img.shape[:3]
            next_bbox = _boundingbox.BoundingBox(
                (0, 0, 0), shape, space=None
            ).transform(img.affine)
            bbox = next_bbox if bbox is None else bbox.union(next_bbox)
        return bbox

    def _merge_fragments(self) -> nib.Nifti1Image:
        """
        Merge all fragments this volume contains into one Nifti1Image.
        """
        bbox = self.get_boundingbox(clip=False, background=0.0)
        num_conflicts = 0
        result = None
        for loader in self._img_loaders.values():
            img = loader()
            if result is None:
                # build the empty result image with its own affine and voxel space
                s0 = np.identity(4)
                s0[:3, -1] = list(bbox.minpoint.transform(np.linalg.inv(img.affine)))
                result_affine = np.dot(img.affine, s0)  # adjust global bounding box offset to get global affine
                voxdims = np.asanyarray(np.ceil(
                    bbox.transform(np.linalg.inv(result_affine)).shape
                ), dtype="int")
                result_arr = np.zeros(voxdims, dtype=img.dataobj.dtype)
                result = nib.Nifti1Image(dataobj=result_arr, affine=result_affine)

            # resample to merge template and update it
            resampled_img = resample_img_to_img(source_img=img, target_img=result)
            arr = np.asanyarray(resampled_img.dataobj)
            nonzero_voxels = arr != 0
            num_conflicts += np.count_nonzero(result_arr[nonzero_voxels])
            result_arr[nonzero_voxels] = arr[nonzero_voxels]

        if num_conflicts > 0:
            num_voxels = np.count_nonzero(result_arr)
            logger.warning(
                f"Merging fragments required to overwrite {num_conflicts} "
                f"conflicting voxels ({num_conflicts / num_voxels * 100.:2.1f}%)."
            )

        return result

    def fetch(
        self,
        fragment: str = None,
        voi: _boundingbox.BoundingBox = None,
        label: int = None
    ):
        """
        Loads and returns a Nifti1Image object

        Parameters
        ----------
        fragment: str
            Optional name of a fragment volume to fetch, if any.
            For example, some volumes are split into left and right hemisphere fragments.
            see :func:`~siibra.volumes.Volume.fragments`
        voi : BoundingBox
            optional specification of a volume of interest to fetch.
        label: int, default: None
            Optional: a label index can be provided. Then the mask of the
            3D volume will be returned, where voxels matching this label
            are marked as "1".
        """

        result = None
        if len(self._img_loaders) > 1:
            if fragment is None:
                logger.info(
                    f"Merging fragments [{', '.join(self._img_loaders.keys())}]. "
                    f"You can select one using {self.__class__.__name__}.fetch(fragment=<name>)."
                )
                result = self._merge_fragments()
            else:
                matched_names = [n for n in self._img_loaders if fragment.lower() in n.lower()]
                if len(matched_names) != 1:
                    raise ValueError(
                        f"Requested fragment '{fragment}' could not be matched uniquely "
                        f"to [{', '.join(self._img_loaders)}]"
                    )
                else:
                    result = self._img_loaders[matched_names[0]]()
        else:
            assert len(self._img_loaders) > 0
            fragment_name, loader = next(iter(self._img_loaders.items()))
            if (fragment_name is not None) and (fragment is not None):
                assert fragment.lower() in fragment_name.lower()
            result = loader()

        if voi is not None:
            zoom_xyz = np.array(result.header.get_zooms())  # voxel dimensions in xyzt_units
            bb_vox = voi.transform(np.linalg.inv(result.affine))
            x0, y0, z0 = np.floor(np.array(bb_vox.minpoint.coordinate) / zoom_xyz).astype(int)
            x1, y1, z1 = np.ceil(np.array(bb_vox.maxpoint.coordinate) / zoom_xyz).astype(int)
            shift = np.identity(4)
            shift[:3, -1] = bb_vox.minpoint
            result = nib.Nifti1Image(
                dataobj=result.dataobj[x0:x1, y0:y1, z0:z1],
                affine=np.dot(result.affine, shift),
                dtype=result.header.get_data_dtype(),
            )

        if label is not None:
            result = nib.Nifti1Image(
                (result.get_fdata() == label).astype('uint8'),
                result.affine,
                dtype='uint8'
            )

        return result

    def get_shape(self, resolution_mm=None):
        if resolution_mm is not None:
            raise NotImplementedError(
                "NiftiVolume does not support to specify different image resolutions"
            )
        try:
            loader_shapes = {loader().shape for loader in self._img_loaders.values()}
            if len(loader_shapes) == 1:
                return next(iter(loader_shapes))
            else:
                raise RuntimeError(f"Fragments have different shapes: {loader_shapes}")
        except AttributeError as e:
            logger.error(
                f"Invalid object type/s {[type(loader()) for loader in self._img_loaders.values()]} of image for {self}."
            )
            raise (e)

    def is_float(self):
        return all(
            loader().dataobj.dtype.kind == "f"
            for loader in self._img_loaders.values()
        )

    def find_peaks(self, min_distance_mm=5):
        """
        Find peaks in the image data.

        Parameters
        ----------
        min_distance_mm : float
            Minimum distance between peaks in mm

        Returns:
        --------
        PointCloud
        """

        from skimage.feature.peak import peak_local_max
        from ...commons import affine_scaling

        img = self.fetch()
        dist = int(min_distance_mm / affine_scaling(img.affine) + 0.5)
        voxels = peak_local_max(
            img.get_fdata(),
            exclude_border=False,
            min_distance=dist,
        )
        return (
            pointcloud.PointCloud(
                [np.dot(img.affine, [x, y, z, 1])[:3] for x, y, z in voxels],
                space=self.space,
            ),
            img,
        )


class ZipContainedNiftiProvider(NiftiProvider, srctype="zip/nii"):

    def __init__(self, src: str):
        """
        Construct a new NIfTI volume source, from url, local file, or Nift1Image object.
        """
        _provider.VolumeProvider.__init__(self)
        zipurl, zipped_file = src.split(" ")
        req = requests.ZipfileRequest(zipurl, zipped_file)
        self._img_loaders = {None: lambda req=req: req.data}

        # required for self._url property
        self._init_url = src
