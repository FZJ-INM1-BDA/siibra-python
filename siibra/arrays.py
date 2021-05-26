# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numbers
from scipy.ndimage import gaussian_filter
import nibabel as nib
import re

def bbox3d(A,affine=None):
    """
    Bounding box of nonzero values in a 3D array.
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    If affine not None, the affine is applied to the bounding box.
    """
    r = np.any(A, axis=(1, 2))
    c = np.any(A, axis=(0, 2))
    z = np.any(A, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    bbox = np.array([
        [rmin, rmax], 
        [cmin, cmax], 
        [zmin, zmax],
        [1,1]
    ])

    return bbox if affine is None else np.dot(affine,bbox)

def parse_coordinate_str(cstr:str,unit='mm'):
    pat=r'([-\d\.]*)'+unit
    digits = re.findall(pat,cstr)
    if len(digits)==3:
        return np.array([float(d) for d in digits])
    else:
        return None

def create_homogeneous_array(xyz):
    """
    From one 3D point or list of multiple 3D points, build the Nx4 homogenous coordinate array.

    Parameters
    ----------

    xyz_phys : 3D coordinate tuple, list of 3D tuples, or Nx3 array of coordinate tuples
        3D point(s) in physical coordinates of the template space of the
        ParcellationMap
    """
    if isinstance(xyz,str):
        parsed = parse_coordinate_str(xyz)
        if parsed is not None:
            XYZH = np.ones((1,4))
            XYZH[0,:3] = parsed
            return XYZH
    elif isinstance(xyz,list) and all(isinstance(e,str) for e in xyz):
        parsed = (parse_coordinate_str(s) for s in xyz)
        valid = [np.r_[p,1] for p in parsed if p is not None]
        if len(valid)>0:
            return np.array(valid)
    elif isinstance(xyz[0],numbers.Number):
        # only a single point provided
        assert(len(xyz) in [3,4])
        XYZH = np.ones((1,4))
        XYZH[0,:len(xyz)] = xyz
        return XYZH
    else:
        # assume list of coordinates
        XYZ = np.array(xyz)
        assert(XYZ.shape[1]==3)
        return np.c_[XYZ,np.ones_like(XYZ[:,0])]
    raise ValueError(f"Could not parse xyz coordinates: {xyz}")

def assert_homogeneous_3d(xyz):
    if len(xyz)==4:
        return xyz
    else:
        return np.r_[xyz,1]

def create_gaussian_kernel(sigma=1,sigma_point=3):
    """
    Compute a 3D Gaussian kernel of the given bandwidth.
    """
    r = int(sigma_point*sigma)
    k_size = 2*r + 1
    impulse = np.zeros((k_size,k_size,k_size))
    impulse[r,r,r] = 1
    kernel = gaussian_filter(impulse, sigma)
    kernel /= kernel.sum()
    return kernel

def argmax_dim4(img,dim=-1):
    """
    Given a nifti image object with four dimensions, returns a modified object
    with 3 dimensions that is obtained by taking the argmax along one of the
    four dimensions (default: the last one). To distinguish the pure background
    voxels from the foreground voxels of channel 0, the argmax indices are
    incremented by 1 and label index 0 is kept to represent the background.
    """
    assert(len(img.shape)==4)
    assert(dim>=-1 and dim<4)
    newarr = np.asarray(img.dataobj).argmax(dim)+1
    # reset the true background voxels to zero
    newarr[np.asarray(img.dataobj).max(dim)==0]=0
    return nib.Nifti1Image(
            dataobj = newarr,
            header = img.header,
            affine = img.affine )