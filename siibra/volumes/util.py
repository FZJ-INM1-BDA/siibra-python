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


import numpy as np
from skimage.filters import gaussian
import nibabel as nib


def create_gaussian_kernel(sigma=1, sigma_point=3):
    """
    Compute a 3D Gaussian kernel of the given bandwidth.
    """
    r = int(sigma_point * sigma)
    k_size = 2 * r + 1
    impulse = np.zeros((k_size, k_size, k_size))
    impulse[r, r, r] = 1
    kernel = gaussian(impulse, sigma)
    kernel /= kernel.sum()
    return kernel


def argmax_dim4(img, dim=-1):
    """
    Given a nifti image object with four dimensions, returns a modified object
    with 3 dimensions that is obtained by taking the argmax along one of the
    four dimensions (default: the last one). To distinguish the pure background
    voxels from the foreground voxels of channel 0, the argmax indices are
    incremented by 1 and label index 0 is kept to represent the background.
    """
    assert len(img.shape) == 4
    assert dim >= -1 and dim < 4
    newarr = np.asarray(img.dataobj).argmax(dim) + 1
    # reset the true background voxels to zero
    newarr[np.asarray(img.dataobj).max(dim) == 0] = 0
    return nib.Nifti1Image(dataobj=newarr, header=img.header, affine=img.affine)


def MI(arr1, arr2, nbins=100, normalized=True):
    """
    Compute the mutual information between two 3D arrays, which need to have the same shape.

    Parameters:
    arr1 : First 3D array
    arr2 : Second 3D array
    nbins : number of bins to use for computing the joint histogram (applies to intensity range)
    normalized : Boolean, default:True
        if True, the normalized MI of arrays X and Y will be returned,
        leading to a range of values between 0 and 1. Normalization is
        achieved by NMI = 2*MI(X,Y) / (H(X) + H(Y)), where  H(x) is the entropy of X
    """

    assert all(len(arr.shape) == 3 for arr in [arr1, arr2])
    assert (all(arr.size > 0) for arr in [arr1, arr2])

    # compute the normalized joint 2D histogram as an
    # empirical measure of the joint probabily of arr1 and arr2
    pxy, _, _ = np.histogram2d(arr1.ravel(), arr2.ravel(), bins=nbins)
    pxy /= pxy.sum()

    # extract the empirical propabilities of intensities
    # from the joint histogram
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x

    # compute the mutual information
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # nonzero value indices
    MI = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    if not normalized:
        return MI

    # normalize, using the sum of their individual entropies H
    def entropy(p):
        nz = p > 0
        assert np.count_nonzero(nz) > 0
        return -np.sum(p[nz] * np.log(p[nz]))

    Hx, Hy = [entropy(p) for p in [px, py]]
    assert (Hx + Hy) > 0
    NMI = 2 * MI / (Hx + Hy)
    return NMI
