from skimage import measure
import numpy as np
from collections import defaultdict
from nibabel.affines import apply_affine

# NOTE this does not work for histological volume like the BigBrain.
def regionprops(label_volumes,img_volume):
    """
    Extracts basic properties of all labelled regions in the volume.

    Parameters
    ----------
    label_volumes : Dictionary of nibabel Nifti1 image objects
        The brain volume labelling (list of parcellation maps).

    img_volume : nibabel Nifti1 image object 
        The corresponding reference image (typically the T1/T2 template).
    """

    # for now we expect isotropic physical coordinates given in mm for the
    # template volume
    # TODO be more flexible here
    pixel_unit = img_volume.header.get_xyzt_units()[0]
    assert(pixel_unit == 'mm')
    pixel_spacings = np.unique(img_volume.header.get_zooms()[:3])
    assert(len(pixel_spacings)==1)
    pixel_spacing = pixel_spacings[0]

    # Extract properties of each connected component, recompute some
    # spatial props in physical coordinates, and return connected componente
    # as a list 
    regionprops = defaultdict(list)
    for desc,nim in label_volumes.items():
        L = np.asanyarray(nim.dataobj) 
        for rprop in measure.regionprops(L):
            labels = np.unique(L[L==rprop.label])
            assert(len(labels)==1)
            rprop.centroid_mm = apply_affine(img_volume.affine,rprop.centroid)
            rprop.area_mm = rprop.area * pixel_spacing**3
            rprop.labelled_volume_description = desc
            regionprops[labels[0]].append(rprop)

    return regionprops
