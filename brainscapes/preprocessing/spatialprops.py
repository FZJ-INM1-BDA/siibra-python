from skimage import measure
import numpy as np
from collections import defaultdict
from nibabel.affines import apply_affine

# NOTE this does not work for histological volume like the BigBrain.
def regionprops(label_volume,img_volume):
    """
    Extracts basic properties of all labelled regions in the volume.

    Parameters
    ----------
    label_volume : nibabel Nifti1 image object 
        The labelled brain volume (parcellation map).

    img_volume : nibabel Nifti1 image object 
        The corresponding reference image (typically the T1/T2 template).
    """

    # relabel connected components to handle the case of multiple
    # areas with identical label (e.g., in the different hemispheres).
    T = np.asanyarray(img_volume.dataobj)
    L = np.asanyarray(label_volume.dataobj)
    L_cc = measure.label(L)
    print('Original map has {} labels, relabelled map has {}.'.format(
        len(np.unique(L)), len(np.unique(L_cc)) ))
        
    # for now we expect isotropic physical coordinates given in mm for the
    # template volume
    # TODO be more flexible here
    pixel_unit = img_volume.header.get_xyzt_units()[0]
    assert(pixel_unit == 'mm')
    pixel_spacings = np.unique(img_volume.header.get_zooms()[:3])
    assert(len(pixel_spacings)==1)
    pixel_spacing = pixel_spacings[0]

    # Extraction properties of each connected component, recompute some
    # spatial props in physical coordinates, and return connected componente
    # as a list # per original atlas label
    regionprops_cc = measure.regionprops(L_cc)
    regionprops = defaultdict(list)
    for rprop_cc in regionprops_cc:
        labels = np.unique(L[L_cc==rprop_cc.label])
        assert(len(labels)==1)
        rprop_cc.centroid_mm = apply_affine(img_volume.affine,rprop_cc.centroid)
        rprop_cc.area_mm = rprop_cc.area * pixel_spacing**3
        regionprops[labels[0]].append(rprop_cc)

    return regionprops
