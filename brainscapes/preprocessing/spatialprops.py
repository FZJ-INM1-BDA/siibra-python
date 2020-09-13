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
    grid2mm = lambda pts : apply_affine(img_volume.affine,pts)

    # Extract properties of each connected component, recompute some
    # spatial props in physical coordinates, and return connected componente
    # as a list 
    regionprops = defaultdict(list)

    for desc,nim in label_volumes.items():
        L = np.asanyarray(nim.dataobj) 
        rprops = measure.regionprops(L)
        for i,rprop in enumerate(rprops):

            print("{0:60.60} {1:10.1%}".format(
                "Extracting spatial props "+desc,
                (i+1)/len(rprops)), end="\r")

            labels = np.unique(L[L==rprop.label])
            assert(len(labels)==1)
            label = labels[0]

            rprop.labelled_volume_description = desc

            # centroid in physical coordinates
            rprop.centroid_mm = grid2mm(rprop.centroid)

            # volume in physical coordinates
            rprop.volume_mm = rprop.area * pixel_spacing**3

            # approximate surface
            verts,faces,_,_ = measure.marching_cubes(L==label)
            rprop.surface_mm = measure.mesh_surface_area(grid2mm(verts),faces)

            regionprops[label].append(rprop)
        print()

    return regionprops
