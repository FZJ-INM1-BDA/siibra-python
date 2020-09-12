from skimage import measure
import numpy as np
from collections import defaultdict
from nibabel.affines import apply_affine

def regionprops(nifti_volume):
    """
    Extracts basic properties of all regions in the volume.
    """
    A = nifti_volume.get_data()
    rprops = measure.regionprops(A)

    # check if we have multiple regions per label
    regions_per_label = defaultdict(int)
    for r in rprops:
        regions_per_label[r.label]+=1
    split_regions = [label 
            for label,N in regions_per_label.items()
            if N>1]
    if len(split_regions)>0:
        print("WARNING: Ignoring regions with multiple components:",split_regions)

    return {
            r.label : {
                'centroid' : apply_affine(nifti_volume.affine,r.centroid),
                'volume' : r.area
                }
            for r in rprops
            if r.label not in split_regions
            }
