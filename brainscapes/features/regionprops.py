from .feature import RegionalFeature
import numpy as np
from skimage import measure
from nibabel.affines import apply_affine
from ..commons import termcolors as tc

class RegionProps(RegionalFeature):
    """
    Computes and represents spatial attributes of a region in an atlas.
    """

    # properties used when printing this feature
    main_props = [
            'label',
            'centroid_mm',
            'volume_mm',
            'surface_mm',
            'is_cortical']

    def __init__(self,atlas,space):

        region = atlas.selected_region
        RegionalFeature.__init__(self,region)

        tmpl = atlas.get_template(space)
        mask = atlas.get_mask(space)
        M = np.asanyarray(mask.dataobj) 

        # Setup coordinate units 
        # for now we expect isotropic physical coordinates given in mm for the
        # template volume
        # TODO be more flexible here
        pixel_unit = tmpl.header.get_xyzt_units()[0]
        assert(pixel_unit == 'mm')
        pixel_spacings = np.unique(tmpl.header.get_zooms()[:3])
        assert(len(pixel_spacings)==1)
        pixel_spacing = pixel_spacings[0]
        grid2mm = lambda pts : apply_affine(tmpl.affine,pts)

        # Extract properties of each connected component, recompute some
        # spatial props in physical coordinates, and return connected componente
        # as a list 
        rprops = measure.regionprops(M)
        assert(len(rprops)==1)
        self.attrs = {}
        for prop in rprops[0]:
            try:
                self.attrs[prop] = rprops[0][prop]
            except NotImplementedError:
                pass

        # Transfer some properties into physical coordinates
        self.attrs['centroid_mm'] = grid2mm(self.attrs['centroid'])
        self.attrs['volume_mm'] = self.attrs['area'] * pixel_spacing**3
        # TODO test if the surface estimate makes really sense
        verts,faces,_,_ = measure.marching_cubes(M)
        self.attrs['surface_mm'] = measure.mesh_surface_area(grid2mm(verts),faces)

        # add additional attributes
        assert(atlas.regiontree.find('cerebral cortex'))
        self.attrs['is_cortical'] = atlas.selected_region.has_parent('cerebral cortex')

    def __dir__(self):
        return list(self.attrs.keys())

    def __str__(self):
        return "\n".join(["{}{:>15}{} {}".format(
            tc.BOLD,label,tc.END,self.attrs[label])
            for label in self.main_props])

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))



