import numpy as np
from skimage import measure
from nibabel.affines import apply_affine

from .feature import RegionalFeature
from .. import logger
from ..termplot import FontStyles as style

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

    def __init__(self,atlas,space,custom_region=None):
        """
        Construct region properties from a region selected in the given atlas,
        in the specified template space. Optinally, a custom region can be
        supplied and the selected region of the atlas is ignored.

        Parameters
        ----------
        atlas : Atlas
            An atlas object
        space : Space
            A template space (needs to be supported by the given atlas)
        custom_region : Region (default: None)
            If specified, this region will be used instead of the currently
            selected region in the atlas.
        """
        assert(space in atlas.spaces)

        region = atlas.selected_region if custom_region is None else custom_region
        RegionalFeature.__init__(self,region)

        tmpl = atlas.get_template(space)
        mask = atlas.get_mask(space)
        M = np.asanyarray(mask.dataobj) 
        self.attrs = {}

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
        if len(rprops)==0:
            logger.warn('Zero size area - constructing an empty region property object .')
            return 
        assert(len(rprops)==1)
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
        """
        Formats the main attributes as a multiline string.
        """
        return "\n".join([style.BOLD+'Region properties of "{}"'.format(self.region)+style.END] 
                +["{}{:>15}{} {}".format(
            style.ITALIC,label,style.END,self.attrs[label])
            for label in self.main_props])

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.attrs.keys():
            return self.attrs[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))


