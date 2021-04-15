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
            'centroid_mm',
            'volume_mm',
            'surface_mm',
            'is_cortical']

    def __init__(self,atlas,space):
        """
        Construct region properties from a region selected in the given atlas,
        in the specified template space. 

        Parameters
        ----------
        atlas : Atlas
            An atlas object
        space : Space
            A template space (needs to be supported by the given atlas)
        """
        assert(space in atlas.spaces)

        RegionalFeature.__init__(self,atlas.selected_region)
        self.attrs = {}
    
        # derive non-spatial properties
        assert(atlas.selected_parcellation.regiontree.find('cerebral cortex'))
        self.attrs['is_cortical'] = atlas.selected_region.has_parent('cerebral cortex')

        # compute mask in the given space
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
        if len(rprops)==0:
            logger.warning('Region "{}" has zero size - {}'.format(
                self.region.name, "constructing an empty region property object"))
            self.attrs['centroid_mm'] = 0.
            self.attrs['volume_mm'] = 0.
            self.attrs['surface_mm'] = 0.
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
        verts,faces,_,_ = measure.marching_cubes_lewiner(M)
        self.attrs['surface_mm'] = measure.mesh_surface_area(grid2mm(verts),faces)


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


