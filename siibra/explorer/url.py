from typing import Optional, TYPE_CHECKING
from urllib.parse import quote_plus
from numpy import int32
import numpy as np
import re
from dataclasses import dataclass
import math

from .util import encode_number, separator, cipher, neg, decode_number, post_process

if TYPE_CHECKING:
    from siibra.core.atlas import Atlas
    from siibra.core.space import Space
    from siibra.locations import BoundingBox, Point
    from siibra.core.parcellation import Parcellation
    from siibra.core.region import Region
    from siibra.features.feature import Feature

class DecodeNavigationException(Exception): pass

min_int32=-2_147_483_648
max_int32=2_147_483_647


default_root_url='https://atlases.ebrains.eu/viewer/'

def sanitize_id(id: str):
    return id.replace('/', ':')

def get_perspective_zoom(atlas: "Atlas", space: "Space", parc: "Parcellation", region: Optional["Region"]):
    import siibra
    if atlas is siibra.atlases['rat'] or atlas is siibra.atlases['mouse']:
        return 200000
    return 2000000

def get_zoom(atlas: "Atlas", space: "Space", parc: "Parcellation", region: Optional["Region"]):
    import siibra
    if atlas is siibra.atlases['rat'] or atlas is siibra.atlases['mouse']:
        return 35000
    return 350000

supported_prefix = (
  "nifti://",
  "swc://",
  "precomputed://",
  "deepzoom://"
)

def append_query_params(url: str, *args, query_params={}, **kwargs):
    query_str = "&".join([f"{key}={quote_plus(value)}" for key, value in query_params.items()])
    if len(query_str) > 0:
        query_str = "?" + query_str
    return url + query_str

@post_process(append_query_params)
def encode_url(atlas: "Atlas", space: "Space", parc: "Parcellation", region: Optional["Region"]=None, *, root_url=default_root_url, external_url:str=None, feature: "Feature"=None, ignore_warning=False, query_params={}):
    
    overlay_url = None
    if external_url:
        assert any([external_url.startswith(prefix) for prefix in supported_prefix]), f"url needs to start with {(' , '.join(supported_prefix))}"
        overlay_url = '/x-overlay-layer:{url}'.format(
            url=external_url.replace("/", "%2F")
        )

    zoom = get_zoom(atlas, space, parc, region)
    pzoom = get_perspective_zoom(atlas, space, parc, region)
    
    zoom_kwargs = {
        "encoded_pzoom": encode_number(pzoom, False),
        "encoded_zoom": encode_number(zoom, False)
    }
    nav_string='/@:0.0.0.-W000.._eCwg.2-FUe3._-s_W.2_evlu..{encoded_pzoom}..{encoded_nav}..{encoded_zoom}'

    return_url='{root_url}#/a:{atlas_id}/t:{template_id}/p:{parc_id}{overlay_url}'.format(
        root_url    = root_url,
        atlas_id    = sanitize_id(atlas.id),
        template_id = sanitize_id(space.id),
        parc_id     = sanitize_id(parc.id),
        overlay_url = overlay_url if overlay_url else "",
    )

    if feature is not None:
        return_url = return_url + f"/f:{sanitize_id(feature.id)}"

    if region is None:
        return return_url + nav_string.format(encoded_nav='0.0.0', **zoom_kwargs)
    
    return_url=f'{return_url}/rn:{get_hash(region.name)}'

    try:
        result_props=region.spatial_props(space, maptype='labelled')
        if len(result_props.components) == 0:
            return return_url + nav_string.format(encoded_nav='0.0.0', **zoom_kwargs)
    except Exception as e:
        print(f'Cannot get_spatial_props {str(e)}')
        if not ignore_warning:
            raise e
        return return_url + nav_string.format(encoded_nav='0.0.0', **zoom_kwargs)

    centroid=result_props.components[0].centroid

    encoded_centroid=separator.join([ encode_number(math.floor(val * 1e6)) for val in centroid ])
    return_url=return_url + nav_string.format(encoded_nav=encoded_centroid, **zoom_kwargs)
    return return_url

@dataclass
class DecodedUrl:
    bounding_box: "BoundingBox"

def decode_url(url: str, vp_length=1000):
    import siibra
    try:
        space_match = re.search(r'/t:(?P<space_id>[^/]+)', url)
        space_id = space_match.group("space_id")
        space_id = space_id.replace(":", "/")
        space = siibra.spaces[space_id]
    except Exception as e:
        raise DecodeNavigationException from e

    nav_match = re.search(r'/@:(?P<navigation_str>.+)/?', url)
    navigation_str = nav_match.group("navigation_str")
    for char in navigation_str:
        assert char in cipher or char in [neg, separator], f"char {char} not in cipher, nor separator/neg"
    
    try:
        ori_enc, pers_ori_enc, pers_zoom_enc, pos_enc, zoomm_enc = navigation_str.split(f"{separator}{separator}")
    except Exception as e:
        raise DecodeNavigationException from e
    
    try:
        x_enc, y_enc, z_enc = pos_enc.split(separator)
        pos = [decode_number(val) for val in [x_enc, y_enc, z_enc]]
        zoom = decode_number(zoomm_enc)

        # zoom = nm/pixel
        pt1 = [(coord - (zoom * vp_length / 2)) / 1e6 for coord in pos]
        pt1 = Point(pt1, space)
        
        pt2 = [(coord + (zoom * vp_length / 2)) / 1e6 for coord in pos]
        pt2 = Point(pt2, space)

    except Exception as e:
        raise DecodeNavigationException from e

    bbx = BoundingBox(pt1, pt2, space)
    return DecodedUrl(bounding_box=bbx)
    
def get_hash(full_string: str):
    return_val=0
    with np.errstate(over="ignore"):
        for char in full_string:
            # overflowing is expected and in fact the whole reason why convert number to int32
            
            # in windows, int32((0 - min_int32) << 5), rather than overflow to wraper around, raises OverflowError
            shifted_5 = int32(
                (return_val - min_int32) if return_val > max_int32 else return_val
            << 5)

            return_val = int32(shifted_5 - return_val + ord(char))
            return_val = return_val & return_val
    hex_val = hex(return_val)
    return hex_val[3:]
