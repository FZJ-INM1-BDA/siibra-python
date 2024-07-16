from ..commons_new.logger import logger
from ..commons_new.binary_op import BinaryOp
from ..commons_new.string import fuzzy_match
from ..concepts import Attribute
from ..exceptions import UnregisteredAttrCompException
from ..descriptions import Modality, RegionSpec, Name, ID, Facet
from ..locations import Pt, PointCloud, BBox, intersect, DataClsLocation


_attr_match: BinaryOp[Attribute, bool] = BinaryOp()

register_attr_comparison = _attr_match.register


def match(attra: Attribute, attrb: Attribute):
    """Attempt to match one attribute with another attribute.

    n.b. the comparison is symmetrical. That is, match(a, b) is the same as match(b, a)

    Parameters
    ----------
    attra: Attribute
    attrb: Attribute

    Returns
    -------
    bool
        If the attra matches with attrb

    Raises
    ------
    UnregisteredAttrCompException
        If the comparison of type(attra) and type(attrb) has not been registered
    InvalidAttrCompException
        If the comparison of type(attra) and type(attrb) has been registered, but the their
        value could not directly be compared (e.g. locations in different spaces)
    """
    val = _attr_match.get(attra, attrb)
    if val is None:
        logger.debug(
            f"{type(attra)} and {type(attrb)} comparison has not been registered"
        )
        raise UnregisteredAttrCompException
    fn, switch_arg = val
    args = [attrb, attra] if switch_arg else [attra, attrb]
    result = fn(*args)
    logger.debug(f"Comparing {attra} with {attrb}, result in {result}")
    return result


@register_attr_comparison(ID, ID)
def compare_id(id0: ID, id1: ID):
    return id0.value == id1.value


@register_attr_comparison(Name, Name)
def compare_name(name1: Name, name2: Name):
    return fuzzy_match(name1.value, name2.value) or fuzzy_match(
        name2.value, name1.value
    )


@register_attr_comparison(Facet, Facet)
def compare_facet(facet_a: Facet, facet_b: Facet):
    return (facet_a.key == facet_b.key
            and facet_a.value == facet_b.value)

@register_attr_comparison(Modality, Modality)
def compare_modality(mod1: Modality, mod2: Modality):
    return mod1.value.lower() == mod2.value.lower()


@register_attr_comparison(RegionSpec, RegionSpec)
def compare_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):
    from .attribute_qualification import qualify_regionspec
    return qualify_regionspec(regspec1, regspec2) is not None


@register_attr_comparison(Pt, Pt)
@register_attr_comparison(Pt, PointCloud)
@register_attr_comparison(Pt, BBox)
@register_attr_comparison(PointCloud, PointCloud)
@register_attr_comparison(PointCloud, BBox)
@register_attr_comparison(BBox, BBox)
def compare_loc_to_loc(loca: DataClsLocation, locb: DataClsLocation):
    return intersect(loca, locb) is not None


# TODO implement
# @register_attr_comparison(BBox, Image)
# def compare_bbox_to_image(bbox: BBox, image: Image):
#     if image.space_id != bbox.space_id:
#         raise InvalidAttrCompException(f"bbox and image are in different space. Cannot compare the two.")
#     raise NotImplementedError
