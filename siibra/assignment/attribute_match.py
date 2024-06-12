from ..commons import logger
from ..commons_new.comparison import Comparison
from ..commons_new.string import fuzzy_match, clear_name
from ..concepts import Attribute
from ..concepts.attribute import TruthyAttr
from ..exceptions import UnregisteredAttrCompException
from ..descriptions import Modality, RegionSpec, Name, ID, Paradigm, Cohort, AggregateBy
from ..locations import Pt, PointCloud, BBox, intersect, DataClsLocation


_attr_match: Comparison[Attribute, bool] = Comparison()

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
    if isinstance(attra, TruthyAttr) or isinstance(attrb, TruthyAttr):
        return True
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


@register_attr_comparison(AggregateBy, AggregateBy)
def compare_aggregate_by(aggbya: AggregateBy, aggbyb: AggregateBy):
    return (aggbya.key == aggbyb.key
            and aggbya.value == aggbyb.value)


@register_attr_comparison(Cohort, Cohort)
@register_attr_comparison(Paradigm, Paradigm)
@register_attr_comparison(Modality, Modality)
def compare_modality(mod1: Modality, mod2: Modality):
    return mod1.value.lower() == mod2.value.lower()


@register_attr_comparison(RegionSpec, RegionSpec)
def compare_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):
    if (
        regspec1.parcellation_id is not None
        and regspec2.parcellation_id is not None
        and regspec1.parcellation_id != regspec2.parcellation_id
    ):
        return False

    cleaned_name1 = clear_name(regspec1.value)
    cleaned_name2 = clear_name(regspec2.value)
    return fuzzy_match(cleaned_name1, cleaned_name2) or fuzzy_match(
        cleaned_name2, cleaned_name1
    )


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