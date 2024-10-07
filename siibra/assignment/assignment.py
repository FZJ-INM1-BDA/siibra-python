# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type, Callable, Dict, Iterable, List, TypeVar, Union
from collections import defaultdict
from itertools import product
import math

from .attribute_qualification import (
    qualify as attribute_qualify,
    is_qualifiable as attribute_is_qualifiable,
)
from ..commons.logger import logger
from ..attributes import AttributeCollection
from ..attributes.locations import Location, BoundingBox
from ..attributes.dataproviders import DataRecipe
from ..concepts import AtlasElement, QueryParam
from ..atlases import Region, Space
from ..exceptions import InvalidAttrCompException, UnregisteredAttrCompException

V = TypeVar("V")

T = Callable[[AttributeCollection], Iterable[AttributeCollection]]

collection_gen: Dict[Type[AttributeCollection], List[V]] = defaultdict(list)


def match(col_a: AttributeCollection, col_b: AttributeCollection) -> bool:
    """Given AttributeCollection col_a, col_b, compare the product of their respective
    attributes, until:

    - If any of the permutation of the attribute matches, returns True.
    - If InvalidAttrCompException is raised, return False.
    - If all product of attributes are exhausted, return False
    """
    try:
        if next(qualify(col_a, col_b)):
            return True
        return False
    except StopIteration:
        return False
    except UnregisteredAttrCompException:
        return False


def qualify(col_a: AttributeCollection, col_b: AttributeCollection):
    """Given AttributeCollection col_a, col_b, yield the tuple (attr_a, attr_b, qualification) from the
    product of the respective attributes.

    Yields: attribute_a, attribute_b, Qualification
    Raises: UnregisteredAttrCompException if no combination of attributes can be found.
    """
    attr_compared_flag = False
    for attra, attrb in product(
        filter(lambda a: attribute_is_qualifiable(type(a)), col_a.attributes),
        filter(lambda a: attribute_is_qualifiable(type(a)), col_b.attributes),
    ):
        try:
            match_result = attribute_qualify(attra, attrb)
            attr_compared_flag = True
            if match_result:
                yield attra, attrb, match_result
        except UnregisteredAttrCompException:
            continue
        except InvalidAttrCompException as e:
            logger.debug(f"match exception {e}")
            return
    if attr_compared_flag:
        return
    raise UnregisteredAttrCompException


def preprocess_concept(concept: Union[AtlasElement, Location, DataRecipe]):
    """
    User provided concepts may need preprocessed. e.g.

    - When user provides a boundingbox (attribute) this function converts it to a QueryParam with
    the corresponding boundingbox attribute.

    - When user provides Region, this function adds all mapped boundingboxes in all available spaces

    - When user provides Space, this function adds an infinite boundingbox
    """
    if isinstance(concept, (Location, DataRecipe)):
        concept = QueryParam(attributes=[concept])
    assert isinstance(
        concept, AttributeCollection
    ), f"Expect concept to be either AtlasElement or Location, but was {type(concept)} instead"

    if isinstance(concept, Region):
        bbox = None
        for space in concept.mapped_spaces:
            bbox = concept.extract_mask(space=space)
            if bbox is not None:
                break
        concept.attributes = (
            *concept.attributes,
            bbox,
        )

    if isinstance(concept, Space):
        # When user query space, we are assuming that they really want to see all features that overlaps with
        # an infinite bounding box in this space. If we query the full space, we run into trouble with comparison
        # of e.g. space.image (template image) x feature.region_spec
        inf_bbox = BoundingBox(
            space_id=concept.ID,
            minpoint=[-math.inf, -math.inf, -math.inf],
            maxpoint=[math.inf, math.inf, math.inf],
        )
        concept = QueryParam(attributes=[inf_bbox])
    return concept
