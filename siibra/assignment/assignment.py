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

from typing import Type, Callable, Dict, Iterable, List, TypeVar
from collections import defaultdict
from itertools import product


from .attribute_qualification import qualify as attribute_qualify
from ..commons_new.logger import logger
from ..attributes import AttributeCollection
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
    for attra, attrb in product(col_a.attributes, col_b.attributes):
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
