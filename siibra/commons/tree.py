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

from anytree import NodeMixin
from typing import TypeVar, List

T = TypeVar("T", bound=NodeMixin)


def collapse_nodes(input_nodes: List[T]) -> List[T]:
    list_of_nodes = list(input_nodes)

    # Old implementation, faulty
    # complete_parents = list(
    #     {
    #         r.parent
    #         for r in list_of_nodes
    #         if (r.parent is not None)
    #         and all((c in list_of_nodes) for c in r.parent.children)
    #     }
    # )

    # if len(complete_parents) == 0:
    #     return list_of_nodes

    # # filter child nodes again
    # list_of_nodes += complete_parents
    # list_of_nodes = list(set(list_of_nodes))
    # return [r for r in list_of_nodes if (r.parent not in list_of_nodes)]

    circuit_breaker = 100
    while True:
        circuit_breaker -= 1
        if circuit_breaker < 0:
            raise RuntimeError
        for active_node in list_of_nodes:
            if active_node.parent is None:
                continue
            siblings = active_node.siblings
            if all((sibling in list_of_nodes) for sibling in siblings):
                list_of_nodes = [
                    node
                    for node in list_of_nodes
                    if (node is not active_node and node not in siblings)
                ]
                parent: T = active_node.parent
                list_of_nodes.append(parent)
                break
        else:
            break
    return list(set(list_of_nodes))
