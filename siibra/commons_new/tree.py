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
