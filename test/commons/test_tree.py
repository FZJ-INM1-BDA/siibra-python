from anytree import Node
import pytest
from siibra.commons_new.tree import collapse_nodes

grand = Node("grand")

parent1 = Node("parent1", parent=grand)
parent2 = Node("parent2", parent=grand)


child11 = Node("child11", parent=parent1)
child12 = Node("child12", parent=parent1)
child21 = Node("child21", parent=parent2)
child22 = Node("child22", parent=parent2)


grandchild221 = Node("grandchild221", parent=child22)


collapse_node_args = [
    ([grand], [grand]),
    ([child11], [child11]),
    ([grandchild221], [child22]),
    ([child11, child12], [parent1]),
    ([child11, child12, parent1], [parent1]),
    ([child11, child12, parent1, child21], [parent1, child21]),
    ([child11, child21], [child11, child21]),
    ([child11, child12, child21, child22], [grand]),
]


@pytest.mark.parametrize("input, expected", collapse_node_args)
def test_collapse_nodes(input, expected):
    assert set(collapse_nodes(input)) == set(expected)
