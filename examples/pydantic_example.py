from siibra.commons import TypedRegistry

from siibra.core.space import CommonCoordinateSpace

reg: TypedRegistry[CommonCoordinateSpace] = CommonCoordinateSpace.REGISTRY

print(
    [s.full_name for s in reg]
)