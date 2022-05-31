# Serializable

This is a recommendation on how new classes should be added to siibra, so that instances can be most easily serialized into json for over HTTP transportation.

1. extend `JSONSerializable`
2. override `get_model_type` class method (per `JSONSerializable` class definition)
3. override `model_id` property (per `JSONSerializable` class definition). Ensure that `model_id` is a substring of `get_model_type()` (e.g. `siibra/features/voi` and `siibra/features/voi/0011-aabb`)
5. override `to_model` method, ensure the return value is annotated, and is a subclass of `pydantic.BaseModel`
6. the `to_model` method should expect `detail` kwarg, and arbitary kwargs (i.e. add `**kwargs` to the argument list).
    - `detail` flag is a boolean, and allow compute intensive and/or IO intensive tasks to only be performed when necessary
