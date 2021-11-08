from siibra.core.jsonable import SiibraSerializable
from pydantic import ValidationError, BaseModel

def get_model(cls: SiibraSerializable):
    if issubclass(cls.SiibraSerializationSchema, BaseModel):
        return cls.SiibraSerializationSchema

    # if Union type, should have attr __args__
    if hasattr(cls.SiibraSerializationSchema, '__args__'):
        def json_serialize(**kwargs):
            errors = ''
            for Model in cls.SiibraSerializationSchema.__args__:
                try:
                    return Model(**kwargs)
                except ValidationError as e:
                    errors += str(e)
                    pass
            else:
                raise ValidationError(f'No models in SiibraSerializationSchemas fits: {errors}')
        return json_serialize
    
    
    raise TypeError('cls.SiibraSerializationSchema must either be a tuple, Union, or extends BaseModel')
