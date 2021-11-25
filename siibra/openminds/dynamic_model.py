url = 'https://raw.githubusercontent.com/HumanBrainProject/openMINDS/documentation/v3/SANDS/v3/miscellaneous/coordinatePoint.schema.json'
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, create_model
from urllib.parse import urlparse
import requests
import json

class OpenMindsValidationException(Exception): pass


def sanitize_key(key: str):

    illegal_chars = r':/\\.'
    replace_char = '_'
    return_key = key
    for char in illegal_chars:
        return_key = return_key.replace(char, replace_char)
    return return_key


class EnumType(str, Enum):
    string = 'string'
    number = 'number'
    object = 'object'
    array = 'array'


class BaseJsonSchema(BaseModel):
    # non standard keys
    under_type: Optional[str] = Field(alias="_type")
    under_instruction: Optional[str] = Field(alias='_instruction')
    under_linked_types: Optional[List[str]] = Field(alias='_linkedTypes')

    description: Optional[str]
    type: Optional[EnumType]
    title: Optional[str]


class JsonSchema(BaseJsonSchema):

    # identifier keys
    dollar_id: Optional[str] = Field(alias="$id")
    dollar_schema: Optional[str] = Field(alias="$schema")

    const: Optional[str]
    required: Optional[List[str]]
    properties: Optional[Dict[str, 'TypeProperty']]
    enum: Optional[List[str]]
    format: Optional[str]


class IfThenElseSchema(BaseModel):
    type: str = Field(EnumType.object, const=EnumType.object)
    key_if: JsonSchema = Field(alias='if')
    key_then: JsonSchema = Field(alias='then')
    key_else: JsonSchema = Field(alias='else')


class ArrayJsonSchema(BaseJsonSchema):
    type: str = Field(EnumType.array, const=EnumType.array)
    items: JsonSchema
    max_items: Optional[int] = Field(alias='maxItems')
    min_items: Optional[int] = Field(alias='minItems')


TypeProperty = Union[IfThenElseSchema, ArrayJsonSchema, JsonSchema]


IfThenElseSchema.update_forward_refs()
ArrayJsonSchema.update_forward_refs()
JsonSchema.update_forward_refs()


class AtBaseModel(BaseModel):
    at_id: Optional[str] = Field(alias='@id')
    at_type: Optional[str] = Field(alias='@type')


ignore_keys = ['@id', '@type']


def store_model_decorator(opts=None):
    cache: Dict[str, BaseModel] = {}

    def create_model_decorator(func):
        def wrapper(model_name: str, json_schema: JsonSchema):
            if model_name in cache:
                return cache[model_name]
            
            output = func(model_name, json_schema)
            cache[model_name] = output

            return output
        return wrapper
    return create_model_decorator


@store_model_decorator()
def get_model_from_json(model_name:str, json_schema: TypeProperty):
    
    model_name_ = model_name
    if model_name_ is None:
        b = urlparse(json_schema.dollar_id)
        model_name_ = '{protocol}://{netloc}{path}'.format(
            protocol=b.scheme,
            netloc=b.netloc,
            path=b.path)

    sanitized_model_name = sanitize_key(model_name_)

    model_prop = {}
    if isinstance(json_schema, IfThenElseSchema):

        assert json_schema.key_if.required
        
        json_schema.key_if.required

        then_json = json_schema.key_then.dict()
        if_json = json_schema.key_if.dict()
        then_json['required'] = [
            *then_json.get('required'),
            *if_json.get('required'),
        ]

        if_then_type = get_model_from_json(f'{model_name_}-ifthen', JsonSchema(**then_json))
        else_type = get_model_from_json(f'{model_name_}-else', json_schema.key_else)

        return Union[if_then_type, else_type]
    elif isinstance(json_schema, ArrayJsonSchema):
        Model = get_model_from_json(f'{model_name_}-item', json_schema.items)
        return List[Model]
    else:
        if json_schema.properties:
            for key in json_schema.properties:
                if key in ignore_keys:
                    continue
                sanitized_key = sanitize_key(key)
                required_flag = key in json_schema.required or []

                model = get_model_from_json(f'{model_name_}-{key}', json_schema.properties[key])

                model_ = model if required_flag else Optional[model]
                model_prop[sanitized_key] = (
                    model_,
                    Field(alias=key),
                )

    return create_model(
        sanitized_model_name,
        **model_prop,
        __base__=AtBaseModel,
    )

def main():
    resp = requests.get('https://raw.githubusercontent.com/HumanBrainProject/openMINDS/dbb4c54/v3/SANDS/v3/miscellaneous/coordinatePoint.schema.json')
    assert (resp.status_code == 200)
    json_content = resp.json()
    schema = JsonSchema(**json_content)
    PointModel = get_model_from_json(None, schema)
    print(
        PointModel
    )
    
if __name__ == '__main__':
    main()