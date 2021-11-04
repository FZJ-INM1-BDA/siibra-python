from typing import Type, Union
from .jsonable import SiibraSerializable
from uuid import uuid4
class CircularReferenceException(Exception): pass
class UnJSONableException(Exception): pass

class JSONEncoder:

    @staticmethod
    def encode(obj: Union[None, str, int, float, list, tuple, dict, SiibraSerializable], detail=False, depth_threshold=1, **kwargs):
        encoder = JSONEncoder(depth_threshold=depth_threshold, **kwargs)
        return encoder.get_json(obj, detail=detail, **kwargs)

    def __init__(self, nested=False, debug=False, depth_threshold=1, **kwargs):
        self.depth = 0

        # all references of objects added are kept in _added_tuple
        # in order to avoid circular reference
        self._triple_added_reference = {}

        # it seems the reference must be added to a map
        # or the id gets recycled, and collision will occur
        self._added_reference = {}

        self.references = []
        self.debug = debug
        self.nested = nested
        self.depth_threshold=depth_threshold
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, trackback):
        pass

    def _serialize(self, obj: Union[None, str, int, float, list, tuple, dict, SiibraSerializable], depth=0, **kwargs):
        
        # if primitive, return original representation
        if obj is None or type(obj) == str or type(obj) == int or type(obj) == float:
            return obj

        # if empty list/tuple, simpley return an empty list
        # otherwise, causes false positive Circular reference
        if (type(obj) == list or type(obj) == tuple) and len(obj) == 0:
            return list(obj)

        if id(obj) in self._triple_added_reference:
            raise CircularReferenceException(f'str {obj} has been referenced three times. Terminating.')

        double_take_flag=id(obj) in self._added_reference
        self._added_reference[id(obj)] = obj

        # if list or tuple, return as a list
        if type(obj) == list or type(obj) == tuple:
            if double_take_flag:
                raise CircularReferenceException(f'obj {str(obj)} has a circular reference. Terminating.')
            return [ self._serialize(item, depth=depth, **kwargs) for item in list(obj) ]

        if type(obj) == dict:
            if double_take_flag:
                raise CircularReferenceException(f'obj {str(obj)} has a circular reference. Terminating.')
            return {
                key: self._serialize(obj[key], depth=depth, **kwargs) for key in obj
            }
        
        if isinstance(obj, SiibraSerializable):
            if double_take_flag:
                kwargs['detail'] = False
                self._triple_added_reference[id(obj)] = obj

            next_depth = depth + 1
            obj_json = obj.to_json(**kwargs)
            serialized_obj= self._serialize(
                obj_json,
                depth=next_depth,
                **kwargs)

            # set an id, if one does not exist
            serialized_obj['@id'] = serialized_obj.get('@id') or str(uuid4())

            above_threshold_flag = depth > self.depth_threshold
            if above_threshold_flag:
                return {
                    '@id': serialized_obj.get('@id'),
                    '@type': serialized_obj.get('@type')
                }
            
            if self.nested or depth == 0:
                return serialized_obj

            if not above_threshold_flag:
                self.references.append(serialized_obj)

            return {
                '@id': serialized_obj.get('@id'),
                '@type': serialized_obj.get('@type')
            }
                

        raise UnJSONableException(f'object with type {type(obj)} cannot be json serialized')


    def get_json(self, obj: Type[SiibraSerializable], detail=False, **kwargs):
        try:
            payload = self._serialize(obj, detail=detail, **kwargs)
            if self.nested:
                return payload
            else:
                return {
                    'payload': payload,
                    'references': self.references
                }
            
        except CircularReferenceException as e:
            if self.debug:
                print(f'args')
            raise e
