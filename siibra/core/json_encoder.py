from typing import Type, Union

from siibra.core.concept import JSONableConcept

class CircularReferenceException(Exception): pass
class UnJSONableException(Exception): pass

class JSONEncoder:
    def __init__(self, debug=False, **kwargs):

        # all references of objects added are kept in _added_set
        # in order to avoid circular reference
        self._triple_set = set()

        # it seems the reference must be added to a map
        # or the id gets recycled, and collision will occur
        self._added_map = {}

        self.debug = debug
        
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, trackback):
        pass

    def _serialize(self, obj: Union[None, str, int, float, list, tuple, dict, Type[JSONableConcept]], **kwargs):
        
        # if primitive, return original representation
        if obj is None or type(obj) == str or type(obj) == int or type(obj) == float:
            return obj

        # if empty list/tuple, simpley return an empty list
        # otherwise, causes false positive Circular reference
        if (type(obj) == list or type(obj) == tuple) and len(obj) == 0:
            return list(obj)

        if id(obj) in self._triple_set:
            raise CircularReferenceException(f'str {obj} has been referenced three times. Terminating.')
        double_take_flag=id(obj) in self._added_map
        self._added_map[id(obj)] = obj

        # if list or tuple, return as a list
        if type(obj) == list or type(obj) == tuple:
            if double_take_flag:
                raise CircularReferenceException(f'obj {str(obj)} has a circular reference. Terminating.')
            return [ self._serialize(item, **kwargs) for item in list(obj) ]

        if type(obj) == dict:
            if double_take_flag:
                raise CircularReferenceException(f'obj {str(obj)} has a circular reference. Terminating.')
            return {
                key: self._serialize(obj[key], **kwargs) for key in obj
            }
        
        if isinstance(obj, JSONableConcept):
            if double_take_flag:
                kwargs['detail'] = False
                self._triple_set.add(id(obj))
            return self._serialize(
                obj.to_json(**kwargs),
                **kwargs)

        raise UnJSONableException(f'object with type {type(obj)} cannot be json serialized')


    def get_json(self, obj: Type[JSONableConcept], **kwargs):
        try:
            return self._serialize(obj, **kwargs)
        except CircularReferenceException as e:
            if self.debug:
                print(f'args')
            raise e
