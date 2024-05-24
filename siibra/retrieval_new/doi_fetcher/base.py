content_type_registry = {}


def register_content_type(content_type: str):
    def outer(fn):
        content_type_registry[content_type] = fn
        return fn

    return outer
