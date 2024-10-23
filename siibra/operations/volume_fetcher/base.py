from typing import Dict, Callable, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...commons.logger import logger

RegisterFormatRead = Callable[[Dict, List[Dict]], List[Dict]]


# TODO (ASAP) re: datarecipe rewrite, consider how to implement on_init, on_append etc
class VolumeFormats:
    IMAGE_FORMATS = []
    MESH_FORMATS = []
    FORMAT_LOOKUP = {
        None: IMAGE_FORMATS + MESH_FORMATS,  # TODO maybe buggy
        "mesh": MESH_FORMATS,
        "image": IMAGE_FORMATS,
    }
    READER_LOOKUP: Dict[str, RegisterFormatRead] = {}

    @classmethod
    def register_format_read(cls, format: str, voltype: Literal["mesh", "image"]):
        """
        Registers image/mesh formats, and the DataOps generated based on ImageRecipe.

        The wrapped function should expect two positional arguments:

        - dict of the ImageRecipe
        - list of default DataOp specification generated (generated from .url, .archive_options etc)

        The wrapped function should return a list of DataOp specifications (List[Dict])
        """

        def outer(fn: RegisterFormatRead):
            if format in cls.READER_LOOKUP:
                logger.warning(
                    f"{format} already registered by {cls.READER_LOOKUP[format].__name__}, overriden by {Cls.__name__}"
                )
            cls.READER_LOOKUP[format] = fn
            if voltype == "mesh":
                cls.MESH_FORMATS.append(format)
            if voltype == "image":
                cls.IMAGE_FORMATS.append(format)
            cls.FORMAT_LOOKUP[None] = cls.IMAGE_FORMATS + cls.MESH_FORMATS
            return fn

        return outer
