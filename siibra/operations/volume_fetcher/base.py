from typing import Dict, List, TYPE_CHECKING, Type

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from ...attributes.dataproviders.volume import VolumeRecipe

from ...operations.base import DataOp
from ...commons.logger import logger


# TODO (ASAP) re: datarecipe rewrite, consider how to implement on_init, on_append etc
class VolumeFormats:
    IMAGE_FORMATS = []
    MESH_FORMATS = []
    FORMAT_LOOKUP = {
        None: IMAGE_FORMATS + MESH_FORMATS,  # TODO maybe buggy
        "mesh": MESH_FORMATS,
        "image": IMAGE_FORMATS,
    }
    READER_LOOKUP: Dict[str, Type["PostProcVolProvider"]] = {}

    @classmethod
    def register_format_read(cls, format: str, voltype: Literal["mesh", "image"]):

        def outer(Cls: Type["PostProcVolProvider"]):
            assert issubclass(
                Cls, PostProcVolProvider
            ), f"register_format_read must target a subclass of volume ret op"

            if format in cls.READER_LOOKUP:
                logger.warning(
                    f"{format} already registered by {cls.READER_LOOKUP[format].__name__}, overriden by {Cls.__name__}"
                )
            cls.READER_LOOKUP[format] = Cls
            if voltype == "mesh":
                cls.MESH_FORMATS.append(format)
            if voltype == "image":
                cls.IMAGE_FORMATS.append(format)
            cls.FORMAT_LOOKUP[None] = cls.IMAGE_FORMATS + cls.MESH_FORMATS
            return Cls

        return outer


# TODO (ASAP) remove, but consider how it should be implemented in lieu with the new datarecipe paradigm
class PostProcVolProvider:

    @classmethod
    def on_post_init(cls, volume_provider: "VolumeRecipe"):
        pass

    @classmethod
    def on_get_retrieval_ops(cls, volume_provider: "VolumeRecipe"):
        from ...attributes.dataproviders.volume import VolumeRecipe

        return super(VolumeRecipe, volume_provider).retrieval_ops

    @classmethod
    def on_append_op(cls, volume_provider: "VolumeRecipe", op: Dict):
        from ...attributes.dataproviders.volume import VolumeRecipe

        return super(VolumeRecipe, volume_provider).append_op(op)
