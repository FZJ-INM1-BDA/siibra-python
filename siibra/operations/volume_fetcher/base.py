from typing import Dict, List, TYPE_CHECKING, Type

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from ...attributes.dataproviders.volume import VolumeProvider

from ...operations.base import DataOp
from ...commons.logger import logger


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


class PostProcVolProvider:

    @classmethod
    def on_post_init(cls, volume_provider: "VolumeProvider"):
        pass

    @classmethod
    def transform_retrieval_ops(
        cls, volume_provider: "VolumeProvider", base_retrieval_ops: List[Dict]
    ):
        return base_retrieval_ops

    @classmethod
    def on_append_op(cls, volume_provider: "VolumeProvider", op: Dict):
        from ...attributes.dataproviders.volume import VolumeProvider

        return super(VolumeProvider, volume_provider).append_op(op)
