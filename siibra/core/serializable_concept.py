from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
import numpy as np
import zlib
import base64
from pydantic import Field

from ..openminds.base import ConfigBaseModel

class JSONSerializable(ABC):

    @abstractclassmethod
    def get_model_type(Cls) -> str:
        """
        The get_modal_type abstract class method should populate the @type attribute of the model returned by to_model
        It should also allow unified indexing of a list of JSONSerializable according to types without first calling to_model method (which could be expensive)

        e.g.

        without get_modal_type abstract class method:
        # filter features based on type
        found_feature = [feat for feat in features if feat.to_model().type == feature_id]

        with get_modal_type abstract class method:
        # finding a feature based on feature_id
        found_feature = [feat for feat in features if feat.get_modal_type() == feature_id]
        """
        raise AttributeError("get_model_type class method needs to be overwritten by subclass!")

    @abstractproperty
    def model_id(self) -> str:
        """
        The model_id abstract property should populate the @id attribute of the model returned by to_model
        It should also allow unified indexing of a list of JSONSerializable without first calling to_model method (which could be expensive)

        e.g.

        without model_id abstract property:
        # finding a feature based on feature_id
        found_feature = [feat for feat in features if feat.model_id == feature_id]

        with model_id abstract property:
        # finding a feature based on feature_id
        found_feature = [feat for feat in features if feat.model_id == feature_id]
        """
        raise AttributeError("model_id property needs to be overwritten by subclass!")

    @abstractmethod
    def to_model(self, **kwargs):
        raise AttributeError("JSONSerializable needs to have to_model overwritten")
    
    def __init_subclass__(cls) -> None:
        assert cls.to_model.__annotations__.get("return"), f"{cls}: JSONSerializable to_model overwritten method needs to be typed."
        return super().__init_subclass__()


class NpArrayDataModel(ConfigBaseModel):
    
    content_type: str = Field("application/octet-stream")
    content_encoding: str = Field("gzip; base64")
    x_width: int = Field(..., alias="x-width")
    x_height: int = Field(..., alias="x-height")
    x_channel: int = Field(..., alias="x-channel")
    dtype: str
    content: str

    def __init__(self, np_data=None, **data) -> None:

        if np_data is None:
            return super().__init__(**data)

        # try to avoid 64 bit any number
        supported_dtype = [
            np.dtype("uint8"),
            np.dtype("int32"),
            np.dtype("float32"),
        ]
        assert type(np_data) is np.ndarray, f"expect input to be numpy array"
        assert np_data.dtype in supported_dtype, f"can only serialize {','.join([str(dtype) for dtype in supported_dtype])} for now"
        assert len(np_data.shape) <= 3, f"can only deal np array with up to 3 dimension"
        
        x_channel = 1 if len(np_data.shape) < 3 else np_data.shape[2]
        x_width = np_data.shape[0]
        x_height = 1 if len(np_data.shape) < 2 else np_data.shape[1]
        dtype = str(np_data.dtype)
        content = base64.b64encode(zlib.compress(np_data.tobytes(order="F")))

        super().__init__(
            x_width=x_width,
            x_height=x_height,
            dtype=dtype,
            content=content,
            x_channel=x_channel,
        )