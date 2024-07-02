from dataclasses import dataclass
from typing import DefaultDict, Literal, Union

from ..base import Data
from ...commons_new.string import check_color, SUPPORTED_COLORMAPS


@dataclass
class Volume(Data):
    schema: str = None
    space_id: str = None
    format: str = None
    url: str = None
    color: str = None
    volume_selection_options: DefaultDict[Literal["label", "z", "t"], Union[int, str]] = None

    def __post_init__(self):
        if self.color and not check_color(self.color):
            print(
                f"'{self.color}' is not a hex color or as supported colormap ({SUPPORTED_COLORMAPS=})"
            )
