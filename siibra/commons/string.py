# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Union, Callable, List, Dict


def splitstr(s: str):
    return [w for w in re.split(r"[^a-zA-Z0-9.]", s) if len(w) > 0]


def fuzzy_match(input: str, dest: str):
    """Fuzzy string search.

    n.b. the comparison is asymetrical, that is, match(a, b) may be not the same as match(b, a)

    Parameters
    ----------
    input: str
    dest: str

    Returns
    -------
    bool
        If the fuzzy matches
    """
    if input.lower().strip() == dest.lower().strip():
        return True

    W = splitstr(dest.lower())
    Q = splitstr(input.lower())
    return all(
        any(
            q == w or "v" + q == w
            # or q in w # required to match "julich" to "Julich-Brain"
            for w in W
        )
        for q in Q
    )


def snake2camel(s: str):
    """Converts a string in snake_case into CamelCase.
    For example: JULICH_BRAIN -> JulichBrain"""
    return "".join([w[0].upper() + w[1:].lower() for w in s.split("_")])


REMOVE_FROM_NAME = [
    "hemisphere",
    " -",
    "-brain",
    "both",
    "Both",
]

REPLACE_IN_NAME = {
    "ctx-lh-": "left ",
    "ctx-rh-": "right ",
}


def clear_name(name: str):
    """clean up a region name to the for matching"""
    result = name
    for word in REMOVE_FROM_NAME:
        result = result.replace(word, "")
    for search, repl in REPLACE_IN_NAME.items():
        result = result.replace(search, repl)
    return " ".join(w for w in result.split(" ") if len(w))


REGEX_STR = re.compile(r"^\/(?P<expression>.+)\/(?P<flags>[a-zA-Z]*)$")
ACCEPTED_FLAGS = "aiLmsux"

SPEC_TYPE = Union[str, re.Pattern]


def get_spec(spec: SPEC_TYPE) -> Callable[[str], bool]:
    if isinstance(spec, re.Pattern):
        return spec.search
    if isinstance(spec, str):
        regex_match = REGEX_STR.match(spec)
        if regex_match:
            flags = regex_match.group("flags")
            expression = regex_match.group("expression")

            for flag in flags or []:  # catch if flags is nullish
                if flag not in ACCEPTED_FLAGS:
                    raise Exception(
                        f"only accepted flag are in {ACCEPTED_FLAGS}. {flag} is not within them"
                    )
            search_regex_str = (f"(?{flags})" if flags else "") + expression
            search_regex = re.compile(search_regex_str)
            return search_regex.search
        return lambda input_str: fuzzy_match(spec, input_str)
    print(spec, type(spec))
    raise RuntimeError("get_spec only accept str or re.Pattern as input")


def create_key(name: str):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return re.sub(
        r" +",
        "_",
        "".join([e if e.isalnum() else " " for e in name]).upper().strip(),
    )


HEX_COLOR_REGEXP = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
SUPPORTED_COLORMAPS = {"magma", "jet", "rgb"}


def is_hex_color(color: str) -> bool:
    return True if HEX_COLOR_REGEXP.search(color) else False


def check_color(color: str) -> bool:
    if color in SUPPORTED_COLORMAPS or is_hex_color(color):
        return True
    return False


def to_hex(color: List[int]) -> str:
    assert len(color) == 3, f"expected 3 component to color, but got {len(color)!r}"
    assert all(
        (isinstance(c, int) and c <= 255 and c >= 0 for c in color)
    ), f"expected {color!r} to be all uint8, but is not"
    return "#" + "".join([hex(c)[2:].rjust(2, "0") for c in color])


def convert_hexcolor_to_rgbtuple(clr: str):
    return tuple(int(clr[p : p + 2], 16) for p in [1, 3, 5])


def extract_uuid(long_id: Union[str, Dict]) -> str:
    """
    Returns the uuid portion of either a fully formed openminds id, or get
    the 'id' property first, and extract the uuid portion of the id.

    Parameters
    ----------
    long_id: str, dict[str, str]

    Returns
    -------
    str

    Raises
    ------
    AssertionError
    RuntimeError
    """
    if isinstance(long_id, str):
        pass
    elif isinstance(long_id, dict):
        long_id = long_id.get("id")
        assert isinstance(long_id, str)
    else:
        raise RuntimeError("uuid arg must be str or object")
    uuid_search = re.search(r"(?P<uuid>[a-f0-9-]+)$", long_id)
    assert uuid_search, "uuid not found"
    return uuid_search.group("uuid")
