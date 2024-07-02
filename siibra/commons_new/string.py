import re
from typing import Union, Callable


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
