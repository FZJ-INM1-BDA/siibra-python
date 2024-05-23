import re

def splitstr(s: str):
    return [w for w in re.split(r"[^a-zA-Z0-9.\-]", s) if len(w) > 0]


def fuzzy_match(input: str, dest: str):
    """Fuzzy string search.

    n.b. the comparison is assymetrical, that is, match(a, b) may be not the same as match(b, a)
    
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
            q == w or 'v' + q == w
            for w in W
        ) for q in Q
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
    """ clean up a region name to the for matching"""
    result = name
    for word in REMOVE_FROM_NAME:
        result = result.replace(word, "")
    for search, repl in REPLACE_IN_NAME.items():
        result = result.replace(search, repl)
    return " ".join(w for w in result.split(" ") if len(w))
