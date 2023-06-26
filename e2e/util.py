from typing import List

def check_duplicate(list_of_str: List[str], fn: lambda a: a):
    seen = set()
    duplicate = set()
    for item in list_of_str:
        if fn(item) in seen:
            duplicate.add(item)
        seen.add(fn(item))
    return duplicate
