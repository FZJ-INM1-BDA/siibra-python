from typing import List

def check_duplicate(list_of_str: List[str]):
    seen = set()
    duplicate = set()
    for item in list_of_str:
        if item in seen:
            duplicate.add(item)
        seen.add(item)
    return duplicate
