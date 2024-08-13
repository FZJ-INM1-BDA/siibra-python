from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))

path_to_version = path.join(ROOT_DIR, "VERSION")
with open(path_to_version, "r", encoding="utf-8") as version_file:
    __version__ = version_file.read().strip()
