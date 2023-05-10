from setuptools import setup, find_packages
import os
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_version():
    path_to_init = os.path.join(ROOT_DIR, "siibra", "__init__.py")
    with open(path_to_init, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(r"^__version__ *= *['\"](.*?)['\"]$", content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Siibra version cannot be found.")


with open(os.path.join(ROOT_DIR, "README.rst"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="siibra",
    version=find_version(),
    author="Big Data Analytics Group, Forschungszentrum Juelich, Institute of Neuroscience and Medicine (INM-1)",
    author_email="inm1-bda@fz-juelich.de",
    description="siibra - Software interfaces for interacting with brain atlases",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/FZJ-INM1-BDA/siibra-python",
    packages=find_packages(include=["siibra", "siibra.*"]),
    include_package_data=True,
    package_data={'siibra': ['features/region_aliases_human.json']},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.6",
    install_requires=[
        "anytree",
        "nibabel",
        "appdirs",
        "scikit-image",
        "requests",
        "memoization",
        "neuroglancer-scripts",
        "nilearn",
        'importlib-resources; python_version < "3.7"',
        'typing-extensions; python_version < "3.8"',
        "pydantic==1.8.2",
    ],
)
