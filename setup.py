from setuptools import setup, find_packages
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_version():
    path_to_version = os.path.join(ROOT_DIR, "siibra", "VERSION")
    with open(path_to_version, "r", encoding="utf-8") as f:
        return f.read()


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
    package_data={
        'siibra': [
            'VERSION',
            'vocabularies/gene_names.json',
            'vocabularies/receptor_symbols.json',
            'vocabularies/region_aliases.json'
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    # Lower bounds only: this is a library, so exact pins would clash with other
    # packages in users' environments. Bounds exclude versions with known
    # vulnerabilities where possible. The fixed requests/filelock releases need
    # Python >= 3.10, so Python 3.9 gets the best versions available to it.
    install_requires=[
        "anytree >= 2.12.1",
        "nibabel >= 5.3.2",
        "appdirs >= 1.4.4",
        "scikit-image >= 0.24.0",
        'requests >= 2.33.0; python_version >= "3.10"',
        'requests >= 2.32.4; python_version < "3.10"',
        "neuroglancer-scripts >= 1.2.0",
        "nilearn >= 0.12",
        'filelock >= 3.20.3; python_version >= "3.10"',
        'filelock >= 3.16.1; python_version < "3.10"',
        "ebrains-drive >= 0.6.0",
        "h5py >= 3.11",
    ],
)
