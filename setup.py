from setuptools import setup, find_packages
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR,"VERSION"),"r") as f:
    version = f.read()
with open(os.path.join(ROOT_DIR,"README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="siibra",
    version=version,
    author="Big Data Analytics Group, Forschungszentrum Juelich, Institute of Neuroscience and Medicine (INM-1)",
    author_email="inm1-bda@fz-juelich.de",
    description="siibra - Software interfaces for interacting with brain atlases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FZJ-INM1-BDA/siibra-python",
    packages=find_packages(include=['siibra', 'siibra.*']),
    include_package_data=True,
    # packages=find_packages(include=['.']),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.6',
    install_requires=['anytree','statsmodels','nibabel','click','appdirs','scikit-image','scipy','statsmodels','requests','python-gitlab','memoization','cloud-volume','nilearn'],
)

